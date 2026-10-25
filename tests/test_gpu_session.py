"""scripts/gpu_session.py: the whole GPU session (preflight, benchmark, smoke test, timed run, evaluation) rehearsed on CPU
with the real frozen tokenizer, plus the balanced validation file that ``llm.train --val-bin`` needs."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pyarrow")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import build_dataset as bd  # noqa: E402
import gpu_session as gs  # noqa: E402
import merge_bins as mb  # noqa: E402
from llm.data import read_header, read_tokens, write_bin  # noqa: E402
from llm.tokenizer import BPETokenizer  # noqa: E402

VOCAB = ROOT / "tokenizers" / "vocab32k_v2" / "vocab32k.tok"


def _docs(tag, n):
    return [f"{tag} document {i}: the value of {i} times {i % 7 + 2} is {i * (i % 7 + 2)} and the sample {tag} stays stable." for i in range(n)]


def test_capped_mix_is_balanced_and_ends_on_document_boundaries(tmp_path):
    eos = 5
    a = np.array(([1, 2, 3, eos] * 100), dtype=np.int64)               # 400 tokens
    b = np.array(([7, 8, eos] * 10), dtype=np.int64)                     # 30 tokens (under the cap)
    write_bin(str(tmp_path / "a.val.bin"), a, 100, eos, 1)
    write_bin(str(tmp_path / "b.val.bin"), b, 100, eos, 1)
    taken = mb.capped_mix([str(tmp_path / "a.val.bin"), str(tmp_path / "b.val.bin")], str(tmp_path / "mix.bin"), 50)
    assert list(taken.values()) == [48, 30]                              # 50 cut back to the last whole document (12 x 4)
    h, toks = read_tokens(str(tmp_path / "mix.bin"))
    assert h.total_tokens == 78 and int(toks[-1]) == eos and h.split == 1


def test_capped_mix_keeps_one_long_document_whole_and_refuses_mixed_vocabularies(tmp_path):
    eos = 5
    write_bin(str(tmp_path / "long.bin"), np.array([1] * 70 + [eos], dtype=np.int64), 100, eos, 1)
    taken = mb.capped_mix([str(tmp_path / "long.bin")], str(tmp_path / "o.bin"), 20)
    assert list(taken.values()) == [71]
    write_bin(str(tmp_path / "other.bin"), np.array([1, 2, 3], dtype=np.int64), 200, 3, 1)
    with pytest.raises(ValueError):
        mb.capped_mix([str(tmp_path / "long.bin"), str(tmp_path / "other.bin")], str(tmp_path / "x.bin"), 20)


def test_frozen_hash_is_readable_and_equals_the_vocabulary_file():
    assert gs.frozen_sha() == BPETokenizer.file_sha256(str(VOCAB))


def test_best_row_prefers_speed_divides_the_batch_and_leaves_memory_headroom():
    rows = [{"preset": "p", "batch": 4, "status": "ok", "tokens_per_s": 100, "peak_mem_GB": 10},
            {"preset": "p", "batch": 8, "status": "ok", "tokens_per_s": 150, "peak_mem_GB": 30},
            {"preset": "p", "batch": 16, "status": "ok", "tokens_per_s": 160, "peak_mem_GB": 78},
            {"preset": "p", "batch": 32, "status": "OOM"},
            {"preset": "q", "batch": 3, "status": "ok", "tokens_per_s": 999, "peak_mem_GB": 1}]
    assert gs.best_row(rows, "p", 128, 1, 80)["batch"] == 8              # 16 would use 97% of the card
    assert gs.best_row(rows, "p", 128, 1, None)["batch"] == 16
    assert gs.best_row(rows, "q", 128, 1, None) is None                  # 3 does not divide 128
    assert gs.best_row(rows, "p", 128, 1, 8) is None


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("session")
    for name, n in (("alpha", 400), ("beta", 300), ("gamma", 250)):
        (tmp / f"{name}.txt").write_text("\n".join(_docs(name, n)) + "\n", encoding="utf-8")
    cfg = {"vocab": str(VOCAB), "out": str(tmp / "bins"), "workers": 1, "val_frac": 0.1, "plan_tokens": 200000,
           "domains": [{"name": n, "kind": "text", "paths": [str(tmp / f"{n}.txt")], "weight": w}
                       for n, w in (("alpha", 0.5), ("beta", 0.3), ("gamma", 0.2))]}
    (tmp / "build.json").write_text(json.dumps(cfg), encoding="utf-8")
    assert bd.main([str(tmp / "build.json")]) == 0
    full = dict(cfg, domains=cfg["domains"] + [{"name": "ghost", "kind": "text", "paths": ["nothing"], "weight": 0.1}])
    (tmp / "mix.json").write_text(json.dumps(full), encoding="utf-8")
    return tmp


def _args(tmp, *extra):
    return ["--bins", str(tmp / "bins"), "--config", str(tmp / "mix.json"), "--out", str(tmp / "out"), "--cpu-ok",
            "--preset", "tiny", "--smoke-preset", "small", "--seq-len", "64", "--global-seqs", "8", "--minutes", "0.1",
            "--bench-batches", "2", "4", "--smoke-batches", "1", "2", "--bench-steps", "4", "--min-free-gb", "0.01",
            "--val-cap", "1500", "--eval-interval", "40", "--save-interval", "40", "--eval-batches", "3", "--smoke-minutes", "3", *extra]


def test_preflight_passes_on_good_bins_and_names_the_missing_domain(built, capsys):
    assert gs.main(["preflight", *_args(built)]) == 0
    out = capsys.readouterr().out
    assert "PREFLIGHT: PASS" in out and "missing: ghost" in out and "tokenizer = frozen hash" in out
    h = read_header(str(built / "bins" / "val_mix.bin"))
    assert h.total_tokens > 0 and h.split == 1


def test_preflight_fails_when_a_manifest_carries_another_tokenizer_hash(built, tmp_path, capsys):
    import shutil
    bins = tmp_path / "bins"
    shutil.copytree(built / "bins", bins)
    m = json.loads((bins / "build_manifest.json").read_text())
    m["tokenizer_sha256"] = "0" * 64
    (bins / "build_manifest.json").write_text(json.dumps(m))
    args = _args(built)
    args[args.index("--bins") + 1] = str(bins)
    args[args.index("--out") + 1] = str(tmp_path / "out")
    assert gs.main(["preflight", *args]) == 1
    assert "FAIL" in capsys.readouterr().out


def test_preflight_fails_on_the_wrong_tag(built, capsys):
    assert gs.main(["preflight", *_args(built), "--expect-tag", "no-such-tag"]) == 1
    assert "checked out = no-such-tag" in capsys.readouterr().out


def test_the_whole_session_runs_end_to_end_on_cpu(built, capsys):
    rc = gs.main(["all", *_args(built)])
    out = capsys.readouterr().out
    assert rc == 0, out[-3000:]
    d = json.loads((built / "out" / "session.json").read_text())
    assert d["smoke"]["status"] == "PASS" and d["smoke"]["last_loss"] < d["smoke"]["first_loss"]
    assert not list((built / "out" / "smoke").glob("*.pt"))                # the big smoke checkpoints are removed
    t = d["train"]
    assert t["max_steps"] >= 50 and t["exit_code"] == 0 and t["tokens_per_step"] == 8 * 64
    run = built / "out" / "run"
    assert (run / "ckpt_last.pt").exists() and (run / "ckpt_best.pt").exists()
    assert set(d["eval"]["ckpt_last"]["domains"]) == {"alpha", "beta", "gamma"} and d["eval"]["ckpt_last"]["ok"]
    report = (built / "out" / "session_report.md").read_text(encoding="utf-8")
    assert "## Benchmark" in report and "## Smoke test" in report and "## Training" in report and "Per-domain validation" in report
    # re-running the training stage resumes the finished run instead of planning a new one
    steps = t["max_steps"]
    assert gs.main(["train", *_args(built)]) == 0
    assert json.loads((built / "out" / "session.json").read_text())["train"]["max_steps"] == steps
    # a second `all` repeats neither the benchmark nor the smoke test, and keeps the same plan
    capsys.readouterr()
    assert gs.main(["all", *_args(built)]) == 0
    again = capsys.readouterr().out
    assert "bench: already done" in again and "smoke: already done" in again
    assert json.loads((built / "out" / "session.json").read_text())["train"]["max_steps"] == steps


def test_a_failing_smoke_test_stops_before_the_expensive_run(built, tmp_path, capsys, monkeypatch):
    args = _args(built)
    args[args.index("--out") + 1] = str(tmp_path / "out2")
    monkeypatch.setattr(gs, "run_logged", lambda cmd, log, timeout_s=None: 1)          # every training process "crashes"
    rc = gs.main(["all", *args])
    out = capsys.readouterr().out
    assert rc == 1 and "STOPPED at smoke" in out and not (tmp_path / "out2" / "run").exists()
