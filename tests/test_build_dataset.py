"""scripts/build_dataset.py: one config -> capped tokenization of Parquet and text domains, merged, checked, mix printed."""
import json
import sys
from pathlib import Path

import pytest

pa = pytest.importorskip("pyarrow")
import pyarrow.parquet as pq  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import build_dataset as bd  # noqa: E402
from llm.bpe_trainer import train_bpe, write_tok  # noqa: E402
from llm.data import read_header  # noqa: E402

BIO = [f"Protein {i} binds receptor {i % 7} in tissue {i % 5} and regulates the cell cycle." for i in range(300)]
QA = [f"Question: what is {i} plus {i}?\nAnswer: {2 * i}" for i in range(200)]


def test_config_to_checked_bins_and_the_printed_mix(tmp_path, capsys):
    vocab = tmp_path / "v.tok"
    write_tok(train_bpe((BIO + QA) * 3, 330), str(vocab))
    for i in range(4):
        part = BIO[i * 75:(i + 1) * 75]
        pq.write_table(pa.table({"doc_id": [f"b{i}-{j}" for j in range(75)], "domain": ["bio"] * 75, "text": part}), tmp_path / f"bio{i}.parquet")
    (tmp_path / "qa.txt").write_text("\n\n".join(QA) + "\n", encoding="utf-8")
    (tmp_path / "tiny.txt").write_text("a short corpus with only a handful of documents\nanother document here\n", encoding="utf-8")
    cfg = {"vocab": str(vocab), "out": str(tmp_path / "bins"), "workers": 1, "val_frac": 0.1, "plan_tokens": 100000,
           "domains": [{"name": "bio", "kind": "parquet", "paths": [str(tmp_path / "bio*.parquet")], "max_files": 2, "weight": 0.6},
                       {"name": "qa", "kind": "text", "paths": [str(tmp_path / "qa.txt")], "blank": True, "weight": 0.4},
                       {"name": "tiny", "kind": "text", "paths": [str(tmp_path / "tiny.txt")], "weight": 0.01}]}
    (tmp_path / "cfg.json").write_text(json.dumps(cfg), encoding="utf-8")
    assert bd.main([str(tmp_path / "cfg.json"), "--allow-unfrozen"]) == 0
    out = capsys.readouterr().out
    bins = tmp_path / "bins"
    assert read_header(str(bins / "bio.train.bin")).total_tokens > 0 and (bins / "qa.val.bin").exists()
    assert not (bins / "work").exists() or not any((bins / "work").iterdir())            # intermediates removed
    assert "--train-mix" in out and "bio.train.bin=0.6" in out and "RESULT: PASS" in out
    # only the first 2 of the 4 bio files were used
    total_bio_tokens = read_header(str(bins / "bio.train.bin")).total_tokens + read_header(str(bins / "bio.val.bin")).total_tokens
    assert 0 < total_bio_tokens < 1.2 * sum(len(t) for t in BIO[:150])


def test_a_missing_input_stops_the_build(tmp_path):
    vocab = tmp_path / "v.tok"
    write_tok(train_bpe(BIO * 3, 300), str(vocab))
    (tmp_path / "cfg.json").write_text(json.dumps({"vocab": str(vocab), "out": str(tmp_path / "b"), "domains": [
        {"name": "x", "kind": "text", "paths": [str(tmp_path / "nothing*.txt")], "weight": 1}]}), encoding="utf-8")
    with pytest.raises(SystemExit):
        bd.main([str(tmp_path / "cfg.json"), "--allow-unfrozen"])


def test_the_shipped_example_config_is_valid_json_with_known_kinds():
    cfg = json.loads((ROOT / "configs" / "run_100m.example.json").read_text(encoding="utf-8"))
    assert {d["kind"] for d in cfg["domains"]} <= {"text", "parquet"}
    assert len({d["name"] for d in cfg["domains"]}) == len(cfg["domains"]) and all(d["weight"] > 0 for d in cfg["domains"])
    assert (ROOT / cfg["vocab"]).is_file()


def test_partial_config_uses_weights_sum_and_writes_a_build_manifest(tmp_path, capsys):
    vocab = tmp_path / "v.tok"
    write_tok(train_bpe(BIO * 3, 330), str(vocab))
    (tmp_path / "bio.txt").write_text("\n".join(BIO * 5) + "\n", encoding="utf-8")
    cfg = {"vocab": str(vocab), "out": str(tmp_path / "bins"), "workers": 1, "val_frac": 0.1, "plan_tokens": 100000, "weights_sum": 0.5,
           "domains": [{"name": "bio", "kind": "text", "paths": [str(tmp_path / "bio.txt")], "weight": 0.25}]}
    (tmp_path / "cfg.json").write_text(json.dumps(cfg), encoding="utf-8")
    assert bd.main([str(tmp_path / "cfg.json"), "--allow-unfrozen"]) == 0
    out = capsys.readouterr().out
    total = read_header(str(tmp_path / "bins" / "bio.train.bin")).total_tokens
    assert f"{0.25 / 0.5:>9.3f}" in out                                   # share 0.5 of the whole mix, not 1.0 of this config
    m = json.loads((tmp_path / "bins" / "build_manifest.json").read_text())
    assert m["tokens"]["bio"]["train"] == total and len(m["tokenizer_sha256"]) == 64 and m["config"]["weights_sum"] == 0.5
    assert m["inputs"]["bio"][0]["bytes"] == (tmp_path / "bio.txt").stat().st_size and "git_commit" in m["run_info"]


def test_a_vocabulary_other_than_the_frozen_one_is_refused_before_any_work(tmp_path):
    digits = ROOT / "tokenizers" / "vocab32k_digits_v2" / "digits_NOT_FROZEN.tok"      # same size and EOS id as the frozen one
    (tmp_path / "t.txt").write_text("\n".join(BIO) + "\n", encoding="utf-8")
    cfg = {"vocab": str(digits), "out": str(tmp_path / "bins"),
           "domains": [{"name": "t", "kind": "text", "paths": [str(tmp_path / "t.txt")], "weight": 1}]}
    (tmp_path / "cfg.json").write_text(json.dumps(cfg), encoding="utf-8")
    with pytest.raises(SystemExit) as e:
        bd.main([str(tmp_path / "cfg.json")])
    assert "NOT the frozen tokenizer" in str(e.value) and not (tmp_path / "bins").exists()
    cfg["vocab"] = str(ROOT / "tokenizers" / "vocab32k_v2" / "vocab32k.tok")
    bd.require_frozen_vocab(cfg["vocab"])                                              # the frozen one is accepted
