"""scripts/sample_outputs.py: real inputs and outputs, held-out accuracy and failure flags from one checkpoint."""
import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("pyarrow")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import build_dataset as bd  # noqa: E402
import sample_outputs as so  # noqa: E402
import torch  # noqa: E402
from llm.checkpoint import save_checkpoint  # noqa: E402
from llm.model import TransformerModel  # noqa: E402
from llm.model_config import ModelConfig  # noqa: E402
from llm.tokenizer import BPETokenizer  # noqa: E402
from llm.train import PRESETS  # noqa: E402

VOCAB = ROOT / "tokenizers" / "vocab32k_v2" / "vocab32k.tok"


def test_repetition_loop_and_flags():
    assert so.repetition_loop([5, 6, 7] * 20) and not so.repetition_loop(list(range(60)))
    assert not so.repetition_loop([1, 2, 3])                                   # too short to judge
    assert so.flags([1, 2], [], "") == ["empty"]
    assert "invalid UTF-8" in so.flags([1], [2, 3], "ab�cd")
    assert "invalid UTF-8" not in so.flags([1], [2, 3], "abcd�")          # a cut at the very end is not a defect
    assert "copies the prompt" in so.flags(list(range(10)), list(range(2, 12)), "x")


def test_the_report_has_every_part_for_a_random_model(tmp_path):
    docs = {n: [f"{n} sample {i}: the value of {i} times {i % 5 + 2} is {i * (i % 5 + 2)} and nothing else changes here." for i in range(300)]
            for n in ("alpha", "beta")}
    for n, d in docs.items():
        (tmp_path / f"{n}.txt").write_text("\n".join(d) + "\n", encoding="utf-8")
    cfg = {"vocab": str(VOCAB), "out": str(tmp_path / "bins"), "workers": 1, "val_frac": 0.2,
           "domains": [{"name": n, "kind": "text", "paths": [str(tmp_path / f"{n}.txt")], "weight": 0.5} for n in docs]}
    (tmp_path / "c.json").write_text(json.dumps(cfg), encoding="utf-8")
    assert bd.main([str(tmp_path / "c.json")]) == 0
    mc = ModelConfig(**{**PRESETS["tiny"], "vocab_size": 32000})
    torch.manual_seed(0)
    model = TransformerModel(mc)
    save_checkpoint(str(tmp_path / "m.pt"), model, mc, step=7, vocab_sha256=BPETokenizer.file_sha256(str(VOCAB)))
    rc = so.main(["--ckpt", str(tmp_path / "m.pt"), "--vocab", str(VOCAB), "--out", str(tmp_path / "rep"),
                  "--val", f"alpha={tmp_path / 'bins' / 'alpha.val.bin'}", f"beta={tmp_path / 'bins' / 'beta.val.bin'}",
                  "--per-domain", "2", "--prompt-tokens", "16", "--new-tokens", "12", "--probe-tokens", "6", "--seq-len", "32", "--batches", "2"])
    assert rc == 0
    r = json.loads((tmp_path / "rep.json").read_text(encoding="utf-8"))
    assert r["step"] == 7 and set(r["domains"]) == {"alpha", "beta"} and len(r["probes"]) == len(so.PROBES)
    s = r["domains"]["alpha"]["samples"][0]
    assert s["prompt"] and set(s) >= {"model", "truth", "flags", "token_agreement"} and 0.0 <= r["domains"]["alpha"]["top1"] <= 1.0
    md = (tmp_path / "rep.md").read_text(encoding="utf-8")
    assert "Fixed probes" in md and "Held-out continuations" in md and "Next-token accuracy" in md
    assert r["summary"]["probes_passed"] <= r["summary"]["probes"]
