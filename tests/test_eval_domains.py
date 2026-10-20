"""scripts/eval_domains.py: per-domain validation loss from a trained checkpoint."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import eval_domains as ed  # noqa: E402
from llm import data  # noqa: E402
from llm.train import TrainConfig, train  # noqa: E402

VOCAB = 64


def _bin(path, n, seed):
    rng = np.random.RandomState(seed)
    ids = np.tile(np.arange(1, 17), n // 16 + 1)[:n]
    ids = np.where(rng.rand(n) < 0.02, rng.randint(1, VOCAB, n), ids)
    data.write_bin(str(path), ids, vocab_size=VOCAB, eos_id=VOCAB - 1)
    return str(path)


def test_per_domain_losses_are_finite_deterministic_and_tiny_files_are_reported(tmp_path):
    tr, v1, v2, tiny = (_bin(tmp_path / f"{n}.bin", k, i) for i, (n, k) in enumerate(
        [("train", 4000), ("a", 2000), ("b", 2000), ("tiny", 20)]))
    train(TrainConfig(train_bin=tr, val_bin=v1, out_dir=str(tmp_path / "run"), preset="tiny", seq_len=32,
                      batch_size=8, max_steps=30, warmup_steps=3, eval_interval=1000, save_interval=1000,
                      lr=3e-3, device="cpu", dtype="fp32"))
    ck = str(tmp_path / "run" / "ckpt_last.pt")
    vals = {"a": v1, "b": v2, "tiny": tiny}
    r1 = ed.evaluate_domains(ck, vals, 32, 4, 10, device="cpu", dtype="fp32")
    r2 = ed.evaluate_domains(ck, vals, 32, 4, 10, device="cpu", dtype="fp32")
    assert r1["ok"] and r1["domains"] == r2["domains"]
    assert r1["domains"]["a"]["val_loss"] < np.log(VOCAB)              # learned something vs uniform
    assert r1["domains"]["tiny"]["windows"] == 0 and r1["domains"]["tiny"]["val_loss"] is None
    out = tmp_path / "e.json"
    assert ed.main(["--ckpt", ck, "--val", f"a={v1}", f"b={v2}", "--seq-len", "32", "--device", "cpu",
                    "--dtype", "fp32", "--out", str(out)]) == 0
    assert set(json.loads(out.read_text())["domains"]) == {"a", "b"}
