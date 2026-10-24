"""Golden regression: single-process training must keep producing the numbers it produced before the scale-up work.

The losses below were produced by the code at commit 9f05cd8 (before multi-GPU, weighted mixing, atomic saves) and are
identical to the current code's (the weights' SHA-256 matched bit for bit on the author's machine). If this test fails,
something changed the training path: either that was intended (then re-run the GPU smoke test, record the new numbers and
say so in CHANGELOG.md) or it was a bug.
"""
import json

import numpy as np
import pytest

from llm import data
from llm.train import TrainConfig, run_info, train

GOLDEN_LOSSES = [4.300751925, 1.240278959, 0.59325856, 0.383198336, 0.418656662]


def test_single_process_training_matches_the_pre_scale_up_numbers(tmp_path):
    rng = np.random.RandomState(0)
    V = 64
    ids = np.tile(np.arange(1, 17), 300)
    ids = np.where(rng.rand(len(ids)) < 0.02, rng.randint(1, V, len(ids)), ids)
    data.write_bin(str(tmp_path / "t.bin"), ids, vocab_size=V, eos_id=V - 1)
    data.write_bin(str(tmp_path / "v.bin"), ids[:2000], vocab_size=V, eos_id=V - 1)
    r = train(TrainConfig(train_bin=str(tmp_path / "t.bin"), val_bin=str(tmp_path / "v.bin"), out_dir=str(tmp_path / "run"),
                          preset="tiny", seq_len=32, batch_size=8, grad_accum=2, max_steps=25, warmup_steps=3,
                          eval_interval=25, save_interval=1000, log_interval=5, lr=3e-3, device="cpu", dtype="fp32", seed=7))
    got = [h["loss"] for h in r["history"] if "loss" in h]
    assert got == pytest.approx(GOLDEN_LOSSES, rel=1e-3)
    info = json.loads((tmp_path / "run" / "train_config.json").read_text())["run_info"]
    assert info["torch"] and info["python"] and "git_commit" in info and set(run_info()) == set(info)
