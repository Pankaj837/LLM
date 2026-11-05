"""Checkpoint reliability (Raju's task 4): rolling backups, validation loss always recorded, graceful stop, auto-resume, corruption."""
import glob
import os
import signal

import pytest
import torch

import llm.train as T
from llm.checkpoint import find_latest_checkpoint, prune_backups, read_checkpoint, verify_checkpoint
from llm.train import train

from test_train import _cfg


def _steps(out_dir):
    return sorted(read_checkpoint(p)[2]["step"] for p in glob.glob(os.path.join(out_dir, "ckpt_step_*.pt")))


def _same_weights(a, b):
    sa, _, _, _ = read_checkpoint(a)
    sb, _, _, _ = read_checkpoint(b)
    return all(torch.equal(sa[k], sb[k]) for k in sa)


def test_last_checkpoint_always_carries_a_validation_loss(tmp_path):
    """The 100M run's step-8,200 checkpoint had val_loss None (saved between evaluations). Now it records the latest one."""
    out = str(tmp_path / "r")
    train(_cfg(tmp_path, out_dir=out, max_steps=25, eval_interval=10, save_interval=7))
    _, _, meta, _ = read_checkpoint(os.path.join(out, "ckpt_last.pt"))
    assert meta["step"] == 25 and meta["val_loss"] is not None
    assert meta["extra"]["best_val"] is not None


def test_checkpoint_is_written_after_every_evaluation(tmp_path):
    out = str(tmp_path / "r")
    seen = []
    real = T.save_checkpoint

    def spy(path, *a, **k):
        seen.append((os.path.basename(path), k.get("step")))
        return real(path, *a, **k)

    T.save_checkpoint, orig = spy, T.save_checkpoint
    try:
        train(_cfg(tmp_path, out_dir=out, max_steps=30, eval_interval=10, save_interval=1000))
    finally:
        T.save_checkpoint = orig
    steps = [s for n, s in seen if n.startswith("ckpt_last")]
    assert steps == [10, 20, 30]                 # one after each of the evaluations at 10 and 20, and the final one


def test_rolling_backups_keep_only_the_newest_n(tmp_path):
    out = str(tmp_path / "r")
    train(_cfg(tmp_path, out_dir=out, max_steps=30, eval_interval=1000, save_interval=5, keep_backups=2))
    assert _steps(out) == [20, 25]                # previous ckpt_last files; the newest (30) is ckpt_last.pt itself
    assert read_checkpoint(os.path.join(out, "ckpt_last.pt"))[2]["step"] == 30
    assert not glob.glob(os.path.join(out, "*.new")) and not glob.glob(os.path.join(out, "*.tmp"))


def test_no_backups_when_disabled(tmp_path):
    out = str(tmp_path / "r")
    train(_cfg(tmp_path, out_dir=out, max_steps=20, eval_interval=1000, save_interval=5, keep_backups=0))
    assert _steps(out) == []


def test_stop_file_saves_and_exits_and_resume_auto_reproduces_the_uninterrupted_run(tmp_path):
    straight = train(_cfg(tmp_path, out_dir=str(tmp_path / "s"), max_steps=20, eval_interval=1000))
    out = str(tmp_path / "p")
    real_eval = T.evaluate

    def eval_then_stop(*a, **k):
        open(os.path.join(out, "STOP"), "w").close()      # an operator (or the cloud scheduler) asks for a stop during step 10
        return real_eval(*a, **k)

    T.evaluate = eval_then_stop
    try:
        part = train(_cfg(tmp_path, out_dir=out, max_steps=20, eval_interval=10))
    finally:
        T.evaluate = real_eval
    assert part["stopped_early"] and part["step"] == 10
    assert not os.path.exists(os.path.join(out, "STOP"))
    assert read_checkpoint(os.path.join(out, "ckpt_last.pt"))[2]["step"] == 10
    rest = train(_cfg(tmp_path, out_dir=out, max_steps=20, eval_interval=1000, resume="auto"))
    assert rest["step"] == 20 and not rest["stopped_early"]
    assert _same_weights(straight["checkpoint"], rest["checkpoint"])


def test_sigterm_finishes_the_step_saves_and_exits(tmp_path):
    out = str(tmp_path / "p")
    real_eval = T.evaluate

    def eval_then_sigterm(*a, **k):
        signal.raise_signal(signal.SIGTERM)
        return real_eval(*a, **k)

    before = signal.getsignal(signal.SIGTERM)
    T.evaluate = eval_then_sigterm
    try:
        part = train(_cfg(tmp_path, out_dir=out, max_steps=20, eval_interval=10))
    finally:
        T.evaluate = real_eval
    assert part["stopped_early"] and part["step"] == 10
    assert read_checkpoint(os.path.join(out, "ckpt_last.pt"))[2]["step"] == 10
    assert signal.getsignal(signal.SIGTERM) == before     # handlers are restored


def test_resume_auto_with_nothing_to_resume_starts_fresh(tmp_path):
    r = train(_cfg(tmp_path, out_dir=str(tmp_path / "new"), max_steps=6, eval_interval=1000, resume="auto"))
    assert r["step"] == 6


def test_resume_auto_skips_a_corrupt_latest_checkpoint(tmp_path):
    out = str(tmp_path / "r")
    train(_cfg(tmp_path, out_dir=out, max_steps=20, eval_interval=1000, save_interval=5, keep_backups=2))
    last = os.path.join(out, "ckpt_last.pt")
    with open(last, "r+b") as f:                    # simulate a half-written / damaged file
        f.truncate(os.path.getsize(last) // 2)
    with pytest.raises(Exception):
        verify_checkpoint(last)
    with pytest.warns(UserWarning, match="skipping unusable"):
        newest = find_latest_checkpoint(out)
    assert newest is not None and os.path.basename(newest).startswith("ckpt_step_")
    assert read_checkpoint(newest)[2]["step"] == 15


def test_a_resumed_run_cannot_overwrite_a_better_best_checkpoint(tmp_path):
    out = str(tmp_path / "r")
    train(_cfg(tmp_path, out_dir=out, max_steps=20, eval_interval=10, stop_step=20))
    best_before = read_checkpoint(os.path.join(out, "ckpt_best.pt"))[2]
    real_eval = T.evaluate
    T.evaluate = lambda *a, **k: {"val_loss": 1e6, "val_ppl": float("inf")}      # a much worse validation loss after resuming
    try:
        train(_cfg(tmp_path, out_dir=out, max_steps=40, eval_interval=10, resume="auto"))
    finally:
        T.evaluate = real_eval
    best_after = read_checkpoint(os.path.join(out, "ckpt_best.pt"))[2]
    assert best_after["step"] == best_before["step"] and best_after["val_loss"] == best_before["val_loss"]


def test_prune_backups_orders_by_step_number(tmp_path):
    for n in (9, 10, 100, 2):
        (tmp_path / f"ckpt_step_{n:08d}.pt").write_bytes(b"x")
    removed = prune_backups(str(tmp_path), keep=2)
    assert sorted(os.path.basename(p) for p in removed) == ["ckpt_step_00000002.pt", "ckpt_step_00000009.pt"]


def test_find_latest_does_not_open_older_backups_when_the_last_file_is_good(tmp_path):
    out = str(tmp_path / "r")
    train(_cfg(tmp_path, out_dir=out, max_steps=20, eval_interval=1000, save_interval=5, keep_backups=2))
    backups = sorted(glob.glob(os.path.join(out, "ckpt_step_*.pt")))
    assert backups
    for b in backups:                               # damage every backup: a good ckpt_last.pt must still be chosen, silently
        with open(b, "r+b") as f:
            f.truncate(10)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        newest = find_latest_checkpoint(out)
    assert os.path.basename(newest) == "ckpt_last.pt"


def test_signal_handlers_are_restored_even_when_training_raises(tmp_path):
    before = signal.getsignal(signal.SIGTERM)
    real_lr = T.lr_at

    def boom(step, cfg):
        if step == 3:
            raise RuntimeError("simulated failure")
        return real_lr(step, cfg)

    T.lr_at = boom
    try:
        with pytest.raises(RuntimeError, match="simulated failure"):
            train(_cfg(tmp_path, out_dir=str(tmp_path / "r"), max_steps=10, eval_interval=1000))
    finally:
        T.lr_at = real_lr
    assert signal.getsignal(signal.SIGTERM) == before
