"""Reference trainer (component C12): schedule, optimisation, determinism, resume, failure modes."""
import json
import math

import numpy as np
import pytest
import torch

from llm import data
from llm.checkpoint import read_checkpoint
from llm.train import PRESETS, TrainConfig, build_optimizer, loss_fn, lr_at, train
from llm.model import TransformerModel
from llm.model_config import ModelConfig

VOCAB = 64


def _bin(tmp_path, name, n=4000, seed=0, pattern=True):
    """Learnable synthetic language: a fixed repeating cycle with a little noise."""
    rng = np.random.RandomState(seed)
    base = np.tile(np.arange(1, 17), n // 16 + 1)[:n]
    if not pattern:
        base = rng.randint(1, VOCAB, n)
    noise = rng.rand(n) < 0.02
    ids = np.where(noise, rng.randint(1, VOCAB, n), base)
    p = str(tmp_path / name)
    data.write_bin(p, ids, vocab_size=VOCAB, eos_id=VOCAB - 1)
    return p


def _cfg(tmp_path, **kw):
    base = dict(train_bin=_bin(tmp_path, "train.bin"), val_bin=_bin(tmp_path, "val.bin", 2000, seed=1),
                out_dir=str(tmp_path / "run"), preset="tiny", seq_len=32, batch_size=8, max_steps=40,
                warmup_steps=5, eval_interval=20, eval_batches=4, save_interval=1000, log_interval=10,
                lr=3e-3, device="cpu", dtype="fp32")
    base.update(kw)
    return TrainConfig(**base)


# ------------------------------------------------------------------ schedule
def test_lr_schedule_warmup_peak_and_floor():
    c = TrainConfig(lr=1.0, warmup_steps=10, max_steps=110, min_lr_ratio=0.1)
    assert lr_at(0, c) == pytest.approx(0.1) and lr_at(9, c) == pytest.approx(1.0)
    assert lr_at(10, c) == pytest.approx(1.0)                                   # cosine starts at the peak
    assert lr_at(60, c) == pytest.approx(0.55, abs=1e-6)                        # midpoint of cosine
    assert lr_at(110, c) == pytest.approx(0.1) and lr_at(10_000, c) == pytest.approx(0.1)
    lrs = [lr_at(s, c) for s in range(10, 111)]
    assert all(a >= b - 1e-12 for a, b in zip(lrs, lrs[1:]))                     # non-increasing after warmup


def test_weight_decay_only_on_matrices():
    m = TransformerModel(ModelConfig(**{**PRESETS["tiny"], "vocab_size": VOCAB}))
    opt = build_optimizer(m, TrainConfig(weight_decay=0.1))
    decay, no_decay = opt.param_groups
    assert decay["weight_decay"] == 0.1 and no_decay["weight_decay"] == 0.0
    assert all(p.ndim >= 2 for p in decay["params"]) and all(p.ndim < 2 for p in no_decay["params"])
    assert len(decay["params"]) + len(no_decay["params"]) == len(list(m.parameters()))


# ------------------------------------------------------------------ learning
def test_initial_loss_is_ln_vocab_and_training_reduces_it(tmp_path):
    out = train(_cfg(tmp_path, max_steps=60, log_interval=1))
    h = out["history"]
    first_train = h[0]["loss"]
    assert abs(first_train - math.log(VOCAB)) < 0.6, first_train        # starts near uniform (init fix)
    assert h[-1]["loss"] < 0.5 * first_train
    val = [r["val_loss"] for r in h if "val_loss" in r]
    assert h[0]["step"] == 1
    assert val[-1] < val[0] and val[-1] < math.log(VOCAB)


def test_can_overfit_one_batch(tmp_path):
    cfg = ModelConfig(**{**PRESETS["tiny"], "vocab_size": VOCAB})
    torch.manual_seed(0)
    m = TransformerModel(cfg)
    x = torch.randint(1, VOCAB, (4, 32)); y = torch.randint(1, VOCAB, (4, 32))
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    for _ in range(150):
        opt.zero_grad(); l = loss_fn(m, x, y); l.backward(); opt.step()
    assert l.item() < 0.1


def test_random_tokens_cannot_be_learned_below_entropy(tmp_path):
    """Sanity check against leakage: on i.i.d. uniform tokens validation loss must stay near ln V."""
    c = _cfg(tmp_path, train_bin=_bin(tmp_path, "rt.bin", pattern=False), val_bin=_bin(tmp_path, "rv.bin", 2000, seed=3, pattern=False), max_steps=40)
    out = train(c)
    assert out["final"]["val_loss"] > math.log(VOCAB) - 0.4


# ------------------------------------------------------------------ determinism & resume
def test_training_is_deterministic_given_the_seed(tmp_path):
    a = train(_cfg(tmp_path, out_dir=str(tmp_path / "ra"), max_steps=15, eval_interval=15))
    b = train(_cfg(tmp_path, out_dir=str(tmp_path / "rb"), max_steps=15, eval_interval=15))
    assert [r["loss"] for r in a["history"] if "loss" in r] == [r["loss"] for r in b["history"] if "loss" in r]


def test_resume_reproduces_the_uninterrupted_run_exactly(tmp_path):
    straight = train(_cfg(tmp_path, out_dir=str(tmp_path / "s"), max_steps=20, eval_interval=1000))
    part = train(_cfg(tmp_path, out_dir=str(tmp_path / "p"), max_steps=20, stop_step=10, eval_interval=1000))
    rest = train(_cfg(tmp_path, out_dir=str(tmp_path / "p2"), max_steps=20, eval_interval=1000, resume=part["checkpoint"]))
    sa, _, ma, _ = read_checkpoint(straight["checkpoint"])
    sb, _, mb, _ = read_checkpoint(rest["checkpoint"])
    assert ma["step"] == mb["step"] == 20
    for k in sa:
        assert torch.equal(sa[k], sb[k]), k


def test_grad_accumulation_matches_a_larger_batch_in_expectation(tmp_path):
    """Both settings must learn (equivalence is statistical: the micro-batches differ)."""
    out = train(_cfg(tmp_path, batch_size=4, grad_accum=2, max_steps=40))
    assert out["history"][-1]["loss"] < out["history"][0]["loss"]


# ------------------------------------------------------------------ artefacts & provenance
def test_metrics_and_config_are_written_and_checkpoint_has_provenance(tmp_path, vocab_path):
    out = train(_cfg(tmp_path, vocab=vocab_path, max_steps=10, eval_interval=10, log_interval=5))
    run = tmp_path / "run"
    lines = [json.loads(l) for l in (run / "metrics.jsonl").read_text().splitlines()]
    assert lines and all("step" in l for l in lines)
    cfgj = json.loads((run / "train_config.json").read_text())
    assert cfgj["model"]["vocab_size"] == VOCAB and cfgj["train"]["seed"] == 0
    _, mcfg, meta, raw = read_checkpoint(out["checkpoint"])
    assert meta["vocab_sha256"] and len(meta["vocab_sha256"]) == 64 and meta["step"] == 10 and "optimizer" in raw
    assert (run / "ckpt_best.pt").exists()


def test_trained_checkpoint_is_not_flagged_untrained(tmp_path):
    out = train(_cfg(tmp_path, max_steps=30))
    _, _, meta, _ = read_checkpoint(out["checkpoint"])
    assert meta["untrained"] is False


# ------------------------------------------------------------------ failure modes
def test_vocab_mismatch_between_train_and_val_is_rejected(tmp_path):
    bad = str(tmp_path / "bad.bin")
    data.write_bin(bad, np.arange(3000) % 30, vocab_size=128, eos_id=127)
    with pytest.raises(ValueError, match="vocab"):
        train(_cfg(tmp_path, val_bin=bad))


def test_seq_len_beyond_context_window_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="max_position_embeddings"):
        train(_cfg(tmp_path, seq_len=1024))


def test_unknown_preset_rejected(tmp_path):
    with pytest.raises(ValueError, match="preset"):
        train(_cfg(tmp_path, preset="huge"))


def test_non_finite_loss_stops_training(tmp_path, monkeypatch):
    import llm.train as T
    monkeypatch.setattr(T, "loss_fn", lambda m, x, y: torch.tensor(float("nan"), requires_grad=True))
    with pytest.raises(FloatingPointError):
        train(_cfg(tmp_path, max_steps=3))


def test_resume_without_optimizer_state_is_rejected(tmp_path):
    from llm.checkpoint import save_checkpoint
    cfg = ModelConfig(**{**PRESETS["tiny"], "vocab_size": VOCAB})
    p = str(tmp_path / "noopt.pt")
    save_checkpoint(p, TransformerModel(cfg), cfg, step=5)
    with pytest.raises(ValueError, match="optimizer"):
        train(_cfg(tmp_path, resume=p))
