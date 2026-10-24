"""Weighted domain mixing (MixtureDataset / --train-mix) and atomic checkpoint writes."""
import numpy as np
import pytest
import torch

from llm import data
from llm.checkpoint import read_checkpoint, save_checkpoint
from llm.data import MixtureDataset
from llm.model import TransformerModel
from llm.model_config import ModelConfig
from llm.train import PRESETS, TrainConfig, parse_mix, train

VOCAB = 64


def _bin(path, value, n=2000, vocab=VOCAB, eos=VOCAB - 1):
    data.write_bin(str(path), np.full(n, value, dtype=np.int64), vocab_size=vocab, eos_id=eos)
    return str(path)


def test_sampling_follows_the_weights_and_is_deterministic(tmp_path):
    a, b = _bin(tmp_path / "a.bin", 3), _bin(tmp_path / "b.bin", 7)
    ds = MixtureDataset([(a, 0.8), (b, 2.0 * 0.1)], seq_len=16)          # weights are normalised: 0.8 : 0.2
    g = torch.Generator()
    g.manual_seed(1)
    x, _ = ds.get_batch(4000, g)
    share_b = float((x[:, 0] == 7).float().mean())
    assert 0.17 < share_b < 0.23
    g.manual_seed(1)
    x2, _ = ds.get_batch(4000, g)
    assert torch.equal(x, x2)
    assert ds.header.total_tokens == 4000 and ds.header.vocab_size == VOCAB
    assert ds.epochs(8000) == pytest.approx([8000 * 0.8 / 2000, 8000 * 0.2 / 2000])


def test_incompatible_files_and_bad_weights_are_rejected(tmp_path):
    a = _bin(tmp_path / "a.bin", 3)
    other = _bin(tmp_path / "o.bin", 3, vocab=128, eos=127)
    with pytest.raises(ValueError, match="different vocabularies"):
        MixtureDataset([(a, 1.0), (other, 1.0)], 16)
    with pytest.raises(ValueError, match="> 0"):
        MixtureDataset([(a, 0.0)], 16)
    with pytest.raises(ValueError):
        MixtureDataset([], 16)


def test_parse_mix_takes_the_weight_after_the_last_equals_sign():
    assert parse_mix([r"C:\data\a=b\math.bin=0.25", "code.bin=1"]) == [(r"C:\data\a=b\math.bin", 0.25), ("code.bin", 1.0)]
    with pytest.raises(ValueError):
        parse_mix(["no_weight.bin"])


def _cfg(tmp_path, **kw):
    rng = np.random.RandomState(0)
    for name, off in (("m1", 0), ("m2", 8)):
        ids = np.tile(np.arange(1 + off, 9 + off), 400)
        ids = np.where(rng.rand(len(ids)) < 0.02, rng.randint(1, VOCAB, len(ids)), ids)
        data.write_bin(str(tmp_path / f"{name}.bin"), ids, vocab_size=VOCAB, eos_id=VOCAB - 1)
    base = dict(train_mix=[f"{tmp_path / 'm1.bin'}=0.7", f"{tmp_path / 'm2.bin'}=0.3"], out_dir=str(tmp_path / "run"),
                preset="tiny", seq_len=32, batch_size=8, max_steps=24, warmup_steps=3, eval_interval=1000,
                save_interval=1000, log_interval=8, lr=3e-3, device="cpu", dtype="fp32")
    base.update(kw)
    return TrainConfig(**base)


def test_training_on_a_mixture_learns_and_resume_reproduces_the_uninterrupted_run(tmp_path):
    (tmp_path / "full").mkdir()
    full = train(_cfg(tmp_path / "full"))
    assert full["history"][-1]["loss"] < full["history"][0]["loss"]
    (tmp_path / "part").mkdir()
    train(_cfg(tmp_path / "part", stop_step=12))
    resumed = train(_cfg(tmp_path / "part", resume=str(tmp_path / "part" / "run" / "ckpt_last.pt")))
    a = read_checkpoint(full["checkpoint"])[0]
    b = read_checkpoint(resumed["checkpoint"])[0]
    assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


def test_checkpoint_write_is_atomic(tmp_path, monkeypatch):
    cfg = ModelConfig(**{**PRESETS["tiny"], "vocab_size": VOCAB})
    model = TransformerModel(cfg)
    path = str(tmp_path / "ckpt.pt")
    save_checkpoint(path, model, cfg, step=1)
    before = open(path, "rb").read()

    def exploding_save(obj, f, *a, **k):
        with open(f, "wb") as fh:
            fh.write(b"partial")
        raise OSError("disk full")

    monkeypatch.setattr(torch, "save", exploding_save)
    with pytest.raises(OSError):
        save_checkpoint(path, model, cfg, step=2)
    assert open(path, "rb").read() == before            # the previous checkpoint is untouched


def test_command_line_accepts_train_mix_and_requires_some_training_data(tmp_path, capsys):
    from llm.train import main
    cfg = _cfg(tmp_path, max_steps=4, log_interval=2)
    args = ["--train-mix", *cfg.train_mix, "--out", str(tmp_path / "cli"), "--preset", "tiny", "--seq-len", "32",
            "--batch-size", "4", "--max-steps", "4", "--warmup-steps", "1", "--device", "cpu", "--dtype", "fp32"]
    main(args)
    out = capsys.readouterr().out
    assert "m1.bin: " in out and "epochs" in out and "tokens per optimizer step" in out
    with pytest.raises(SystemExit):
        main(["--out", str(tmp_path / "none")])
