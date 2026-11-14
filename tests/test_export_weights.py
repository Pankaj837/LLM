"""scripts/export_weights.py: weights-only copy of a training checkpoint that every ``--ckpt`` script still loads."""
import sys
from pathlib import Path

import pytest
import torch

from helpers import small_lm_config
from llm.checkpoint import load_checkpoint, read_checkpoint, save_checkpoint
from llm.model import TransformerModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import export_weights as ew  # noqa: E402


@pytest.fixture()
def full_ckpt(tmp_path):
    torch.manual_seed(0)
    cfg = small_lm_config()
    m = TransformerModel(cfg)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    m(torch.randint(0, cfg.vocab_size, (2, 16))).float().mean().backward()
    opt.step()                                       # gives the optimizer real moments
    p = str(tmp_path / "ckpt_full.pt")
    save_checkpoint(p, m, cfg, step=477, train_loss=2.62, val_loss=3.72, vocab_sha256="abc", optimizer=opt)
    return p, m, cfg


def test_fp32_export_is_exact_smaller_and_loadable(full_ckpt, tmp_path):
    p, m, cfg = full_ckpt
    out = str(tmp_path / "w32.pt")
    r = ew.export(p, out, "fp32")
    assert r["bytes"] < r["source_bytes"] * 0.6                       # weights only: no AdamW moments
    state, cfg2, meta, raw = read_checkpoint(out)
    assert "optimizer" not in raw and cfg2 == cfg
    assert meta["step"] == 477 and meta["val_loss"] == 3.72 and meta["exported"]["optimizer_dropped"] is True
    fresh = TransformerModel(cfg)
    load_checkpoint(out, fresh)
    for (k, a), (_, b) in zip(m.state_dict().items(), fresh.state_dict().items()):
        assert torch.equal(a, b), k


def test_bf16_export_halves_the_weights_and_stays_close(full_ckpt, tmp_path):
    p, m, cfg = full_ckpt
    out32, out16 = str(tmp_path / "w32.pt"), str(tmp_path / "w16.pt")
    r32, r16 = ew.export(p, out32, "fp32"), ew.export(p, out16, "bf16")
    assert r16["bytes"] < r32["bytes"] * 0.6
    state, _, _, _ = read_checkpoint(out16)
    assert all(v.dtype == torch.bfloat16 for v in state.values() if v.is_floating_point())
    fresh = TransformerModel(cfg)
    load_checkpoint(out16, fresh)
    ids = torch.randint(0, cfg.vocab_size, (1, 16))
    m.eval(), fresh.eval()
    with torch.no_grad():
        a, b = m(ids).float(), fresh(ids).float()
    assert (a - b).abs().max().item() < 0.2                           # bf16 rounding only


def test_command_line_and_refusals(full_ckpt, tmp_path, capsys):
    p, _, _ = full_ckpt
    out = str(tmp_path / "w.pt")
    assert ew.main(["--ckpt", p, "--out", out]) == 0
    printed = capsys.readouterr().out
    assert "sha256" in printed and ew.sha256(out) in printed
    assert ew.main(["--ckpt", p, "--out", p]) == 2                    # never overwrites the source
    bad = tmp_path / "legacy.pt"
    torch.save({"layers.0.weight": torch.zeros(2)}, bad)
    assert ew.main(["--ckpt", str(bad), "--out", str(tmp_path / "x.pt")]) == 1
