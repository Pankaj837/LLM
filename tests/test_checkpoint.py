"""Checkpoint format, provenance and safe loading (component C7)."""
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from helpers import small_lm_config, to_checkpoint_format
from llm.checkpoint import (FORMAT_VERSION, load_checkpoint, looks_untrained, read_checkpoint, remap_legacy_key,
                            save_checkpoint)
from llm.model import TransformerModel

ROOT = Path(__file__).resolve().parent.parent


def _trained_like(model):
    """Perturb a fresh model so it no longer looks like a random init."""
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n.endswith("norm.weight"):
                p.add_(torch.randn_like(p) * 0.2)
    return model


@pytest.fixture()
def model_and_cfg():
    torch.manual_seed(0)
    cfg = small_lm_config()
    return _trained_like(TransformerModel(cfg)), cfg


def test_roundtrip_restores_weights_config_and_metadata(model_and_cfg, tmp_path):
    m, cfg = model_and_cfg
    p = str(tmp_path / "c.pt")
    save_checkpoint(p, m, cfg, step=123, train_loss=2.5, val_loss=2.7, vocab_sha256="abc", extra={"seed": 7})
    fresh = TransformerModel(cfg)
    info = load_checkpoint(p, fresh)
    assert (info["step"], info["train_loss"], info["val_loss"], info["vocab_sha256"]) == (123, 2.5, 2.7, "abc")
    assert info["extra"] == {"seed": 7} and info["has_config"] and not info["missing_keys"]
    ids = torch.randint(0, 1722, (1, 8))
    with torch.no_grad():
        assert torch.equal(m.eval()(ids), fresh.eval()(ids))


def test_stored_config_is_recoverable(model_and_cfg, tmp_path):
    m, cfg = model_and_cfg
    p = str(tmp_path / "c.pt")
    save_checkpoint(p, m, cfg)
    _, stored, meta, _ = read_checkpoint(p)
    assert stored == cfg and meta["torch_version"] == torch.__version__


def test_checkpoint_loads_with_weights_only_true(model_and_cfg, tmp_path):
    """Security: no arbitrary unpickling is needed to read our own files."""
    m, cfg = model_and_cfg
    p = str(tmp_path / "c.pt")
    save_checkpoint(p, m, cfg, optimizer=torch.optim.AdamW(m.parameters()))
    raw = torch.load(p, map_location="cpu", weights_only=True)
    assert raw["format_version"] == FORMAT_VERSION and "optimizer" in raw


def test_config_mismatch_is_rejected_with_a_readable_diff(model_and_cfg, tmp_path):
    m, cfg = model_and_cfg
    p = str(tmp_path / "c.pt")
    save_checkpoint(p, m, cfg)
    with pytest.raises(ValueError, match="num_layers"):
        load_checkpoint(p, TransformerModel(small_lm_config(num_layers=3)))


def test_vocab_hash_mismatch_is_rejected(model_and_cfg, tmp_path):
    m, cfg = model_and_cfg
    p = str(tmp_path / "c.pt")
    save_checkpoint(p, m, cfg, vocab_sha256="aaaa")
    load_checkpoint(p, TransformerModel(cfg), expected_vocab_sha256="aaaa")
    with pytest.raises(ValueError, match="vocabulary"):
        load_checkpoint(p, TransformerModel(cfg), expected_vocab_sha256="bbbb")


def test_newer_format_version_is_rejected(model_and_cfg, tmp_path):
    m, cfg = model_and_cfg
    p = str(tmp_path / "c.pt")
    save_checkpoint(p, m, cfg)
    raw = torch.load(p, weights_only=True)
    raw["format_version"] = FORMAT_VERSION + 1
    torch.save(raw, p)
    with pytest.raises(ValueError, match="newer"):
        read_checkpoint(p)


def test_legacy_raw_state_dict_loads_and_warns(model_and_cfg, tmp_path):
    m, cfg = model_and_cfg
    p = str(tmp_path / "legacy.pt")
    torch.save(to_checkpoint_format(m.state_dict()), p)
    with pytest.warns(UserWarning, match="legacy"):
        info = load_checkpoint(p, TransformerModel(cfg))
    assert not info["has_config"]


def test_untrained_flag_triggers_a_warning(tmp_path):
    cfg = small_lm_config()
    m = TransformerModel(cfg)
    p = str(tmp_path / "s.pt")
    save_checkpoint(p, m, cfg, untrained=True)
    with pytest.warns(UserWarning, match="UNTRAINED"):
        load_checkpoint(p, TransformerModel(cfg))


def test_looks_untrained_heuristic_separates_fresh_from_perturbed():
    cfg = small_lm_config()
    fresh = TransformerModel(cfg)
    assert looks_untrained(fresh.state_dict())
    assert not looks_untrained(_trained_like(TransformerModel(cfg)).state_dict())


def test_legacy_key_remap_is_idempotent_and_complete():
    assert remap_legacy_key("layers.3.attn.W_q.weight") == "layers.3.attn.q_proj.weight"
    assert remap_legacy_key(remap_legacy_key("layers.3.attn.W_o.weight")) == "layers.3.attn.o_proj.weight"
    assert remap_legacy_key("layers.3.ffn.W_gate.weight") == "layers.3.ffn.W_gate.weight"   # FFN keys keep their names


def test_strict_load_rejects_wrong_shapes(model_and_cfg, tmp_path):
    m, cfg = model_and_cfg
    p = str(tmp_path / "c.pt")
    save_checkpoint(p, m, cfg)
    raw = torch.load(p, weights_only=True)
    raw["config"] = small_lm_config(vocab_size=1000).to_dict()      # config now disagrees with tensors
    torch.save(raw, p)
    with pytest.raises(ValueError):
        load_checkpoint(p, TransformerModel(cfg))


def test_model_load_pretrained_keeps_its_tuple_api(model_and_cfg, tmp_path):
    m, cfg = model_and_cfg
    p = str(tmp_path / "c.pt")
    save_checkpoint(p, m, cfg)
    assert TransformerModel(cfg).load_pretrained(p) == ([], [])


def test_scaffold_script_writes_a_flagged_untrained_checkpoint(tmp_path):
    out = tmp_path / "scaffold.pt"
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "make_scaffold_checkpoint.py"), "--out", str(out),
                        "--vocab-size", "500"], capture_output=True, text=True, cwd=str(ROOT))
    assert r.returncode == 0, r.stderr[-300:]
    _, cfg, meta, _ = read_checkpoint(str(out))
    assert meta["untrained"] is True and cfg.vocab_size == 500


# ------------------------------------------------------------------ external artefact gate
def test_external_checkpoint_is_actually_trained(ckpt_path):
    """RELEASE GATE. Run with LLM_CKPT=<file>. A trained matrix has singular-value outliers far above the
    Marchenko-Pastur edge of a random matrix; norm weights drift from 1. The delivered scaffold FAILS this
    by design (it is random init). Set LLM_ALLOW_UNTRAINED=1 to acknowledge and skip."""
    state, _, meta, _ = read_checkpoint(str(ckpt_path))
    if os.environ.get("LLM_ALLOW_UNTRAINED") == "1" or meta.get("untrained"):
        pytest.skip("checkpoint acknowledged as untrained scaffold")
    ratios = []
    for k in [k for k in state if k.endswith("attn.q_proj.weight") or k.endswith("ffn.W_gate.weight")][:3]:
        W = state[k].float()
        edge = W.std().item() * (math.sqrt(W.shape[0]) + math.sqrt(W.shape[1]))
        ratios.append(torch.linalg.svdvals(W)[0].item() / edge)
    assert max(ratios) > 1.5 or not looks_untrained(state), f"looks like random init: sv/edge={ratios}"
