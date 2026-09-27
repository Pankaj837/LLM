"""Checkpoint format with provenance, and safe loading.

Why this exists: the original ``pretrain_model.pt`` is a bare ``state_dict`` with no
configuration, tokenizer identity or training history, so it cannot be verified or
reproduced (and, per the integration report, it is a structural scaffold with random
weights despite its name). New checkpoints are self-describing:

    {
      "format_version": 1,
      "model":  state_dict,                    # native parameter names
      "config": ModelConfig.to_dict(),
      "meta":   {"step", "train_loss", "val_loss", "vocab_sha256", "untrained", "extra", ...},
      "optimizer": optional optimizer state
    }

Legacy raw state_dicts (keys like ``layers.0.attn.W_q.weight``) still load, with a warning.
Loading always uses ``torch.load(weights_only=True)``.
"""
from __future__ import annotations

import warnings
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from .model_config import ModelConfig

FORMAT_VERSION = 1

_LEGACY_RENAMES = ((".attn.W_q.", ".attn.q_proj."), (".attn.W_k.", ".attn.k_proj."),
                   (".attn.W_v.", ".attn.v_proj."), (".attn.W_o.", ".attn.o_proj."))


def remap_legacy_key(key: str) -> str:
    """``layers.i.attn.W_q.*`` -> ``layers.i.attn.q_proj.*`` (idempotent)."""
    for old, new in _LEGACY_RENAMES:
        key = key.replace(old, new)
    return key


def looks_untrained(state_dict: Dict[str, torch.Tensor]) -> bool:
    """Cheap heuristic for a randomly initialised model: every RMSNorm weight is within 1e-2 of
    1.0 and the embedding std is within 5% of the 0.02 init. Trained networks move their norm
    weights well beyond that. Used only to warn, never to block."""
    norms = [v.float() for k, v in state_dict.items() if k.endswith("norm.weight")]
    emb = state_dict.get("tok_embeddings.weight")
    if not norms or emb is None:
        return False
    close_to_one = all((n - 1).abs().max().item() < 1e-2 for n in norms)
    return close_to_one and abs(emb.float().std().item() / 0.02 - 1) < 0.05


def save_checkpoint(
    path: str,
    model: nn.Module,
    config: ModelConfig,
    *,
    step: int = 0,
    train_loss: Optional[float] = None,
    val_loss: Optional[float] = None,
    vocab_sha256: Optional[str] = None,
    untrained: bool = False,
    optimizer: Optional[torch.optim.Optimizer] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    payload = {
        "format_version": FORMAT_VERSION,
        "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "config": config.to_dict(),
        "meta": {"step": step, "train_loss": train_loss, "val_loss": val_loss,
                 "vocab_sha256": vocab_sha256, "untrained": untrained, "extra": extra or {},
                 "torch_version": str(torch.__version__)},
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    torch.save(payload, path)


def read_checkpoint(path: str) -> Tuple[Dict[str, torch.Tensor], Optional[ModelConfig], Dict[str, Any], Dict[str, Any]]:
    """Return (state_dict with native names, config or None, meta, raw payload). Handles both formats."""
    raw = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(raw, dict) and "format_version" in raw:
        if raw["format_version"] > FORMAT_VERSION:
            raise ValueError(f"checkpoint format {raw['format_version']} is newer than supported ({FORMAT_VERSION})")
        state = {remap_legacy_key(k): v for k, v in raw["model"].items()}
        return state, ModelConfig.from_dict(raw["config"]), raw.get("meta", {}), raw
    state = {remap_legacy_key(k): v for k, v in raw.items()}
    return state, None, {}, raw


def load_checkpoint(
    path: str,
    model: nn.Module,
    *,
    strict: bool = True,
    expected_vocab_sha256: Optional[str] = None,
) -> Dict[str, Any]:
    """Load weights into ``model`` and return the checkpoint's metadata.

    Raises on: unreadable file, key/shape mismatch (``strict``), stored config that
    disagrees with ``model.config``, or a vocabulary hash that differs from
    ``expected_vocab_sha256``. Warns on legacy (no provenance) or untrained weights.
    """
    state, cfg, meta, _ = read_checkpoint(path)

    model_cfg = getattr(model, "config", None)
    if cfg is not None and model_cfg is not None and cfg != model_cfg:
        diff = {k: (v, getattr(model_cfg, k)) for k, v in cfg.to_dict().items() if getattr(model_cfg, k) != v}
        raise ValueError(f"checkpoint config differs from model config (checkpoint, model): {diff}")

    stored = meta.get("vocab_sha256")
    if expected_vocab_sha256 and stored and stored != expected_vocab_sha256:
        raise ValueError("checkpoint was trained with a different vocabulary file (sha256 mismatch)")

    missing, unexpected = model.load_state_dict(state, strict=strict)
    if cfg is None:
        warnings.warn("legacy checkpoint: no config/tokenizer/training metadata, provenance unknown", stacklevel=2)
    if meta.get("untrained") or (cfg is None and looks_untrained(state)):
        warnings.warn("checkpoint holds UNTRAINED (randomly initialised) weights; outputs are meaningless "
                      "until a trained checkpoint is used", stacklevel=2)
    return {**meta, "missing_keys": list(missing), "unexpected_keys": list(unexpected), "has_config": cfg is not None}
