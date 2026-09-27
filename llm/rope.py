"""Rotary Positional Embedding (RoPE) with Dynamic NTK and YaRN context extension.

Three mutually exclusive modes (``ModelConfig.rope_mode``):

  "rope"         plain RoPE (Su et al., 2021).
  "dynamic_ntk"  theta is derived from the planned context length ``L``:
                 unchanged for L <= L_orig, else theta * (L / L_orig) ** (d / (d - 2))
                 (``DynamicNTK``; the cos/sin tables are cached by ``RopeCache``).
  "yarn"         YaRN (Peng et al., 2023): "NTK-by-parts" frequency interpolation plus an
                 attention-logit multiplier (see ``attention_temperature``).

Where RoPE is applied: inside every attention block, to Q and K only, *before* the
KV heads are repeated for GQA (see ``gqa.py``). It is never added to the embeddings.

Context length ("planned context") and KV-caching
--------------------------------------------------
Dynamic NTK makes the rotation frequencies depend on the sequence length. With a KV cache
the keys of earlier tokens are stored already rotated, so the frequencies must not change
while a sequence is being decoded. The model therefore fixes ``context_len`` per call:
the KV-cache capacity (prompt + max_new_tokens) when decoding, otherwise the input length.
Every token of one generation is rotated with the same theta, which makes cached and
uncached decoding identical.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn

from .dynamic_ntk import DynamicNTK
from .model_config import ModelConfig
from .rope_cache import RopeCache


class RotaryEmbedding(nn.Module):
    """Rotary embedding shared by all transformer layers.

    Public surface
    --------------
    ``get_cos_sin(position_ids, dtype, context_len)``  cos/sin for the given positions,
        shape (batch, 1, seq, head_dim); no device->host synchronisation.
    ``cos_cached`` / ``sin_cached`` / ``inv_freq``     tables for ``max_position_embeddings``
        (the layout ``GroupedQueryAttention`` uses when it owns its RoPE).
    ``build_cache(seq_len, device, dtype)``            (seq_len, head_dim) tables, seq_len positions.
    ``attention_temperature()``                        logit multiplier (1.0 unless YaRN).
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.head_dim = cfg.head_dim
        self.max_position_embeddings = cfg.max_position_embeddings
        self.mode = cfg.rope_mode

        inv_freq = self._compute_inv_freq(cfg)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        # fp32 master copy that module.to(dtype) never touches: casting the registered buffer
        # to bf16/fp16 would corrupt the rotation angles (error grows with position).
        self._inv_freq_master = inv_freq.detach().clone()

        self._ntk: Optional[DynamicNTK] = None
        if self.mode == "dynamic_ntk":
            self._ntk = DynamicNTK(cfg.theta_base, cfg.trained_context_len, cfg.head_dim)
        self._cache = RopeCache()

        cos, sin = self._tables(cfg.max_position_embeddings, torch.device("cpu"))
        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)

    # ── frequencies ─────────────────────────────────────────────────────
    def _compute_inv_freq(self, cfg: ModelConfig) -> torch.Tensor:
        """Static frequencies (float32). For Dynamic NTK these are the *base* frequencies;
        the scaled theta is applied per call in ``_tables``."""
        dim, theta = cfg.head_dim, cfg.theta_base
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        if cfg.use_yarn:
            inv_freq = self._apply_yarn(inv_freq, cfg)
        return inv_freq

    def _apply_yarn(self, inv_freq: torch.Tensor, cfg: ModelConfig) -> torch.Tensor:
        """YaRN "NTK-by-parts" interpolation (Peng et al., 2023; matches the reference
        implementation, e.g. Hugging Face ``_compute_yarn_parameters``).

        The blend ramp is linear in the frequency-dimension *index* between the dimensions
        that complete ``yarn_beta`` and ``yarn_alpha`` rotations over the trained context:
          - more than ``yarn_beta`` rotations: unchanged (extrapolate)
          - fewer than ``yarn_alpha`` rotations: divided by the scale (interpolate)
          - in between: linear blend
        """
        dim, base, length = cfg.head_dim, cfg.theta_base, cfg.trained_context_len

        def correction_dim(num_rotations: float) -> float:
            return dim * math.log(length / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

        low = max(math.floor(correction_dim(cfg.yarn_beta)), 0)
        high = min(math.ceil(correction_dim(cfg.yarn_alpha)), dim - 1)
        if low == high:
            high += 0.001  # avoid a zero-width ramp

        ramp = ((torch.arange(dim // 2, dtype=torch.float32) - low) / (high - low)).clamp(0.0, 1.0)
        extrapolation = 1.0 - ramp
        return (inv_freq / cfg.yarn_scale_factor) * (1.0 - extrapolation) + inv_freq * extrapolation

    def attention_temperature(self) -> float:
        """YaRN attention-logit multiplier.

        The paper scales q and k each by sqrt(1/t) = 0.1*ln(s) + 1, so logits are multiplied
        by 1/t = (0.1*ln(s) + 1)^2 (>= 1: sharper attention, countering entropy growth at long
        context). ``GroupedQueryAttention`` applies ``scale = multiplier / sqrt(head_dim)``.
        Returns 1.0 unless YaRN is active.
        """
        if not self.cfg.use_yarn or self.cfg.yarn_scale_factor <= 1.0:
            return 1.0
        return (0.1 * math.log(self.cfg.yarn_scale_factor) + 1.0) ** 2

    # ── tables ──────────────────────────────────────────────────────────
    def theta_for(self, context_len: int) -> float:
        """Base frequency used for a planned context length (only Dynamic NTK changes it)."""
        return self._ntk.get_theta(context_len) if self._ntk is not None else self.cfg.theta_base

    def _tables(self, context_len: int, device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Full-width float32 (rows, head_dim) tables valid for a planned ``context_len``.

        Rows cover at least ``context_len`` positions. When the frequencies do not depend on
        ``context_len`` (plain RoPE, YaRN, or Dynamic NTK at/below the trained length) the
        table is extended to ``max_position_embeddings`` so it is reused across calls.
        """
        theta = self.theta_for(context_len)
        if self._ntk is not None and theta != self.cfg.theta_base:
            return self._cache.build(context_len, self.head_dim, theta, device=device, duplicate=True)
        rows = max(context_len, self.max_position_embeddings)
        return self._cache.build(rows, self.head_dim, inv_freq=self._inv_freq_master, freq_tag="static",
                                 device=device, duplicate=True)

    def build_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        """(seq_len, head_dim) tables for positions [0, seq_len), planned context = seq_len."""
        cos, sin = self._tables(seq_len, device)
        return cos[:seq_len].to(dtype), sin[:seq_len].to(dtype)

    def get_cos_sin(
        self, position_ids: torch.Tensor, dtype: torch.dtype, context_len: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """cos/sin for ``position_ids`` (batch, seq) -> two tensors (batch, 1, seq, head_dim).

        ``context_len`` is the planned context (must exceed the largest position). When it is
        omitted it is derived from ``position_ids`` (one device->host sync); the model always
        supplies it so the forward path stays synchronisation-free.
        """
        if context_len is None:
            context_len = int(position_ids.max().item()) + 1
        cos, sin = self._tables(context_len, position_ids.device)
        return cos[position_ids].to(dtype).unsqueeze(1), sin[position_ids].to(dtype).unsqueeze(1)

    def forward(self, seq_len: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """(cos, sin) tables of shape (seq_len, head_dim); the full cached tables if None."""
        if seq_len is None:
            return self.cos_cached, self.sin_cached
        return self.build_cache(seq_len, self.cos_cached.device, self.cos_cached.dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of x: [-x2, x1]."""
    half_dim = x.shape[-1] // 2
    x1 = x[..., :half_dim]
    x2 = x[..., half_dim:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Applies rotary position embedding to x.
    x:   (batch, heads, seq_len, head_dim)
    cos: (batch, 1, seq_len, head_dim) or broadcastable
    sin: (batch, 1, seq_len, head_dim) or broadcastable
    """
    return (x * cos) + (rotate_half(x) * sin)
