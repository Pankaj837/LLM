"""Cache for RoPE cosine / sine tables.

Specification (team's work report): build the position x frequency angle table once
and reuse it while the request is unchanged; keep state on the instance.

Behaviour of the delivered ``RopeCache.build(seq_len, head_dim, theta)`` is preserved
(default arguments return the same half-width ``(seq_len, head_dim/2)`` tables).
Two hardening changes:

  * The cache key now includes ``theta``, ``head_dim``, device and dtype (the delivered
    version keyed on ``seq_len`` only, so a different theta for the same length would
    have returned stale tables).
  * Optional ``inv_freq`` + ``freq_tag`` (custom frequencies, e.g. YaRN), ``duplicate`` (return the
    full-width ``cat(angles, angles)`` layout used by rotate-half attention) and
    ``device`` / ``dtype`` arguments.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch


class RopeCache:
    def __init__(self):
        self.cached_seq: Optional[int] = None
        self.cached_cos: Optional[torch.Tensor] = None
        self.cached_sin: Optional[torch.Tensor] = None
        self._key = None

    def build(
        self,
        seq_len: int,
        head_dim: int,
        theta: Optional[float] = None,
        *,
        inv_freq: Optional[torch.Tensor] = None,
        freq_tag=None,
        device=None,
        dtype: torch.dtype = torch.float32,
        duplicate: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if theta is None and inv_freq is None:
            raise ValueError("provide theta or inv_freq")
        device = torch.device(device) if device is not None else (inv_freq.device if inv_freq is not None else torch.device("cpu"))
        # freq_tag: hashable identity of a custom inv_freq (so equal frequencies hit the cache)
        key = (seq_len, head_dim, theta, None if inv_freq is None else freq_tag, device, dtype, duplicate)
        if key == self._key:
            return self.cached_cos, self.cached_sin

        position = torch.arange(seq_len, dtype=torch.float32, device=device)
        if inv_freq is None:
            freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
        else:
            freq = inv_freq.to(device=device, dtype=torch.float32)
        angles = torch.outer(position, freq)             # every (position, frequency) pair
        if duplicate:
            angles = torch.cat((angles, angles), dim=-1)
        cos, sin = torch.cos(angles).to(dtype), torch.sin(angles).to(dtype)

        self.cached_seq, self.cached_cos, self.cached_sin, self._key = seq_len, cos, sin, key
        return cos, sin

    # We return cos/sin lookup tables instead of rotating vectors here: Q and K are created
    # later in the attention layer and the same tables rotate both.
