"""Deprecated shim kept for imports written against the first integration.

``YaRNRotaryEmbedding`` is now ``llm.rope.RotaryEmbedding`` (which also implements Dynamic NTK
and plain RoPE). New code should import from ``llm.rope``.
"""
from __future__ import annotations

from .rope import RotaryEmbedding, apply_rope, rotate_half


class YaRNRotaryEmbedding(RotaryEmbedding):
    """Alias of ``RotaryEmbedding`` for backward compatibility."""


__all__ = ["RotaryEmbedding", "YaRNRotaryEmbedding", "rotate_half", "apply_rope"]
