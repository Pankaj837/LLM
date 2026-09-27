"""Dynamic NTK scaling of the RoPE base frequency.

Specification (team's "Dynamic NTK and RoPE Cache" work report):

  * Strict thresholding: for ``current_seq_len <= original_max_seq_len`` the base
    theta is returned unchanged.
  * Beyond the original length:
        scale = current_seq_len / original_max_seq_len
        theta = base_theta * scale ** (head_dim / (head_dim - 2))
    with the exponent computed in floating point.
  * State is instance-local; nothing global is mutated.

The formula and its behaviour are identical to the delivered implementation; this
version only adds argument validation. Reference values from the report
(base 10000, original 4096, head_dim 128): 8192 -> 20221.2617, 32768 -> 82684.6226
(asserted in ``tests/test_dynamic_ntk.py``).
"""
from __future__ import annotations


class DynamicNTK:
    def __init__(self, base_theta: float = 10000.0, original_max_seq_len: int = 4096, head_dim: int = 128):
        if base_theta <= 0:
            raise ValueError(f"base_theta must be > 0, got {base_theta}")
        if original_max_seq_len <= 0:
            raise ValueError(f"original_max_seq_len must be > 0, got {original_max_seq_len}")
        if head_dim <= 2 or head_dim % 2:
            raise ValueError(f"head_dim must be an even integer > 2, got {head_dim}")
        self.base_theta = float(base_theta)             # theta used in the model's original training
        self.original_max_seq_len = original_max_seq_len
        self.head_dim = head_dim

    def get_theta(self, current_seq_len: int) -> float:
        """Theta for a sequence of ``current_seq_len`` positions."""
        if current_seq_len <= self.original_max_seq_len:    # strict thresholding
            return self.base_theta
        scale = current_seq_len / self.original_max_seq_len
        exponent = float(self.head_dim) / float(self.head_dim - 2)
        return self.base_theta * (scale ** exponent)
