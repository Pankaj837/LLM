"""Unified model configuration.

One dataclass holds every hyperparameter of the model:
- vocabulary and embedding
- core architecture (GQA, SwiGLU, RMSNorm)
- RoPE and its context-extension modes (Dynamic NTK or YaRN, mutually exclusive)
- regularisation

Defaults describe the 51.5M-parameter reference model (11 layers, d=640, 8 query / 2 KV
heads). The configuration is serialised into every checkpoint (see ``checkpoint.py``).
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class ModelConfig:
    # ── Vocabulary / Embedding ──────────────────────────────────────────
    vocab_size: int = 8000                # must equal the tokenizer's vocabulary size

    # ── Core transformer dimensions ─────────────────────────────────────
    hidden_dim: int = 640                 # d_model
    num_layers: int = 11
    num_query_heads: int = 8
    num_kv_heads: int = 2                 # GQA: fewer KV heads than query heads
    head_dim: int = 80                    # hidden_dim / num_query_heads
    intermediate_dim: int = 1664          # SwiGLU FFN intermediate size

    # ── Positional encoding (RoPE) ──────────────────────────────────────
    max_position_embeddings: int = 2048   # context window the model is configured for
    rope_theta: float = 10000.0           # RoPE base frequency

    # ── Context extension: at most one of the two ───────────────────────
    # YaRN (Peng et al., 2023): static frequency interpolation + attention scaling.
    # Enabled automatically when yarn_scale_factor > 1.
    use_yarn: bool = False
    yarn_scale_factor: float = 1.0
    yarn_original_max_position: int = 2048   # context length the model was TRAINED at
    yarn_beta_fast: float = 32.0
    yarn_beta_slow: float = 1.0
    # Dynamic NTK: theta = rope_theta * (L / L_orig) ** (d / (d - 2)) once the planned
    # context L exceeds L_orig (= yarn_original_max_position); unchanged below it.
    use_dynamic_ntk: bool = False

    # ── Regularisation ──────────────────────────────────────────────────
    attention_dropout: float = 0.0
    rms_norm_eps: float = 1e-6

    # ── Convenience aliases used by the RoPE modules ────────────────────
    @property
    def theta_base(self) -> float:
        return self.rope_theta

    @property
    def trained_context_len(self) -> int:
        """Context length the model was trained at (threshold for Dynamic NTK / YaRN)."""
        return self.yarn_original_max_position

    @property
    def yarn_alpha(self) -> float:
        return self.yarn_beta_slow

    @property
    def yarn_beta(self) -> float:
        return self.yarn_beta_fast

    @property
    def rope_mode(self) -> str:
        return "yarn" if self.use_yarn else "dynamic_ntk" if self.use_dynamic_ntk else "rope"

    # ── Validation ──────────────────────────────────────────────────────
    def __post_init__(self):
        # explicit exceptions (not `assert`) so validation survives `python -O`
        if self.hidden_dim != self.num_query_heads * self.head_dim:
            raise ValueError(
                f"hidden_dim ({self.hidden_dim}) must equal "
                f"num_query_heads * head_dim ({self.num_query_heads * self.head_dim})"
            )
        if self.num_query_heads % self.num_kv_heads:
            raise ValueError(
                f"num_query_heads ({self.num_query_heads}) must be divisible by "
                f"num_kv_heads ({self.num_kv_heads})"
            )
        if self.head_dim % 2:
            raise ValueError(f"head_dim must be even for RoPE, got {self.head_dim}")
        if self.yarn_scale_factor < 1.0:
            raise ValueError(f"yarn_scale_factor must be >= 1.0, got {self.yarn_scale_factor}")

        if self.yarn_scale_factor > 1.0:
            self.use_yarn = True
        if self.use_yarn and self.use_dynamic_ntk:
            raise ValueError("use_yarn and use_dynamic_ntk are mutually exclusive.")

        if self.use_yarn or self.use_dynamic_ntk:
            if self.yarn_original_max_position > self.max_position_embeddings:
                raise ValueError(
                    f"yarn_original_max_position ({self.yarn_original_max_position}) must be <= "
                    f"max_position_embeddings ({self.max_position_embeddings})"
                )
        if self.use_yarn and not self.yarn_alpha < self.yarn_beta:
            raise ValueError(
                f"yarn_beta_slow ({self.yarn_alpha}) must be strictly less than yarn_beta_fast "
                f"({self.yarn_beta}); the ramp denominator would be zero or negative."
            )

    # ── Serialisation ───────────────────────────────────────────────────
    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ModelConfig":
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown ModelConfig fields: {sorted(unknown)}")
        return cls(**d)
