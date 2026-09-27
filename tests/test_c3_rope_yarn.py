"""C3 Positional encoding: RoPE, NTK-aware scaling, YaRN."""
import math

import pytest
import torch

from llm.gqa import apply_rotary_pos_emb
from llm.model_config import ModelConfig
from llm.rope import RotaryEmbedding


def _ref_yarn_inv_freq(dim, base, s, L, alpha=1.0, beta=32.0):
    """Reference: Peng et al. 2023 / HF `_compute_yarn_parameters` (ramp linear in dim index)."""
    pos = base ** (torch.arange(0, dim, 2).float() / dim)
    extrap, interp = 1.0 / pos, 1.0 / (s * pos)

    def corr(n): return dim * math.log(L / (n * 2 * math.pi)) / (2 * math.log(base))
    low, high = max(math.floor(corr(beta)), 0), min(math.ceil(corr(alpha)), dim - 1)
    ramp = ((torch.arange(dim // 2).float() - low) / max(high - low, 1e-3)).clamp(0, 1)
    return interp * ramp + extrap * (1 - ramp)


def _base_inv_freq(cfg):
    return 1.0 / (cfg.rope_theta ** (torch.arange(0, cfg.head_dim, 2).float() / cfg.head_dim))


def _yarn_cfg(s):
    return ModelConfig(yarn_scale_factor=s, max_position_embeddings=2048 * int(s), yarn_original_max_position=2048)


def test_rope_relative_position_property():
    """<R(m)q, R(n)k> depends only on m-n."""
    r = RotaryEmbedding(ModelConfig())
    q, k = torch.randn(1, 1, 1, 80), torch.randn(1, 1, 1, 80)

    def dot(m, n):
        c, s = r.cos_cached, r.sin_cached
        a = apply_rotary_pos_emb(q, c[m][None, None, None], s[m][None, None, None])
        b = apply_rotary_pos_emb(k, c[n][None, None, None], s[n][None, None, None])
        return (a * b).sum().item()

    assert dot(5, 2) == pytest.approx(dot(105, 102), abs=1e-3) == pytest.approx(dot(1005, 1002), abs=1e-3)


def test_rope_preserves_vector_norm():
    r = RotaryEmbedding(ModelConfig())
    x = torch.randn(2, 8, 16, 80)
    c, s = r.cos_cached[:16][None, None], r.sin_cached[:16][None, None]
    assert torch.allclose(apply_rotary_pos_emb(x, c, s).norm(dim=-1), x.norm(dim=-1), atol=1e-4)


def test_base_inv_freq_formula():
    cfg = ModelConfig()
    assert torch.allclose(RotaryEmbedding(cfg).inv_freq, _base_inv_freq(cfg))


@pytest.mark.parametrize("s", [2.0, 4.0, 8.0])
def test_yarn_extremes_are_correct(s):
    """Highest-frequency dims untouched, lowest-frequency dims divided by s (true for the code AND the reference)."""
    cfg = _yarn_cfg(s)
    f, base = RotaryEmbedding(cfg).inv_freq, _base_inv_freq(cfg)
    assert torch.allclose(f[:4], base[:4], rtol=1e-4)
    assert torch.allclose(f[-4:], base[-4:] / s, rtol=1e-4)


@pytest.mark.finding("F-04")
@pytest.mark.parametrize("s", [2.0, 4.0, 8.0])
def test_yarn_inv_freq_matches_reference(s):
    """The blend ramp must be linear in dimension index (paper/HF). The delivered code is
    linear in wavelength -> up to 25-71% relative error in the blend zone."""
    cfg = _yarn_cfg(s)
    ref = _ref_yarn_inv_freq(cfg.head_dim, cfg.rope_theta, s, 2048)
    assert torch.allclose(RotaryEmbedding(cfg).inv_freq, ref, rtol=1e-3)


@pytest.mark.finding("F-03")
@pytest.mark.parametrize("s", [2.0, 4.0, 8.0])
def test_yarn_attention_multiplier_direction_and_value(s):
    """Paper: q,k scaled by 0.1 ln s + 1 => logits x (0.1 ln s + 1)^2 (>1, sharper).
    Delivered code multiplies logits by 1/sqrt(0.1 ln s + 1) (<1)."""
    mult = RotaryEmbedding(_yarn_cfg(s)).attention_temperature()
    assert mult > 1.0
    assert mult == pytest.approx((0.1 * math.log(s) + 1.0) ** 2, rel=1e-6)


def test_no_scaling_is_identity():
    r = RotaryEmbedding(ModelConfig())
    assert r.attention_temperature() == 1.0


def test_cache_extends_beyond_initial_length():
    r = RotaryEmbedding(ModelConfig())
    cos, sin = r.build_cache(3000, torch.device("cpu"), torch.float32)
    assert cos.shape == (3000, 80) and torch.isfinite(cos).all() and torch.isfinite(sin).all()


def test_cache_is_consistent_across_growth():
    """Positions < 2048 must not change when the table is grown past 2048."""
    r = RotaryEmbedding(ModelConfig())
    before = r.cos_cached[:100].clone()
    r.build_cache(4096, torch.device("cpu"), torch.float32)
    after, _ = r.build_cache(100, torch.device("cpu"), torch.float32)
    assert torch.allclose(before, after)
