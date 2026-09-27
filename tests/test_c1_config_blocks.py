"""C1 Model configuration  +  C2 Core blocks (RMSNorm, SwiGLU, embedding/init)."""
import math

import pytest
import torch
import torch.nn.functional as F

from helpers import small_lm_config
from llm.model_config import ModelConfig
from llm.modules import RMSNorm, SwiGLUFeedForward


# ------------------------------------------------------------------ C1 config
def test_defaults_are_self_consistent():
    c = ModelConfig()
    assert c.hidden_dim == c.num_query_heads * c.head_dim
    assert c.num_query_heads % c.num_kv_heads == 0
    assert not c.use_yarn and not c.use_dynamic_ntk


def test_head_dim_mismatch_rejected():
    with pytest.raises((AssertionError, ValueError)):
        ModelConfig(hidden_dim=641)


def test_gqa_group_mismatch_rejected():
    with pytest.raises((AssertionError, ValueError)):
        ModelConfig(num_kv_heads=3)


def test_yarn_auto_enabled_by_scale():
    assert ModelConfig(yarn_scale_factor=2.0).use_yarn


def test_yarn_and_ntk_mutually_exclusive():
    with pytest.raises(ValueError):
        ModelConfig(yarn_scale_factor=2.0, use_dynamic_ntk=True)


def test_yarn_alpha_beta_order_enforced():
    with pytest.raises((AssertionError, ValueError)):
        ModelConfig(yarn_scale_factor=2.0, yarn_beta_fast=1.0, yarn_beta_slow=32.0)


@pytest.mark.finding("F-11")
@pytest.mark.parametrize("kw", [dict(yarn_scale_factor=0.5),
                                dict(yarn_scale_factor=2.0, yarn_original_max_position=8192),
                                dict(use_dynamic_ntk=True, yarn_original_max_position=8192),
                                dict(head_dim=81, hidden_dim=648, num_query_heads=8)])
def test_nonsensical_scaling_rejected(kw):
    """Scale < 1 shrinks the context; yarn_original > max_position is meaningless."""
    with pytest.raises((ValueError, AssertionError)):
        ModelConfig(**kw)


# ------------------------------------------------------------------ C2 blocks
def test_rmsnorm_matches_torch_reference():
    x = torch.randn(3, 5, 64)
    assert torch.allclose(RMSNorm(64)(x), torch.nn.RMSNorm(64, eps=1e-6)(x), atol=1e-6)


def test_rmsnorm_preserves_dtype_and_handles_large_values():
    x = (torch.randn(2, 4, 64) * 1e4).to(torch.bfloat16)
    y = RMSNorm(64)(x)
    assert y.dtype == torch.bfloat16 and torch.isfinite(y.float()).all()


def test_rmsnorm_zero_input_is_finite():
    assert torch.isfinite(RMSNorm(64)(torch.zeros(1, 2, 64))).all()


def test_swiglu_matches_manual_formula():
    ffn = SwiGLUFeedForward(32, 64)
    x = torch.randn(2, 3, 32)
    manual = ffn.W_down(F.silu(ffn.W_gate(x)) * ffn.W_up(x))
    assert torch.allclose(ffn(x), manual)


@pytest.mark.finding("F-06")
def test_fresh_model_starts_near_uniform_loss():
    """A model built by the constructor must start at CE ~ ln(V). With the default
    N(0,1) embedding tied to the output head, initial CE is ~570 (logit std ~26)."""
    from llm.model import TransformerModel
    torch.manual_seed(0)
    cfg = small_lm_config(vocab_size=8000, hidden_dim=640, num_query_heads=8, head_dim=80, num_kv_heads=2,
                          intermediate_dim=1664, num_layers=2, max_position_embeddings=2048,
                          yarn_original_max_position=2048)
    m = TransformerModel(cfg).eval()
    ids = torch.randint(0, 8000, (4, 64))
    with torch.no_grad():
        lg = m(ids)
    ce = F.cross_entropy(lg[:, :-1].reshape(-1, 8000), ids[:, 1:].reshape(-1)).item()
    assert abs(ce - math.log(8000)) < 1.0, f"initial CE {ce:.1f} vs ln V {math.log(8000):.2f}"


def test_weight_tying_has_no_separate_output_head():
    from llm.model import TransformerModel
    m = TransformerModel(small_lm_config())
    assert not any(n.startswith("output") or n.startswith("lm_head") for n, _ in m.named_parameters())
