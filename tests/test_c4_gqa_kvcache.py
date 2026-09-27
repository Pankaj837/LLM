"""C4 Attention: grouped-query attention + KV cache."""
import math

import pytest
import torch

from llm.gqa import GroupedQueryAttention, KVCache, apply_rotary_pos_emb, repeat_kv

D, H, KV, HD = 64, 4, 2, 16


def _gqa():
    torch.manual_seed(0)
    return GroupedQueryAttention(D, H, KV, HD, max_seq_len=128).eval()


def _naive(g, x, pos):
    """Independent reference: explicit per-head softmax attention, head h uses KV head h // group."""
    B, T, _ = x.shape
    q = g.q_proj(x).view(B, T, H, HD).transpose(1, 2)
    k = g.k_proj(x).view(B, T, KV, HD).transpose(1, 2)
    v = g.v_proj(x).view(B, T, KV, HD).transpose(1, 2)
    c, s = g._get_rope(pos, x.dtype)
    q, k = apply_rotary_pos_emb(q, c, s), apply_rotary_pos_emb(k, c, s)
    out = torch.zeros(B, H, T, HD)
    for h in range(H):
        kv = h // (H // KV)
        sc = q[:, h] @ k[:, kv].transpose(-1, -2) / math.sqrt(HD)
        sc = sc.masked_fill(torch.triu(torch.ones(T, T, dtype=torch.bool), 1), float("-inf"))
        out[:, h] = sc.softmax(-1) @ v[:, kv]
    return g.o_proj(out.transpose(1, 2).reshape(B, T, D))


def test_matches_naive_reference():
    g, x = _gqa(), torch.randn(2, 12, D)
    pos = torch.arange(12)[None].expand(2, -1)
    with torch.no_grad():
        assert torch.allclose(g(x, pos), _naive(g, x, pos), atol=1e-5)


def test_repeat_kv_groups_are_contiguous():
    x = torch.arange(2).view(1, 2, 1, 1).float()          # kv heads 0 and 1
    assert repeat_kv(x, 2).flatten().tolist() == [0, 0, 1, 1]


def test_causality_future_tokens_do_not_change_past_outputs():
    g, x = _gqa(), torch.randn(1, 10, D)
    x2 = x.clone(); x2[:, 6:] = torch.randn(1, 4, D)
    pos = torch.arange(10)[None]
    with torch.no_grad():
        a, b = g(x, pos), g(x2, pos)
    assert torch.allclose(a[:, :6], b[:, :6], atol=1e-6) and not torch.allclose(a[:, 6:], b[:, 6:])


def test_kv_cache_incremental_equals_full_forward():
    g, x = _gqa(), torch.randn(2, 9, D)
    pos = torch.arange(9)[None].expand(2, -1)
    cache = KVCache(2, 9, KV, HD)
    with torch.no_grad():
        full = g(x, pos)
        inc = torch.cat([g(x[:, i:i + 1], pos[:, i:i + 1], kv_cache=cache) for i in range(9)], dim=1)
    assert torch.allclose(full, inc, atol=1e-5)


def test_kv_cache_chunked_prefill_equals_full_forward():
    """Prefill in two chunks (rectangular causal mask path) must equal one full pass."""
    g, x = _gqa(), torch.randn(1, 10, D)
    pos = torch.arange(10)[None]
    cache = KVCache(1, 10, KV, HD)
    with torch.no_grad():
        full = g(x, pos)
        out = torch.cat([g(x[:, :6], pos[:, :6], kv_cache=cache), g(x[:, 6:], pos[:, 6:], kv_cache=cache)], dim=1)
    assert torch.allclose(full, out, atol=1e-5)


def test_kv_cache_is_group_ratio_smaller_than_mha():
    a, b = KVCache(1, 2048, 2, 80), KVCache(1, 2048, 8, 80)
    assert a.memory_footprint_bytes * 4 == b.memory_footprint_bytes


def test_kv_cache_overflow_and_bad_shapes_are_rejected():
    c = KVCache(1, 4, KV, HD)
    with pytest.raises(AssertionError):
        c.update(torch.zeros(1, KV, 5, HD), torch.zeros(1, KV, 5, HD))
    with pytest.raises(AssertionError):
        c.update(torch.zeros(1, KV + 1, 1, HD), torch.zeros(1, KV + 1, 1, HD))


def test_kv_cache_reset():
    c = KVCache(1, 4, KV, HD)
    c.update(torch.ones(1, KV, 2, HD), torch.ones(1, KV, 2, HD))
    c.reset()
    assert c.seen_tokens == 0 and c.k_cache.abs().sum() == 0


def test_explicit_padding_mask_makes_pads_invisible():
    """Right-padded row must produce the same outputs for real tokens as the unpadded row."""
    g, x = _gqa(), torch.randn(2, 7, D)
    pos = torch.arange(7)[None].expand(2, -1)
    mask = torch.ones(2, 7, dtype=torch.bool); mask[1, 4:] = False
    with torch.no_grad():
        padded = g(x, pos, attention_mask=mask[:, None, None, :])
        alone = g(x[1:2, :4], pos[1:2, :4])
    assert torch.allclose(padded[1, :4], alone[0], atol=1e-5)


def test_output_is_finite_in_bf16():
    g = _gqa().to(torch.bfloat16)
    x = torch.randn(1, 8, D).to(torch.bfloat16)
    with torch.no_grad():
        assert torch.isfinite(g(x, torch.arange(8)[None]).float()).all()


def test_position_beyond_cache_extends_table():
    g = _gqa()
    x = torch.randn(1, 2, D)
    with torch.no_grad():
        out = g(x, torch.tensor([[500, 501]]))     # table only has 128 rows
    assert torch.isfinite(out).all()


def test_rotating_before_or_after_repeating_kv_heads_gives_the_same_result():
    """Documented in docs/ARCHITECTURE.md: the order is an efficiency choice (rotate 2 KV heads, not 8),
    not a correctness requirement, because the rotation depends only on position."""
    torch.manual_seed(0)
    g = _gqa()
    x = torch.randn(1, 9, D)
    pos = torch.arange(9)[None]
    k = g.k_proj(x).view(1, 9, KV, HD).transpose(1, 2)
    cos, sin = g._get_rope(pos, x.dtype)
    rotate_then_repeat = repeat_kv(apply_rotary_pos_emb(k, cos, sin), H // KV)
    repeat_then_rotate = apply_rotary_pos_emb(repeat_kv(k, H // KV), cos, sin)
    assert torch.equal(rotate_then_repeat, repeat_then_rotate)
