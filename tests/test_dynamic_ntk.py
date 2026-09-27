"""Dynamic NTK (team spec), RopeCache, and the RoPE modes inside the full model."""
import pytest
import torch

from helpers import small_lm_config
from llm.dynamic_ntk import DynamicNTK
from llm.model import TransformerModel
from llm.rope import RotaryEmbedding
from llm.rope_cache import RopeCache


# ------------------------------------------------------------------ spec: DynamicNTK
def test_report_reference_values():
    """Values printed in the team's Dynamic NTK report (base 10000, original 4096, head_dim 128)."""
    n = DynamicNTK(10000.0, 4096, 128)
    assert n.get_theta(2048) == 10000.0
    assert n.get_theta(4096) == 10000.0
    assert n.get_theta(8192) == pytest.approx(20221.2617, abs=1e-3)
    assert n.get_theta(32768) == pytest.approx(82684.6226, abs=1e-3)


def test_strict_threshold_is_exclusive_at_the_original_length():
    n = DynamicNTK(10000.0, 4096, 128)
    assert n.get_theta(4096) == 10000.0 and n.get_theta(4097) > 10000.0


def test_formula_matches_definition():
    n = DynamicNTK(10000.0, 2048, 80)
    L = 6000
    assert n.get_theta(L) == pytest.approx(10000.0 * (L / 2048) ** (80 / 78), rel=1e-12)


def test_theta_is_monotonic_in_length():
    n = DynamicNTK(10000.0, 2048, 80)
    thetas = [n.get_theta(L) for L in range(2048, 20000, 500)]
    assert thetas == sorted(thetas)


@pytest.mark.parametrize("kw", [dict(base_theta=0), dict(original_max_seq_len=0), dict(head_dim=2), dict(head_dim=63)])
def test_invalid_arguments_rejected(kw):
    with pytest.raises(ValueError):
        DynamicNTK(**kw)


def test_instances_do_not_share_state():
    a, b = DynamicNTK(10000.0, 100, 64), DynamicNTK(50000.0, 100, 64)
    assert a.get_theta(400) != b.get_theta(400)


# ------------------------------------------------------------------ spec: RopeCache
def test_cache_returns_same_object_when_request_is_unchanged():
    c = RopeCache()
    a = c.build(64, 32, 10000.0)
    b = c.build(64, 32, 10000.0)
    assert a[0] is b[0] and a[1] is b[1]


def test_cache_recomputes_when_length_changes():
    c = RopeCache()
    assert c.build(64, 32, 10000.0)[0].shape[0] == 64 and c.build(128, 32, 10000.0)[0].shape[0] == 128


def test_cache_is_keyed_on_theta_not_only_length():
    """The delivered version keyed on seq_len only and would return stale tables for a new theta."""
    c = RopeCache()
    a = c.build(64, 32, 10000.0)[0].clone()
    b = c.build(64, 32, 50000.0)[0]
    assert not torch.allclose(a, b)


def test_cache_default_layout_is_half_width_as_delivered():
    cos, sin = RopeCache().build(10, 32, 10000.0)
    assert cos.shape == (10, 16) and sin.shape == (10, 16)


def test_cache_values_are_cos_and_sin_of_position_times_frequency():
    cos, sin = RopeCache().build(8, 16, 10000.0)
    freq = 1.0 / (10000.0 ** (torch.arange(0, 16, 2).float() / 16))
    angles = torch.outer(torch.arange(8).float(), freq)
    assert torch.allclose(cos, angles.cos()) and torch.allclose(sin, angles.sin())


def test_cache_duplicate_layout_is_cat_of_halves():
    half = RopeCache().build(8, 16, 10000.0)[0]
    full = RopeCache().build(8, 16, 10000.0, duplicate=True)[0]
    assert torch.equal(full, torch.cat([half, half], dim=-1))


def test_cache_requires_theta_or_inv_freq():
    with pytest.raises(ValueError):
        RopeCache().build(8, 16)


# ------------------------------------------------------------------ integration: RotaryEmbedding
def _dyn_cfg(**kw):
    return small_lm_config(use_dynamic_ntk=True, max_position_embeddings=512, yarn_original_max_position=64, **kw)


def test_dynamic_ntk_is_identity_below_and_at_the_trained_length():
    plain = RotaryEmbedding(small_lm_config(max_position_embeddings=512, yarn_original_max_position=64))
    dyn = RotaryEmbedding(_dyn_cfg())
    pos = torch.arange(64)[None]
    for ctx in (10, 64):
        a, b = plain.get_cos_sin(pos, torch.float32, ctx), dyn.get_cos_sin(pos, torch.float32, ctx)
        assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


def test_dynamic_ntk_changes_rotation_beyond_the_trained_length_only_for_slow_dims():
    dyn = RotaryEmbedding(_dyn_cfg())
    pos = torch.arange(200)[None]
    c_short = dyn.get_cos_sin(pos, torch.float32, 64)[0]
    c_long = dyn.get_cos_sin(pos, torch.float32, 200)[0]
    assert not torch.allclose(c_short, c_long)
    # the first frequency pair (fastest) is scaled least, the last most
    d = (c_short - c_long).abs()[0, 0]
    half = d.shape[-1] // 2
    assert d[:, half - 1].max() <= d[:, half // 2].max() + 1.0   # sanity: differences are bounded


def test_theta_for_follows_the_spec_formula():
    dyn = RotaryEmbedding(_dyn_cfg())
    assert dyn.theta_for(64) == 10000.0
    assert dyn.theta_for(256) == pytest.approx(10000.0 * (256 / 64) ** (16 / 14), rel=1e-9)


def test_get_cos_sin_equals_table_gather_and_has_expected_shape():
    r = RotaryEmbedding(small_lm_config())
    pos = torch.tensor([[0, 5, 7], [1, 2, 3]])
    cos, sin = r.get_cos_sin(pos, torch.float32, 16)
    assert cos.shape == (2, 1, 3, 16)
    table_c, _ = r.build_cache(16, torch.device("cpu"), torch.float32)
    assert torch.equal(cos[1, 0], table_c[pos[1]])


def test_state_is_instance_local():
    a, b = RotaryEmbedding(_dyn_cfg()), RotaryEmbedding(_dyn_cfg())
    a.get_cos_sin(torch.arange(300)[None], torch.float32, 300)
    assert a._cache is not b._cache and b._cache.cached_seq != 300


def test_casting_the_module_to_bf16_does_not_corrupt_frequencies():
    """module.to(bf16) casts registered buffers; the fp32 master copy must be unaffected."""
    m = TransformerModel(small_lm_config(num_layers=1)).to(torch.bfloat16)
    ref = RotaryEmbedding(small_lm_config(num_layers=1))
    pos = torch.arange(200)[None]
    got = m.rope.get_cos_sin(pos, torch.float32, 256)[0]
    assert torch.allclose(got, ref.get_cos_sin(pos, torch.float32, 256)[0])


# ------------------------------------------------------------------ integration: full model
def _weights_shared(cfg_a, cfg_b):
    torch.manual_seed(0)
    a = TransformerModel(cfg_a).eval()
    b = TransformerModel(cfg_b).eval()
    b.load_state_dict(a.state_dict())
    return a, b


def test_dynamic_ntk_model_equals_plain_rope_model_within_trained_length():
    a, b = _weights_shared(small_lm_config(max_position_embeddings=512, yarn_original_max_position=64), _dyn_cfg())
    ids = torch.randint(0, 1722, (2, 60))
    with torch.no_grad():
        assert torch.equal(a(ids), b(ids))


def test_dynamic_ntk_model_differs_from_plain_beyond_trained_length():
    a, b = _weights_shared(small_lm_config(max_position_embeddings=512, yarn_original_max_position=64), _dyn_cfg())
    ids = torch.randint(0, 1722, (1, 200))
    with torch.no_grad():
        assert not torch.allclose(a(ids), b(ids), atol=1e-4)


@pytest.mark.parametrize("mode_kw", [
    {},
    dict(yarn_scale_factor=2.0),
    dict(use_dynamic_ntk=True),
], ids=["rope", "yarn", "dynamic_ntk"])
def test_cached_decode_equals_uncached_decode_across_the_trained_length(mode_kw):
    """Crossing the trained length mid-generation must not change results between the two decode paths."""
    cfg = small_lm_config(max_position_embeddings=256, yarn_original_max_position=64, **mode_kw)
    torch.manual_seed(1)
    m = TransformerModel(cfg).eval()
    p = torch.randint(0, 1722, (2, 12))
    a = m.generate(p, max_new_tokens=90, temperature=0.0, use_cache=True)
    b = m.generate(p, max_new_tokens=90, temperature=0.0, use_cache=False)
    assert torch.equal(a, b)


def test_chunked_prefill_equals_single_pass_with_dynamic_ntk():
    cfg = _dyn_cfg()
    m = TransformerModel(cfg).eval()
    from llm.gqa import KVCache
    ids = torch.randint(0, 1722, (1, 100))

    def caches():
        return [KVCache(1, 100, cfg.num_kv_heads, cfg.head_dim) for _ in range(cfg.num_layers)]

    with torch.no_grad():
        full = m(ids, kv_caches=caches())
        c = caches()
        part = torch.cat([m(ids[:, :60], kv_caches=c), m(ids[:, 60:], kv_caches=c)], dim=1)
    assert torch.allclose(full, part, atol=1e-4)


def test_forward_path_has_no_host_device_sync(monkeypatch):
    """`.item()` forces a device->host sync (blocks CUDA graphs / torch.compile and stalls the GPU).
    Prefill and every decode step must be free of it."""
    m = TransformerModel(small_lm_config()).eval()
    p = torch.randint(0, 1722, (1, 6))

    def boom(self):
        raise AssertionError("Tensor.item() called in the forward/decode path")

    monkeypatch.setattr(torch.Tensor, "item", boom)
    out = m.generate(p, max_new_tokens=8, temperature=0.0)
    assert out.shape == (1, 14)


def test_explicit_position_ids_beyond_context_still_work():
    m = TransformerModel(small_lm_config()).eval()
    ids = torch.randint(0, 1722, (1, 4))
    with torch.no_grad():
        out = m(ids, position_ids=torch.tensor([[300, 301, 302, 303]]))
    assert torch.isfinite(out).all()


def test_planned_context_is_a_property_of_the_call_not_hidden_state():
    """Two calls with the same inputs and the same planned context give identical results even
    after an unrelated call with a much longer planned context."""
    cfg = _dyn_cfg()
    m = TransformerModel(cfg).eval()
    ids = torch.randint(0, 1722, (1, 80))
    with torch.no_grad():
        first = m(ids, rope_context_len=100)
        m(ids, rope_context_len=400)
        again = m(ids, rope_context_len=100)
    assert torch.equal(first, again)


# ------------------------------------------------------------------ logit-level policy tests (tokens alone are too insensitive)
def test_planned_context_changes_logits_and_cached_uncached_logits_agree():
    """Sensitivity + equivalence: the planned context must matter (else these tests prove nothing) and
    cached prefill must equal an uncached pass that uses the same planned context."""
    from llm.gqa import KVCache
    cfg = _dyn_cfg()
    torch.manual_seed(2)
    m = TransformerModel(cfg).eval()
    with torch.no_grad():                    # a random init has near-uniform attention and barely notices
        for layer in m.layers:               # rotation changes; sharpen it so the test is sensitive
            layer.attn.q_proj.weight.mul_(40)
            layer.attn.k_proj.weight.mul_(40)
    ids = torch.randint(0, 1722, (1, 100))
    planned = 160
    with torch.no_grad():
        unc_planned = m(ids, rope_context_len=planned)[:, -1]
        unc_default = m(ids)[:, -1]                                    # context = 100 -> different theta
        caches = [KVCache(1, planned, cfg.num_kv_heads, cfg.head_dim) for _ in range(cfg.num_layers)]
        cached = m(ids, kv_caches=caches)[:, -1]                        # context defaults to cache capacity = 160
    assert (unc_planned - unc_default).abs().max() > 100 * 1e-4         # sensitivity >> equivalence tolerance
    assert torch.allclose(cached, unc_planned, atol=1e-4)


@pytest.mark.parametrize("use_cache", [True, False])
def test_generate_uses_one_planned_context_for_every_forward_call(use_cache):
    cfg = _dyn_cfg()
    m = TransformerModel(cfg).eval()
    seen = []
    real = m.forward

    def spy(*a, **k):
        kc = k.get("kv_caches")
        seen.append(k.get("rope_context_len", kc[0].max_seq_len if kc else None))
        return real(*a, **k)

    m.forward = spy
    p = torch.randint(0, 1722, (1, 10))
    m.generate(p, max_new_tokens=30, temperature=0.0, use_cache=use_cache)
    assert len(seen) >= 30 and set(seen) == {40}, set(seen)
