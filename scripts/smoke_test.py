"""
Smoke test for the GQA Transformer + Conditional Reward Model pipeline.

Validates every core module can be instantiated, produces correct shapes,
and passes basic numerical sanity checks — all WITHOUT requiring checkpoint
or .tok vocabulary files.

Run:  python smoke_test.py
"""

from __future__ import annotations

import math
import os
import sys
import time

import torch
import torch.nn.functional as F

# Ensure safe UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Ensure project root is importable
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from llm.model_config import ModelConfig
from llm.modules import RMSNorm, SwiGLUFeedForward
from llm.rope import RotaryEmbedding
from llm.gqa import GroupedQueryAttention, KVCache
from llm.model import TransformerModel
from llm.tokenizer import BPETokenizer
from llm.pipeline import LLMPipeline

# Conditional Reward Model imports
from llm.reward.transformer import TransformerConfig, TransformerBackbone
from llm.reward.reward_model import (
    FiLMConditioner,
    ConditionalRewardModel,
    preference_loss,
)
from llm.reward.preferences import (
    CONDITIONS,
    COND_TO_ID,
    build_dataset,
    build_responses,
    prefers,
    split,
    condition_blind_ceiling,
)


# ── Helpers ──────────────────────────────────────────────────────────────

_PASS_COUNT = 0
_FAIL_COUNT = 0


def _report(name: str, passed: bool, detail: str = ""):
    global _PASS_COUNT, _FAIL_COUNT
    if passed:
        _PASS_COUNT += 1
        tag = "[PASS]"
    else:
        _FAIL_COUNT += 1
        tag = "[FAIL]"
    suffix = f" — {detail}" if detail else ""
    print(f"  {tag} {name}{suffix}")


def _no_nan_inf(tensor: torch.Tensor) -> bool:
    return not (torch.isnan(tensor).any() or torch.isinf(tensor).any())


# ══════════════════════════════════════════════════════════════════════════
#  PART 1 — GQA Transformer Pipeline
# ══════════════════════════════════════════════════════════════════════════

def test_model_config():
    """ModelConfig creation and validation."""
    cfg = ModelConfig()
    ok = (
        cfg.hidden_dim == cfg.num_query_heads * cfg.head_dim
        and cfg.num_query_heads % cfg.num_kv_heads == 0
        and cfg.use_yarn is False
        and cfg.use_dynamic_ntk is False
    )
    _report("ModelConfig defaults", ok)

    yarn_cfg = ModelConfig(yarn_scale_factor=2.0, yarn_original_max_position=2048)
    _report("ModelConfig YaRN auto-enable", yarn_cfg.use_yarn is True)


def test_rms_norm_and_swiglu():
    """RMSNorm and SwiGLUFeedForward shape + numerics."""
    B, S, D, I = 2, 16, 640, 1664
    x = torch.randn(B, S, D)

    norm = RMSNorm(D)
    out_norm = norm(x)
    ok_norm = out_norm.shape == (B, S, D) and _no_nan_inf(out_norm)
    _report("RMSNorm shape + numerics", ok_norm)

    ffn = SwiGLUFeedForward(D, I)
    out_ffn = ffn(x)
    ok_ffn = out_ffn.shape == (B, S, D) and _no_nan_inf(out_ffn)
    _report("SwiGLUFeedForward shape + numerics", ok_ffn)


def test_rotary_embedding():
    """RotaryEmbedding: base, YaRN, Dynamic NTK."""
    base_cfg = ModelConfig()
    base_rope = RotaryEmbedding(base_cfg)
    cos, sin = base_rope.forward(32)
    ok_base = cos.shape == (32, base_cfg.head_dim) and base_rope.attention_temperature() == 1.0
    _report("RotaryEmbedding base", ok_base)

    yarn_cfg = ModelConfig(
        yarn_scale_factor=2.0, max_position_embeddings=4096, yarn_original_max_position=2048,
    )
    yarn_rope = RotaryEmbedding(yarn_cfg)
    temp = yarn_rope.attention_temperature()
    # Peng et al. 2023: logits are multiplied by 1/t = (0.1*ln(s) + 1)^2 (>= 1, sharper)
    expected_temp = (0.1 * math.log(2.0) + 1.0) ** 2
    ok_yarn = abs(temp - expected_temp) < 1e-6 and temp > 1.0
    _report("RotaryEmbedding YaRN", ok_yarn, f"temp={temp:.4f}")

    ntk_cfg = ModelConfig(use_dynamic_ntk=True, max_position_embeddings=4096)
    ntk_rope = RotaryEmbedding(ntk_cfg)
    cos_ntk, _ = ntk_rope.forward(32)
    ok_ntk = cos_ntk.shape == (32, ntk_cfg.head_dim) and ntk_rope.attention_temperature() == 1.0
    _report("RotaryEmbedding Dynamic NTK", ok_ntk)


def test_grouped_query_attention():
    """GroupedQueryAttention forward pass."""
    B, S, D = 2, 32, 640
    gqa = GroupedQueryAttention(d_model=D, n_heads=8, n_kv_heads=2, head_dim=80)
    gqa.eval()
    x = torch.randn(B, S, D)
    pos = torch.arange(S).unsqueeze(0).expand(B, -1)

    with torch.no_grad():
        out = gqa(x, position_ids=pos)

    ok = out.shape == (B, S, D) and _no_nan_inf(out)
    _report("GQA forward shape + numerics", ok)


def test_kv_cache():
    """KVCache memory reduction + incremental decoding parity."""
    gqa_cache = KVCache(max_batch_size=1, max_seq_len=2048, n_kv_heads=2, head_dim=80)
    mha_cache = KVCache(max_batch_size=1, max_seq_len=2048, n_kv_heads=8, head_dim=80)
    ratio = gqa_cache.memory_footprint_bytes / mha_cache.memory_footprint_bytes
    _report("KVCache 4x memory reduction", ratio == 0.25, f"ratio={ratio}")

    B, S, D = 2, 10, 640
    gqa = GroupedQueryAttention(d_model=D, n_heads=8, n_kv_heads=2, head_dim=80)
    gqa.eval()
    x = torch.randn(B, S, D)
    pos = torch.arange(S).unsqueeze(0).expand(B, -1)

    with torch.no_grad():
        full_out = gqa(x, position_ids=pos, use_causal_mask=True)

    cache = KVCache(max_batch_size=B, max_seq_len=S, n_kv_heads=2, head_dim=80)
    inc_outs = []
    with torch.no_grad():
        for i in range(S):
            out_i = gqa(
                x[:, i:i+1, :], position_ids=pos[:, i:i+1],
                kv_cache=cache, use_causal_mask=True,
            )
            inc_outs.append(out_i)
    stacked = torch.cat(inc_outs, dim=1)

    ok = torch.allclose(full_out, stacked, atol=1e-5, rtol=1e-5)
    diff = (full_out - stacked).abs().max().item()
    _report("KVCache incremental parity", ok, f"max_diff={diff:.2e}")


def test_transformer_forward():
    """TransformerModel full forward pass."""
    cfg = ModelConfig(num_layers=2)
    model = TransformerModel(cfg)
    model.eval()

    B, S = 1, 32
    ids = torch.randint(0, cfg.vocab_size, (B, S))

    with torch.no_grad():
        logits = model(ids)

    expected = (B, S, cfg.vocab_size)
    ok = logits.shape == expected and _no_nan_inf(logits)
    _report("TransformerModel forward", ok, f"shape={logits.shape}")


def test_transformer_generate():
    """TransformerModel greedy generation: cached == uncached."""
    torch.manual_seed(42)
    cfg = ModelConfig(num_layers=2)
    model = TransformerModel(cfg)
    model.eval()

    prompt = torch.tensor([[10, 25, 42, 99, 105]], dtype=torch.long)
    gen_cached = model.generate(prompt, max_new_tokens=6, temperature=0.0, use_cache=True)
    gen_uncached = model.generate(prompt, max_new_tokens=6, temperature=0.0, use_cache=False)

    ok = torch.equal(gen_cached, gen_uncached)
    _report("Generate cached vs uncached parity", ok)


def test_tokenizer_roundtrip():
    """BPETokenizer encode/decode roundtrip."""
    tok = BPETokenizer()
    samples = ["Hello, world!", "don't they've we'll", "    def test():\n        pass"]

    all_ok = True
    for text in samples:
        if tok.decode(tok.encode(text)) != text:
            all_ok = False

    sp_text = "Start<|endoftext|>End"
    sp_ids = tok.encode(sp_text)
    if tok.decode(sp_ids) != sp_text or tok.eot_token_id not in sp_ids:
        all_ok = False

    _report("BPETokenizer roundtrip", all_ok)


def test_pipeline_no_checkpoint():
    """LLMPipeline instantiation and generation without a checkpoint."""
    cfg = ModelConfig(num_layers=2)
    model = TransformerModel(cfg)
    tok = BPETokenizer()

    pipe = LLMPipeline(model=model, tokenizer=tok, config=cfg)
    output = pipe.generate("Hello", max_new_tokens=5, temperature=0.0)

    ok = isinstance(output, str) and len(output) > 0
    _report("LLMPipeline end-to-end (no ckpt)", ok, f"output_len={len(output)}")


def test_causal_masking():
    """Causal mask: future tokens produce zero gradients."""
    gqa = GroupedQueryAttention(d_model=640, n_heads=8, n_kv_heads=2, head_dim=80)

    x = torch.randn(1, 5, 640, requires_grad=True)
    pos = torch.arange(5).unsqueeze(0)

    out = gqa(x, position_ids=pos, use_causal_mask=True)
    out[0, 2, :].sum().backward()

    grad_3 = x.grad[0, 3, :].abs().max().item()
    grad_4 = x.grad[0, 4, :].abs().max().item()
    grad_0 = x.grad[0, 0, :].abs().max().item()

    ok = grad_3 == 0.0 and grad_4 == 0.0 and grad_0 > 0.0
    _report("Causal masking (future grads=0)", ok, f"g3={grad_3}, g4={grad_4}")


# ══════════════════════════════════════════════════════════════════════════
#  PART 2 — Conditional Reward Model
# ══════════════════════════════════════════════════════════════════════════

def test_reward_transformer_backbone():
    """TransformerBackbone forward pass produces (B, T, d_model)."""
    cfg = TransformerConfig(vocab_size=500, d_model=64, n_layer=2, n_head=4, max_seq_len=64)
    backbone = TransformerBackbone(cfg)
    backbone.eval()

    B, T = 2, 16
    ids = torch.randint(0, cfg.vocab_size, (B, T))
    mask = torch.ones(B, T, dtype=torch.long)

    with torch.no_grad():
        out = backbone(ids, attention_mask=mask)

    ok = out.shape == (B, T, cfg.d_model) and _no_nan_inf(out)
    _report("Reward TransformerBackbone forward", ok, f"shape={out.shape}")


def test_film_conditioner():
    """FiLM conditioning: starts at identity (gamma=1, beta=0)."""
    num_cond, cond_dim, d_model = 3, 32, 64
    film = FiLMConditioner(num_cond, cond_dim, d_model)

    B = 4
    h = torch.randn(B, d_model)
    cond_ids = torch.randint(0, num_cond, (B,))

    out = film(h, cond_ids)
    ok_shape = out.shape == (B, d_model)

    # At init, gamma ≈ 1 and beta ≈ 0, so output ≈ input
    ok_identity = torch.allclose(out, h, atol=1e-5)

    _report("FiLM conditioner shape", ok_shape)
    _report("FiLM init ≈ identity", ok_identity)


def test_conditional_reward_model_film():
    """ConditionalRewardModel (FiLM): forward produces scalar per sample."""
    cfg = TransformerConfig(vocab_size=500, d_model=64, n_layer=2, n_head=4, max_seq_len=64)
    model = ConditionalRewardModel(cfg, num_conditions=3, cond_dim=32, conditioning="film")
    model.eval()

    B, T = 4, 16
    ids = torch.randint(0, cfg.vocab_size, (B, T))
    mask = torch.ones(B, T, dtype=torch.long)
    cond = torch.randint(0, 3, (B,))

    with torch.no_grad():
        scores = model(ids, mask, cond)

    ok = scores.shape == (B,) and _no_nan_inf(scores)
    _report("CRM FiLM forward → scalar scores", ok, f"shape={scores.shape}")


def test_conditional_reward_model_concat():
    """ConditionalRewardModel (concat): forward produces scalar per sample."""
    cfg = TransformerConfig(vocab_size=500, d_model=64, n_layer=2, n_head=4, max_seq_len=64)
    model = ConditionalRewardModel(cfg, num_conditions=3, cond_dim=32, conditioning="concat")
    model.eval()

    B, T = 4, 16
    ids = torch.randint(0, cfg.vocab_size, (B, T))
    mask = torch.ones(B, T, dtype=torch.long)
    cond = torch.randint(0, 3, (B,))

    with torch.no_grad():
        scores = model(ids, mask, cond)

    ok = scores.shape == (B,) and _no_nan_inf(scores)
    _report("CRM concat forward → scalar scores", ok, f"shape={scores.shape}")


def test_score_all_conditions():
    """score_all_conditions: one backbone pass → (B, num_conditions) scores."""
    cfg = TransformerConfig(vocab_size=500, d_model=64, n_layer=2, n_head=4, max_seq_len=64)
    num_cond = 3
    model = ConditionalRewardModel(cfg, num_conditions=num_cond, conditioning="film")
    model.eval()

    B, T = 2, 16
    ids = torch.randint(0, cfg.vocab_size, (B, T))
    mask = torch.ones(B, T, dtype=torch.long)

    with torch.no_grad():
        all_scores = model.score_all_conditions(ids, mask)

    ok = all_scores.shape == (B, num_cond) and _no_nan_inf(all_scores)
    _report("score_all_conditions shape", ok, f"shape={all_scores.shape}")


def test_preference_loss():
    """Bradley-Terry preference_loss: gradients flow, loss is non-negative."""
    r_chosen = torch.tensor([1.0, 0.5], requires_grad=True)
    r_rejected = torch.tensor([0.0, -0.5], requires_grad=True)

    loss, bt_loss = preference_loss(r_chosen, r_rejected, l2_coef=1e-3)
    loss.backward()

    ok_positive = loss.item() > 0
    ok_grad = r_chosen.grad is not None and r_chosen.grad.abs().sum() > 0
    # When chosen > rejected, BT loss should be low
    ok_correct_dir = bt_loss.item() < 1.0

    _report("preference_loss non-negative", ok_positive, f"loss={loss.item():.4f}")
    _report("preference_loss gradients flow", ok_grad)
    _report("preference_loss correct direction", ok_correct_dir, f"bt={bt_loss.item():.4f}")


def test_preference_dataset():
    """Synthetic preference dataset: pairs are generated with conflicts."""
    pairs = build_dataset(seed=0)
    train, val = split(pairs, val_frac=0.2, seed=0)

    ok_total = len(pairs) > 0
    ok_split = len(train) > 0 and len(val) > 0 and len(train) + len(val) == len(pairs)

    n_conflict = sum(p.conflicting for p in pairs)
    ok_conflicts = n_conflict > 0

    ceiling = condition_blind_ceiling(val)
    ok_ceiling = 0.0 < ceiling < 1.0  # Must be strictly between 0 and 1

    _report("Dataset generation", ok_total, f"{len(pairs)} pairs")
    _report("Dataset train/val split", ok_split, f"train={len(train)}, val={len(val)}")
    _report("Dataset has conflicting pairs", ok_conflicts, f"n_conflict={n_conflict}")
    _report("Condition-blind ceiling", ok_ceiling, f"ceiling={ceiling:.3f}")


def test_crm_condition_sensitivity():
    """Condition input affects output: different condition → different score."""
    torch.manual_seed(42)
    cfg = TransformerConfig(vocab_size=500, d_model=64, n_layer=2, n_head=4, max_seq_len=64)
    model = ConditionalRewardModel(cfg, num_conditions=3, conditioning="film")

    # Do a few gradient steps so the model moves away from identity FiLM init
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    ids = torch.randint(0, 500, (4, 16))
    mask = torch.ones(4, 16, dtype=torch.long)

    for _ in range(20):
        r_c = model(ids[:2], mask[:2], torch.tensor([0, 1]))
        r_r = model(ids[2:], mask[2:], torch.tensor([0, 1]))
        loss, _ = preference_loss(r_c, r_r)
        opt.zero_grad()
        loss.backward()
        opt.step()

    # After training, scoring the same input under different conditions
    # should produce different scores
    model.eval()
    with torch.no_grad():
        all_scores = model.score_all_conditions(ids[:1], mask[:1])  # (1, 3)

    scores = all_scores[0].tolist()
    # At least two conditions should give different scores
    ok = not all(abs(s - scores[0]) < 1e-6 for s in scores[1:])
    _report("CRM condition sensitivity", ok, f"scores={[f'{s:.3f}' for s in scores]}")


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print("=" * 60)
    print("  GQA Transformer + CRM — Smoke Test Suite")
    print("=" * 60)

    # Part 1: GQA Transformer
    print("\n-- GQA Transformer Pipeline --")
    test_model_config()
    test_rms_norm_and_swiglu()
    test_rotary_embedding()
    test_grouped_query_attention()
    test_kv_cache()
    test_transformer_forward()
    test_transformer_generate()
    test_tokenizer_roundtrip()
    test_pipeline_no_checkpoint()
    test_causal_masking()

    # Part 2: Conditional Reward Model
    print("\n-- Conditional Reward Model --")
    test_reward_transformer_backbone()
    test_film_conditioner()
    test_conditional_reward_model_film()
    test_conditional_reward_model_concat()
    test_score_all_conditions()
    test_preference_loss()
    test_preference_dataset()
    test_crm_condition_sensitivity()

    elapsed = time.time() - t0
    print("\n" + "=" * 60)
    total = _PASS_COUNT + _FAIL_COUNT
    print(f"  {_PASS_COUNT}/{total} passed, {_FAIL_COUNT} failed  ({elapsed:.2f}s)")
    print("=" * 60)

    sys.exit(0 if _FAIL_COUNT == 0 else 1)


if __name__ == "__main__":
    main()
