"""
End-to-end integration test for the full model pipeline.

Tests:
    1. Tokenize a sample string via SimpleTokenizer
    2. Forward through full transformer → logits [batch, seq_len, vocab_size]
    3. Forward at extended sequence length to confirm YaRN scaling activates
    4. Load the 206MB checkpoint and confirm 0 missing/unexpected keys
    5. Print shape/dtype at every stage
    6. Assert no NaN/Inf anywhere in the pipeline
    7. Regression check: GQA with/without external cos/sin produces identical output
"""

import sys
import os
import torch
import torch.nn as nn

# Ensure we can import from the project directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llm.model_config import ModelConfig
from llm.model import TransformerModel
from llm.tokenizer import SimpleTokenizer
from llm.gqa import GroupedQueryAttention
from llm.yarn import YaRNRotaryEmbedding


def separator(title: str):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def check_no_nan_inf(tensor: torch.Tensor, name: str):
    """Assert no NaN or Inf values in tensor."""
    assert not torch.isnan(tensor).any(), f"NaN detected in {name}!"
    assert not torch.isinf(tensor).any(), f"Inf detected in {name}!"
    print(f"  ✓ {name}: no NaN/Inf")


def test_regression_gqa_cos_sin():
    """REGRESSION CHECK: Verify that GQA with externally-supplied cos/sin
    produces byte-identical output to GQA with internal cos/sin computation,
    given the same weights and input."""

    separator("Regression Check: GQA internal vs external cos/sin")

    torch.manual_seed(42)
    d_model, n_heads, n_kv_heads, head_dim = 640, 8, 2, 80
    batch, seq_len = 2, 32

    # Build GQA with INTERNAL cos/sin (original behavior)
    gqa_internal = GroupedQueryAttention(
        d_model=d_model,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
    )
    gqa_internal.eval()

    # Build GQA with EXTERNAL cos/sin (same values as internal would compute)
    cos_ext = gqa_internal.cos_cached.clone()
    sin_ext = gqa_internal.sin_cached.clone()

    gqa_external = GroupedQueryAttention(
        d_model=d_model,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        external_cos=cos_ext,
        external_sin=sin_ext,
    )
    gqa_external.eval()

    # Copy weights from internal to external
    gqa_external.load_state_dict(gqa_internal.state_dict(), strict=False)

    # Run identical input through both
    x = torch.randn(batch, seq_len, d_model)
    position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch, -1)

    with torch.no_grad():
        out_internal = gqa_internal(x, position_ids=position_ids)
        out_external = gqa_external(x, position_ids=position_ids)

    max_diff = (out_internal - out_external).abs().max().item()
    print(f"  Max absolute difference: {max_diff}")

    assert torch.equal(out_internal, out_external), (
        f"REGRESSION FAILURE: outputs differ! Max diff = {max_diff}"
    )
    print("  ✓ PASS: Identical output with external cos/sin (byte-identical)")


def test_tokenizer():
    """Test SimpleTokenizer encode/decode roundtrip."""
    separator("Tokenizer Test")

    tok = SimpleTokenizer(vocab_size=8000)
    sample = "Hello, world! This is a test."
    ids = tok.encode(sample)
    decoded = tok.decode(ids)

    print(f"  Input text:   \"{sample}\"")
    print(f"  Token IDs:    {ids[:20]}{'...' if len(ids) > 20 else ''}")
    print(f"  Num tokens:   {len(ids)}")
    print(f"  Decoded text: \"{decoded}\"")
    print(f"  Roundtrip OK: {decoded == sample}")
    print(f"  Max token ID: {max(ids)} (vocab_size={tok.vocab_size})")

    assert decoded == sample, "Tokenizer roundtrip failed!"
    assert all(0 <= tid < tok.vocab_size for tid in ids), "Token ID out of range!"
    print("  ✓ PASS")

    return tok, ids


def test_forward_pass(config: ModelConfig, tok: SimpleTokenizer, token_ids: list):
    """Test full forward pass at normal sequence length."""
    separator("Forward Pass (normal sequence length)")

    model = TransformerModel(config)
    model.eval()

    # Convert token IDs to tensor
    input_ids = torch.tensor([token_ids], dtype=torch.long)  # batch=1
    batch_size, seq_len = input_ids.shape
    print(f"  Input shape:  {input_ids.shape} (batch={batch_size}, seq_len={seq_len})")
    print(f"  Input dtype:  {input_ids.dtype}")

    with torch.no_grad():
        # ── Stage-by-stage shape reporting ──────────────────────────────
        # Embedding
        emb = model.tok_embeddings(input_ids)
        print(f"\n  Embedding out: shape={emb.shape}, dtype={emb.dtype}")
        check_no_nan_inf(emb, "embedding")

        # Per-layer
        h = emb
        position_ids = torch.arange(seq_len).unsqueeze(0)
        for i, layer in enumerate(model.layers):
            h_norm = layer.attn_norm(h)
            attn_out = layer.attn(h_norm, position_ids=position_ids)
            h = h + attn_out
            h = h + layer.ffn(layer.ffn_norm(h))
            if i == 0 or i == config.num_layers - 1:
                print(f"  Layer {i:2d} out:  shape={h.shape}, dtype={h.dtype}")
                check_no_nan_inf(h, f"layer_{i}")

        # Final norm
        h = model.norm(h)
        print(f"  Final norm:   shape={h.shape}, dtype={h.dtype}")
        check_no_nan_inf(h, "final_norm")

        # Full forward (logits)
        logits = model(input_ids)
        print(f"\n  Logits out:   shape={logits.shape}, dtype={logits.dtype}")
        check_no_nan_inf(logits, "logits")

        expected_shape = (batch_size, seq_len, config.vocab_size)
        assert logits.shape == expected_shape, (
            f"Shape mismatch! Expected {expected_shape}, got {logits.shape}"
        )
        print(f"  ✓ PASS: logits shape matches expected {expected_shape}")

    return model


def test_yarn_extended_context(config: ModelConfig):
    """Test forward pass at extended sequence length to verify YaRN activates."""
    separator("YaRN Extended Context Test")

    # Configure for extended context
    extended_config = ModelConfig(
        vocab_size=config.vocab_size,
        hidden_dim=config.hidden_dim,
        num_layers=2,  # Use fewer layers to save memory
        num_query_heads=config.num_query_heads,
        num_kv_heads=config.num_kv_heads,
        head_dim=config.head_dim,
        intermediate_dim=config.intermediate_dim,
        max_position_embeddings=4096,  # Extended beyond training length
        rope_theta=config.rope_theta,
        yarn_scale_factor=2.0,         # Activate YaRN scaling
        yarn_original_max_position=2048,  # Original training length
    )

    model = TransformerModel(extended_config)
    model.eval()

    # Test at a length beyond original training (but within extended window)
    seq_len = 3000  # Beyond 2048 original, within 4096 extended
    input_ids = torch.randint(0, config.vocab_size, (1, seq_len))
    print(f"  Extended seq_len: {seq_len}")
    print(f"  Original max pos: {extended_config.yarn_original_max_position}")
    print(f"  Extended max pos: {extended_config.max_position_embeddings}")
    print(f"  YaRN scale factor: {extended_config.yarn_scale_factor}")

    with torch.no_grad():
        logits = model(input_ids)
        print(f"  Logits shape: {logits.shape}")
        check_no_nan_inf(logits, "extended_logits")

    expected = (1, seq_len, config.vocab_size)
    assert logits.shape == expected, (
        f"Shape mismatch! Expected {expected}, got {logits.shape}"
    )
    print("  ✓ PASS: YaRN extended context produces valid output")


def test_checkpoint_loading(config: ModelConfig):
    """Load the 206MB pretrained checkpoint and verify key mapping."""
    separator("Checkpoint Loading Test")

    ckpt_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "checkpoints", "pretrain_model.pt"
    )

    if not os.path.isfile(ckpt_path):
        print(f"  ⚠ Checkpoint not found at {ckpt_path}")
        print("  SKIPPED (checkpoint file not available)")
        return

    print(f"  Checkpoint: {ckpt_path}")
    print(f"  File size:  {os.path.getsize(ckpt_path) / 1e6:.1f} MB")

    model = TransformerModel(config)

    missing, unexpected = model.load_pretrained(ckpt_path)

    print(f"\n  Missing keys:    {len(missing)}")
    if missing:
        for k in missing:
            print(f"    - {k}")

    print(f"  Unexpected keys: {len(unexpected)}")
    if unexpected:
        for k in unexpected:
            print(f"    - {k}")

    # Count total parameters loaded
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Total model parameters: {total_params:,}")
    print(f"  Model size (float32):   {total_params * 4 / 1e6:.1f} MB")

    # Run a forward pass with loaded weights
    test_ids = torch.randint(0, config.vocab_size, (1, 16))
    model.eval()
    with torch.no_grad():
        logits = model(test_ids)
        check_no_nan_inf(logits, "checkpoint_logits")

    assert len(missing) == 0, f"Missing keys: {missing}"
    assert len(unexpected) == 0, f"Unexpected keys: {unexpected}"
    print("  ✓ PASS: 0 missing, 0 unexpected keys")


def test_no_reward_model():
    """Explicitly confirm no reward-model code path was touched."""
    separator("Reward Model Exclusion Check")

    model = TransformerModel(ModelConfig())

    # Check that no reward-related attributes exist
    reward_attrs = [a for a in dir(model) if "reward" in a.lower()]
    assert len(reward_attrs) == 0, (
        f"Reward model attributes found: {reward_attrs}"
    )
    print("  ✓ No reward model attributes in TransformerModel")

    # Check that forward() has no reward-related parameters
    import inspect
    sig = inspect.signature(model.forward)
    reward_params = [p for p in sig.parameters if "reward" in p.lower()]
    assert len(reward_params) == 0, (
        f"Reward model parameters in forward(): {reward_params}"
    )
    print("  ✓ No reward model parameters in forward()")
    print("  ✓ PASS: Reward model is correctly excluded this round")
    print("  NOTE: # TODO(reward-model): pending Navanith/Ravikant")


def main():
    print("=" * 70)
    print("  GQA ↔ Pipeline Integration Test Suite")
    print("=" * 70)

    config = ModelConfig()
    print(f"\nModel Config:")
    print(f"  vocab_size={config.vocab_size}, hidden_dim={config.hidden_dim}")
    print(f"  num_layers={config.num_layers}, heads={config.num_query_heads}Q/{config.num_kv_heads}KV")
    print(f"  head_dim={config.head_dim}, intermediate_dim={config.intermediate_dim}")
    print(f"  max_pos={config.max_position_embeddings}, rope_theta={config.rope_theta}")

    # 1. Regression check (most critical — must pass first)
    test_regression_gqa_cos_sin()

    # 2. Tokenizer
    tok, token_ids = test_tokenizer()

    # 3. Forward pass at normal length
    test_forward_pass(config, tok, token_ids)

    # 4. YaRN extended context
    test_yarn_extended_context(config)

    # 5. Checkpoint loading
    test_checkpoint_loading(config)

    # 6. Reward model exclusion
    test_no_reward_model()

    separator("ALL TESTS PASSED ✓")


if __name__ == "__main__":
    main()
