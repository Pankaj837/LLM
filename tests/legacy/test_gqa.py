import unittest
import torch
from llm.gqa import GroupedQueryAttention, KVCache


class TestGroupedQueryAttention(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.d_model = 640
        self.n_heads = 8
        self.n_kv_heads = 2
        self.head_dim = 80
        self.max_seq_len = 2048
        self.rope_theta = 10000.0
        self.attention_dropout = 0.0

    def test_shape(self):
        """Shape test: forward pass with batch=2, seq_len=128 returns (2, 128, 640)."""
        model = GroupedQueryAttention(
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_kv_heads=self.n_kv_heads,
            head_dim=self.head_dim,
            max_seq_len=self.max_seq_len,
            rope_theta=self.rope_theta,
            attention_dropout=self.attention_dropout,
        )
        batch = 2
        seq_len = 128
        x = torch.randn(batch, seq_len, self.d_model)
        position_ids = torch.arange(seq_len).unsqueeze(0).repeat(batch, 1)

        out = model(x, position_ids=position_ids)
        self.assertEqual(out.shape, (batch, seq_len, self.d_model))

    def test_param_count(self):
        """
        Param count check: confirm k_proj/v_proj weight count equals
        d_model * (n_kv_heads * head_dim) — i.e. 4x smaller than standard MHA projection.
        """
        model = GroupedQueryAttention(
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_kv_heads=self.n_kv_heads,
            head_dim=self.head_dim,
            max_seq_len=self.max_seq_len,
        )
        expected_kv_params = self.d_model * (self.n_kv_heads * self.head_dim)
        expected_q_params = self.d_model * (self.n_heads * self.head_dim)

        self.assertEqual(model.k_proj.weight.numel(), expected_kv_params)
        self.assertEqual(model.v_proj.weight.numel(), expected_kv_params)
        self.assertEqual(model.q_proj.weight.numel(), expected_q_params)
        self.assertEqual(model.o_proj.weight.numel(), expected_q_params)

        # Confirm KV projections are 4x smaller than Q projection
        ratio = expected_q_params / expected_kv_params
        self.assertEqual(ratio, 4.0)

    def test_kv_cache_memory(self):
        """
        KV-cache memory test: instantiate KV cache for sequence length 2048
        and assert its memory footprint is exactly n_kv_heads / n_heads = 0.25x
        (4x reduction) compared to a standard multi-head attention cache.
        """
        batch_size = 2
        seq_len = 2048

        # GQA Cache (n_kv_heads = 2)
        gqa_cache = KVCache(
            max_batch_size=batch_size,
            max_seq_len=seq_len,
            n_kv_heads=self.n_kv_heads,
            head_dim=self.head_dim,
            dtype=torch.float32,
        )

        # MHA Cache (n_kv_heads = n_heads = 8)
        mha_cache = KVCache(
            max_batch_size=batch_size,
            max_seq_len=seq_len,
            n_kv_heads=self.n_heads,
            head_dim=self.head_dim,
            dtype=torch.float32,
        )

        gqa_memory = gqa_cache.memory_footprint_bytes
        mha_memory = mha_cache.memory_footprint_bytes

        ratio = gqa_memory / mha_memory
        self.assertAlmostEqual(ratio, self.n_kv_heads / self.n_heads, places=6)
        self.assertEqual(ratio, 0.25)

    def test_equivalence_to_mha(self):
        """
        Equivalence-to-MHA sanity check: with n_kv_heads == n_heads, the module
        should reduce to standard multi-head attention (grouping size 1).
        """
        mha_model = GroupedQueryAttention(
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_kv_heads=self.n_heads,  # n_kv_heads == n_heads
            head_dim=self.head_dim,
            max_seq_len=self.max_seq_len,
        )
        self.assertEqual(mha_model.group_size, 1)
        self.assertEqual(
            mha_model.k_proj.weight.numel(), mha_model.q_proj.weight.numel()
        )

        x = torch.randn(2, 64, self.d_model)
        position_ids = torch.arange(64).unsqueeze(0).repeat(2, 1)

        out = mha_model(x, position_ids=position_ids)
        self.assertEqual(out.shape, (2, 64, self.d_model))

    def test_incremental_decoding(self):
        """
        Incremental decoding test: run module once over full 10-token sequence,
        then run it token-by-token with KV cache over same 10 tokens; outputs at
        each position must match (within floating point tolerance) between the two modes.
        """
        model = GroupedQueryAttention(
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_kv_heads=self.n_kv_heads,
            head_dim=self.head_dim,
            max_seq_len=self.max_seq_len,
        )
        model.eval()

        batch_size = 2
        seq_len = 10
        x = torch.randn(batch_size, seq_len, self.d_model)
        position_ids = torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1)

        # 1. Full sequence forward pass
        with torch.no_grad():
            full_out = model(x, position_ids=position_ids, use_causal_mask=True)

        # 2. Token-by-token pass with KV cache
        kv_cache = KVCache(
            max_batch_size=batch_size,
            max_seq_len=seq_len,
            n_kv_heads=self.n_kv_heads,
            head_dim=self.head_dim,
            dtype=x.dtype,
            device=x.device,
        )

        incremental_outs = []
        with torch.no_grad():
            for i in range(seq_len):
                x_i = x[:, i : i + 1, :]
                pos_i = position_ids[:, i : i + 1]
                out_i = model(
                    x_i, position_ids=pos_i, kv_cache=kv_cache, use_causal_mask=True
                )
                incremental_outs.append(out_i)

        stacked_out = torch.cat(incremental_outs, dim=1)

        # Verify outputs match within float32 tolerance
        max_diff = (full_out - stacked_out).abs().max().item()
        self.assertTrue(
            torch.allclose(full_out, stacked_out, atol=1e-5, rtol=1e-5),
            f"Outputs differ! Max difference: {max_diff}",
        )

    def test_causal_masking(self):
        """
        Causal masking test: confirm token at position i cannot attend to tokens
        at position > i (verify via controlled gradient flow).
        """
        model = GroupedQueryAttention(
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_kv_heads=self.n_kv_heads,
            head_dim=self.head_dim,
        )

        batch_size = 1
        seq_len = 5
        x = torch.randn(batch_size, seq_len, self.d_model, requires_grad=True)
        position_ids = torch.arange(seq_len).unsqueeze(0)

        out = model(x, position_ids=position_ids, use_causal_mask=True)

        # Compute loss on output at position 2
        # If causal masking works, loss at pos 2 must NOT produce gradients for inputs at pos 3 or pos 4
        loss = out[0, 2, :].sum()
        loss.backward()

        # Check gradients for position 3 and 4 are zero
        grad_pos_3 = x.grad[0, 3, :].abs().max().item()
        grad_pos_4 = x.grad[0, 4, :].abs().max().item()

        self.assertEqual(
            grad_pos_3, 0.0, f"Token 2 attended to Token 3! Grad: {grad_pos_3}"
        )
        self.assertEqual(
            grad_pos_4, 0.0, f"Token 2 attended to Token 4! Grad: {grad_pos_4}"
        )

        # Check gradients for positions 0, 1, 2 are non-zero
        grad_pos_0 = x.grad[0, 0, :].abs().max().item()
        grad_pos_1 = x.grad[0, 1, :].abs().max().item()
        grad_pos_2 = x.grad[0, 2, :].abs().max().item()

        self.assertGreater(grad_pos_0, 0.0)
        self.assertGreater(grad_pos_1, 0.0)
        self.assertGreater(grad_pos_2, 0.0)

    def test_load_checkpoint_weights(self):
        """Test loading pretrain_model checkpoint attention weights into GroupedQueryAttention module."""
        import os

        ckpt_path = "checkpoints/pretrain_model.pt"
        if not os.path.exists(ckpt_path):
            self.skipTest("Checkpoint file pretrain_model.pt not found")

        ckpt = torch.load(ckpt_path, map_location="cpu")
        model = GroupedQueryAttention(
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_kv_heads=self.n_kv_heads,
            head_dim=self.head_dim,
            max_seq_len=self.max_seq_len,
        )

        layer0_state_dict = {
            "q_proj.weight": ckpt["layers.0.attn.W_q.weight"],
            "k_proj.weight": ckpt["layers.0.attn.W_k.weight"],
            "v_proj.weight": ckpt["layers.0.attn.W_v.weight"],
            "o_proj.weight": ckpt["layers.0.attn.W_o.weight"],
        }

        missing_keys, unexpected_keys = model.load_state_dict(
            layer0_state_dict, strict=False
        )
        self.assertEqual(len(missing_keys), 0)
        self.assertEqual(len(unexpected_keys), 0)

        # Run forward pass with loaded pretrain weights
        batch = 2
        seq_len = 16
        x = torch.randn(batch, seq_len, self.d_model)
        position_ids = torch.arange(seq_len).unsqueeze(0).repeat(batch, 1)

        out = model(x, position_ids=position_ids)
        self.assertEqual(out.shape, (batch, seq_len, self.d_model))

    def test_chunked_decoding_causal_masking(self):
        """
        Tests multi-token chunk appended to an already-populated cache (speculative / chunked decoding).
        Confirms via gradient flow that within the multi-token chunk (e.g. 3 new tokens), later tokens
        cannot leak into earlier tokens' outputs.
        """
        model = GroupedQueryAttention(
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_kv_heads=self.n_kv_heads,
            head_dim=self.head_dim,
        )

        batch_size = 1
        initial_seq_len = 5
        chunk_seq_len = 3
        total_seq_len = initial_seq_len + chunk_seq_len

        # Populate cache with initial sequence (5 tokens)
        x_init = torch.randn(batch_size, initial_seq_len, self.d_model)
        pos_init = torch.arange(initial_seq_len).unsqueeze(0)
        kv_cache = KVCache(
            max_batch_size=batch_size,
            max_seq_len=total_seq_len,
            n_kv_heads=self.n_kv_heads,
            head_dim=self.head_dim,
        )

        with torch.no_grad():
            model(x_init, position_ids=pos_init, kv_cache=kv_cache, use_causal_mask=True)

        # Now pass a 3-token chunk to the populated cache (positions 5, 6, 7)
        x_chunk = torch.randn(batch_size, chunk_seq_len, self.d_model, requires_grad=True)
        pos_chunk = torch.arange(initial_seq_len, total_seq_len).unsqueeze(0)

        out_chunk = model(x_chunk, position_ids=pos_chunk, kv_cache=kv_cache, use_causal_mask=True)

        # Loss on first token of the chunk (index 0 in chunk, index 5 in sequence)
        loss = out_chunk[0, 0, :].sum()
        loss.backward()

        # Check gradients for future tokens in the chunk (index 1 and 2 in chunk) are strictly 0.0
        grad_chunk_token_1 = x_chunk.grad[0, 1, :].abs().max().item()
        grad_chunk_token_2 = x_chunk.grad[0, 2, :].abs().max().item()

        self.assertEqual(
            grad_chunk_token_1,
            0.0,
            f"Chunk token 0 attended to Chunk token 1! Grad: {grad_chunk_token_1}",
        )
        self.assertEqual(
            grad_chunk_token_2,
            0.0,
            f"Chunk token 0 attended to Chunk token 2! Grad: {grad_chunk_token_2}",
        )

        # Check gradient for chunk token 0 itself is non-zero
        grad_chunk_token_0 = x_chunk.grad[0, 0, :].abs().max().item()
        self.assertGreater(grad_chunk_token_0, 0.0)

    def test_chunked_decoding_versus_single_step(self):
        """
        Verifies that chunked decoding (e.g. 5 tokens + 3 tokens + 2 tokens) yields identical outputs
        to single-step decoding and full-sequence prefill.
        """
        model = GroupedQueryAttention(
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_kv_heads=self.n_kv_heads,
            head_dim=self.head_dim,
        )
        model.eval()

        batch_size = 2
        seq_len = 10
        x = torch.randn(batch_size, seq_len, self.d_model)
        position_ids = torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1)

        # 1. Full sequence forward pass
        with torch.no_grad():
            full_out = model(x, position_ids=position_ids, use_causal_mask=True)

        # 2. Chunked pass: 5 tokens then 3 tokens then 2 tokens
        kv_cache = KVCache(
            max_batch_size=batch_size,
            max_seq_len=seq_len,
            n_kv_heads=self.n_kv_heads,
            head_dim=self.head_dim,
        )

        chunks = [(0, 5), (5, 8), (8, 10)]
        chunk_outputs = []
        with torch.no_grad():
            for start, end in chunks:
                x_chunk = x[:, start:end, :]
                pos_chunk = position_ids[:, start:end]
                out_chunk = model(
                    x_chunk, position_ids=pos_chunk, kv_cache=kv_cache, use_causal_mask=True
                )
                chunk_outputs.append(out_chunk)

        stacked_chunk_out = torch.cat(chunk_outputs, dim=1)

        max_diff = (full_out - stacked_chunk_out).abs().max().item()
        self.assertTrue(
            torch.allclose(full_out, stacked_chunk_out, atol=1e-5, rtol=1e-5),
            f"Chunked decoding output differs from full sequence! Max diff: {max_diff}",
        )

    def test_padding_mask(self):
        """
        Tests explicit attention_mask (padding mask) to confirm padded positions
        are correctly masked out and produce no gradient flow.
        """
        model = GroupedQueryAttention(
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_kv_heads=self.n_kv_heads,
            head_dim=self.head_dim,
        )

        batch_size = 1
        seq_len = 4
        x = torch.randn(batch_size, seq_len, self.d_model, requires_grad=True)
        position_ids = torch.arange(seq_len).unsqueeze(0)

        # Boolean mask: True = keep, False = mask out. Mask out token index 1 (padded)
        attn_mask = torch.tensor([[[[True, False, True, True]]]])  # (1, 1, 1, 4) broadcastable

        out = model(x, position_ids=position_ids, attention_mask=attn_mask, use_causal_mask=False)

        # Loss on last token (position 3)
        loss = out[0, 3, :].sum()
        loss.backward()

        # Token 1 is masked, so gradient for token 1 should be 0.0
        grad_token_1 = x.grad[0, 1, :].abs().max().item()
        self.assertEqual(grad_token_1, 0.0, f"Padded token 1 was attended to! Grad: {grad_token_1}")

        # Non-padded tokens 0, 2, 3 should receive gradients
        self.assertGreater(x.grad[0, 0, :].abs().max().item(), 0.0)
        self.assertGreater(x.grad[0, 2, :].abs().max().item(), 0.0)
        self.assertGreater(x.grad[0, 3, :].abs().max().item(), 0.0)




if __name__ == "__main__":
    unittest.main()
