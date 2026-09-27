"""
TransformerModel — the full pipeline wiring.

    tok_embeddings → [TransformerBlock × num_layers] → final RMSNorm → output

Each TransformerBlock:
    attn_norm (RMSNorm) → GroupedQueryAttention (with YaRN cos/sin & temperature) → residual
    ffn_norm  (RMSNorm) → SwiGLUFeedForward                                       → residual

Output head is weight-tied to tok_embeddings (no separate output.weight in
the checkpoint).

Checkpoint loading maps:
    checkpoint key                    → module parameter
    ─────────────────────────────────   ───────────────────────────────
    tok_embeddings.weight             → tok_embeddings.weight
    layers.{i}.attn_norm.weight       → layers.{i}.attn_norm.weight
    layers.{i}.attn.W_q.weight        → layers.{i}.attn.q_proj.weight
    layers.{i}.attn.W_k.weight        → layers.{i}.attn.k_proj.weight
    layers.{i}.attn.W_v.weight        → layers.{i}.attn.v_proj.weight
    layers.{i}.attn.W_o.weight        → layers.{i}.attn.o_proj.weight
    layers.{i}.ffn_norm.weight        → layers.{i}.ffn_norm.weight
    layers.{i}.ffn.W_gate.weight      → layers.{i}.ffn.W_gate.weight
    layers.{i}.ffn.W_up.weight        → layers.{i}.ffn.W_up.weight
    layers.{i}.ffn.W_down.weight      → layers.{i}.ffn.W_down.weight
    norm.weight                       → norm.weight
"""

from __future__ import annotations

import re
import warnings
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model_config import ModelConfig
from .modules import RMSNorm, SwiGLUFeedForward
from .gqa import GroupedQueryAttention, KVCache
from .rope import RotaryEmbedding


class TransformerBlock(nn.Module):
    """Single transformer layer: attention + feed-forward with pre-norm."""

    def __init__(
        self,
        config: ModelConfig,
        external_cos: Optional[torch.Tensor] = None,
        external_sin: Optional[torch.Tensor] = None,
        rope: Optional[nn.Module] = None,
        attn_temperature: float = 1.0,
    ):
        super().__init__()
        self.attn_norm = RMSNorm(config.hidden_dim, eps=config.rms_norm_eps)
        self.attn = GroupedQueryAttention(
            d_model=config.hidden_dim,
            n_heads=config.num_query_heads,
            n_kv_heads=config.num_kv_heads,
            head_dim=config.head_dim,
            max_seq_len=config.max_position_embeddings,
            rope_theta=config.rope_theta,
            attention_dropout=config.attention_dropout,
            external_cos=external_cos,
            external_sin=external_sin,
            attn_temperature=attn_temperature,
            rope=rope,
        )
        self.ffn_norm = RMSNorm(config.hidden_dim, eps=config.rms_norm_eps)
        self.ffn = SwiGLUFeedForward(config.hidden_dim, config.intermediate_dim)

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
        attention_mask: Optional[torch.Tensor] = None,
        use_causal_mask: bool = True,
        rope_cos_sin: Optional[tuple] = None,
    ) -> torch.Tensor:
        # Pre-norm attention with residual
        h = x + self.attn(
            self.attn_norm(x),
            position_ids=position_ids,
            kv_cache=kv_cache,
            attention_mask=attention_mask,
            use_causal_mask=use_causal_mask,
            rope_cos_sin=rope_cos_sin,
        )
        # Pre-norm feed-forward with residual
        out = h + self.ffn(self.ffn_norm(h))
        return out


class TransformerModel(nn.Module):
    """Full transformer model wiring all components end-to-end.

    Pipeline:
        token_ids → tok_embeddings → [TransformerBlock × N] → norm → output logits

    The output projection is weight-tied to tok_embeddings.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # ── Embedding ───────────────────────────────────────────────────
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.hidden_dim)

        # ── RoPE / YaRN Rotary Embedding (shared across all layers) ─────
        self.rope = RotaryEmbedding(config)

        # ── Transformer layers ──────────────────────────────────────────
        self.layers = nn.ModuleList([
            TransformerBlock(
                config,
                external_cos=self.rope.cos_cached,
                external_sin=self.rope.sin_cached,
                rope=self.rope,
                attn_temperature=self.rope.attention_temperature(),
            )
            for _ in range(config.num_layers)
        ])

        # ── Final norm ──────────────────────────────────────────────────
        self.norm = RMSNorm(config.hidden_dim, eps=config.rms_norm_eps)

        # ── Output head (weight-tied to embedding) ──────────────────────
        # No separate nn.Linear — we manually matmul with tok_embeddings.weight

        # GPT-style init (std 0.02). Without this, nn.Embedding defaults to
        # N(0, 1); because the output head is tied to the embedding, logits
        # start with std ~26 and the initial loss is ~570 instead of ln(V) ~ 9.
        # Matches the statistics observed in pretrain_model.pt.
        self.apply(self._init_weights)

        # The conditional reward model is deliberately NOT part of this forward pass: it consumes
        # `forward(..., return_hidden=True)` through `llm.reward.LMBackboneAdapter`.

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        token_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        kv_caches: Optional[List[KVCache]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        use_causal_mask: bool = True,
        return_hidden: bool = False,
        rope_context_len: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Args:
            token_ids:  (batch, seq_len)  integer token IDs
            position_ids: (batch, seq_len) optional; auto-generated if None
            kv_caches: optional list of KVCache, one per layer
            attention_mask: optional. Either a (batch, total_len) padding mask
                (1/True = real token, 0/False = pad; RIGHT padding recommended)
                or a broadcastable 4-D bool/float mask (batch, 1, q_len, k_len).
            use_causal_mask: whether to apply causal masking
            return_hidden: return the final-norm hidden states
                (batch, seq_len, hidden_dim) instead of logits, e.g. to feed a
                reward/value head.
            rope_context_len: planned total context (prompt + new tokens) used to pick the
                RoPE frequencies. Only matters for Dynamic NTK (see rope.py). Defaults to the
                KV-cache capacity when caches are given, else to the input length.

        Returns:
            logits: (batch, seq_len, vocab_size), or hidden states if return_hidden
        """
        batch_size, seq_len = token_ids.shape

        if attention_mask is not None and attention_mask.dim() == 2:
            attention_mask = attention_mask.to(torch.bool)[:, None, None, :]

        # Auto-generate position_ids if not provided
        explicit_positions = position_ids is not None
        if position_ids is None:
            if kv_caches is not None and kv_caches[0].seen_tokens > 0:
                start_pos = kv_caches[0].seen_tokens
                position_ids = torch.arange(
                    start_pos, start_pos + seq_len, device=token_ids.device
                ).unsqueeze(0).expand(batch_size, -1)
            else:
                position_ids = torch.arange(
                    seq_len, device=token_ids.device
                ).unsqueeze(0).expand(batch_size, -1)

        # ── RoPE tables: computed once, shared by every layer ───────────
        past = kv_caches[0].seen_tokens if kv_caches is not None else 0
        context_len = rope_context_len
        if context_len is None:
            context_len = kv_caches[0].max_seq_len if kv_caches is not None else seq_len
        context_len = max(context_len, past + seq_len)
        if explicit_positions:                 # user-supplied ids: one host sync to stay safe
            context_len = max(context_len, int(position_ids.max().item()) + 1)
        h = self.tok_embeddings(token_ids)     # (batch, seq_len, hidden_dim)
        rope_cos_sin = self.rope.get_cos_sin(position_ids, h.dtype, context_len)

        # ── Transformer layers ──────────────────────────────────────────
        for i, layer in enumerate(self.layers):
            cache = kv_caches[i] if kv_caches is not None else None
            h = layer(
                h,
                position_ids=position_ids,
                kv_cache=cache,
                attention_mask=attention_mask,
                use_causal_mask=use_causal_mask,
                rope_cos_sin=rope_cos_sin,
            )

        # ── Final norm ──────────────────────────────────────────────────
        h = self.norm(h)  # (batch, seq_len, hidden_dim)

        if return_hidden:
            return h

        # ── Output logits (weight-tied) ─────────────────────────────────
        logits = torch.mm(
            h.view(-1, self.config.hidden_dim),
            self.tok_embeddings.weight.t(),
        ).view(batch_size, seq_len, self.config.vocab_size)

        return logits

    # ── Autoregressive Generation with KV Cache ─────────────────────────

    @torch.no_grad()
    def generate(
        self,
        token_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        eos_token_id: Optional[int] = None,
        use_cache: bool = True,
    ) -> torch.Tensor:
        """Autoregressively generates continuation tokens from prompt token IDs.

        Args:
            token_ids: (batch_size, seq_len) tensor of input token IDs
            max_new_tokens: maximum number of tokens to generate
            temperature: sampling temperature (0.0 or <= 1e-6 for greedy decoding)
            top_k: if set > 0, filter logits to top k highest values
            top_p: if set in (0, 1), nucleus sampling probability mass
            eos_token_id: stop generation if all sequences produce this ID
            use_cache: use KVCache for efficient step-by-step decoding

        Returns:
            (batch_size, seq_len + num_generated) full token sequence
        """
        self.eval()
        if token_ids.dim() != 2 or token_ids.size(1) == 0:
            raise ValueError(f"token_ids must be (batch, seq_len >= 1), got {tuple(token_ids.shape)}")
        if temperature < 0:
            raise ValueError(f"temperature must be >= 0, got {temperature}")
        if max_new_tokens <= 0:
            return token_ids.clone()

        batch_size, prompt_len = token_ids.shape
        device = token_ids.device

        generated = token_ids.clone()
        max_total_len = prompt_len + max_new_tokens
        if max_total_len > self.config.max_position_embeddings:
            warnings.warn(
                f"prompt ({prompt_len}) + max_new_tokens ({max_new_tokens}) = {max_total_len} exceeds "
                f"max_position_embeddings ({self.config.max_position_embeddings}); "
                "quality degrades beyond the trained context unless YaRN/NTK scaling is configured.",
                stacklevel=2,
            )

        # Helper to sample next token from logits
        def sample_next_token(logits_step: torch.Tensor) -> torch.Tensor:
            # logits_step: (batch_size, vocab_size)
            if temperature <= 1e-6 or (top_k is not None and top_k == 1):
                return torch.argmax(logits_step, dim=-1, keepdim=True)

            scaled_logits = logits_step / max(temperature, 1e-6)

            if top_k is not None and top_k > 0:
                val, _ = torch.topk(scaled_logits, min(top_k, scaled_logits.size(-1)))
                kth = val[:, -1].unsqueeze(-1)
                scaled_logits = torch.where(
                    scaled_logits < kth,
                    torch.tensor(float("-inf"), device=device),
                    scaled_logits,
                )

            if top_p is not None and 0.0 < top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift mask right so the first token above threshold is kept
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = False
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                scaled_logits = scaled_logits.masked_fill(indices_to_remove, float("-inf"))

            probs = F.softmax(scaled_logits, dim=-1)
            return torch.multinomial(probs, num_samples=1)

        # Per-row EOS bookkeeping: once a row has emitted EOS it keeps emitting
        # EOS (padding) instead of continuing to generate text.
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        def emit(next_token: torch.Tensor) -> torch.Tensor:
            nonlocal generated, finished
            if eos_token_id is not None:
                next_token = torch.where(
                    finished.unsqueeze(-1), torch.full_like(next_token, eos_token_id), next_token
                )
                finished = finished | (next_token.squeeze(-1) == eos_token_id)
            generated = torch.cat([generated, next_token], dim=-1)
            return next_token

        if use_cache:
            # 1. Initialize KV cache per layer
            kv_caches = [
                KVCache(
                    max_batch_size=batch_size,
                    max_seq_len=max_total_len,
                    n_kv_heads=self.config.num_kv_heads,
                    head_dim=self.config.head_dim,
                    dtype=self.tok_embeddings.weight.dtype,
                    device=device,
                )
                for _ in range(self.config.num_layers)
            ]

            # 2. Prefill phase (process prompt tokens)
            logits = self.forward(
                token_ids,
                kv_caches=kv_caches,
                use_causal_mask=True,
                rope_context_len=max_total_len,
            )
            next_token = emit(sample_next_token(logits[:, -1, :]))

            # 3. Incremental decoding loop
            for _ in range(max_new_tokens - 1):
                if eos_token_id is not None and finished.all():
                    break
                logits = self.forward(
                    next_token,
                    kv_caches=kv_caches,        # positions follow the cache (no host-side tensors)
                    use_causal_mask=False,
                    rope_context_len=max_total_len,
                )
                next_token = emit(sample_next_token(logits[:, -1, :]))
        else:
            # Non-cached path (recomputes full sequence every step)
            for _ in range(max_new_tokens):
                if eos_token_id is not None and finished.all():
                    break
                logits = self.forward(generated, use_causal_mask=True, rope_context_len=max_total_len)
                emit(sample_next_token(logits[:, -1, :]))

        return generated

    # ── Checkpoint loading ──────────────────────────────────────────────

    @staticmethod
    def _remap_checkpoint_key(ckpt_key: str) -> str:
        """Legacy key names (``attn.W_q``) -> this module's names (``attn.q_proj``)."""
        from .checkpoint import remap_legacy_key
        return remap_legacy_key(ckpt_key)

    def load_pretrained(
        self,
        checkpoint_path: str,
        strict: bool = True,
        expected_vocab_sha256: Optional[str] = None,
    ) -> Tuple[List[str], List[str]]:
        """Loads a checkpoint (self-describing format or a legacy raw state_dict).

        Returns (missing_keys, unexpected_keys); with ``strict=True`` (default) any mismatch raises.
        Use ``llm.checkpoint.load_checkpoint`` to also get the metadata.
        """
        from .checkpoint import load_checkpoint
        info = load_checkpoint(checkpoint_path, self, strict=strict, expected_vocab_sha256=expected_vocab_sha256)
        return info["missing_keys"], info["unexpected_keys"]
