import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class KVCache:
    """
    Key-Value Cache for autoregressive incremental decoding.
    Stores Key and Value tensors at n_kv_heads size: (batch, n_kv_heads, max_seq_len, head_dim).
    This provides an exact (n_kv_heads / n_heads) reduction in memory footprint compared to standard MHA cache.
    """

    def __init__(
        self,
        max_batch_size: int,
        max_seq_len: int,
        n_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
    ):
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device

        self.k_cache: torch.Tensor = torch.zeros(
            (max_batch_size, n_kv_heads, max_seq_len, head_dim),
            dtype=dtype,
            device=device,
        )
        self.v_cache: torch.Tensor = torch.zeros(
            (max_batch_size, n_kv_heads, max_seq_len, head_dim),
            dtype=dtype,
            device=device,
        )
        self.seen_tokens: int = 0

    @property
    def memory_footprint_bytes(self) -> int:
        """Total memory allocated for key and value caches in bytes."""
        return (
            self.k_cache.element_size() * self.k_cache.numel()
            + self.v_cache.element_size() * self.v_cache.numel()
        )

    def update(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        start_pos: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Appends new K and V tensors in-place.
        k, v shape: (batch, n_kv_heads, seq_len, head_dim)
        Returns full cached slice up to current position: (batch, n_kv_heads, total_seq_len, head_dim)
        """
        batch_size, n_kv_heads, seq_len, head_dim = k.shape
        assert (
            n_kv_heads == self.n_kv_heads
        ), f"Expected n_kv_heads={self.n_kv_heads}, got {n_kv_heads}"
        assert (
            head_dim == self.head_dim
        ), f"Expected head_dim={self.head_dim}, got {head_dim}"
        assert (
            batch_size <= self.max_batch_size
        ), f"Batch size {batch_size} exceeds max_batch_size {self.max_batch_size}"

        if start_pos is None:
            start_pos = self.seen_tokens

        end_pos = start_pos + seq_len
        assert (
            end_pos <= self.max_seq_len
        ), f"Sequence length {end_pos} exceeds max_seq_len {self.max_seq_len}"

        # Copy in-place into preallocated cache tensor
        self.k_cache[:batch_size, :, start_pos:end_pos, :] = k
        self.v_cache[:batch_size, :, start_pos:end_pos, :] = v
        self.seen_tokens = max(self.seen_tokens, end_pos)

        return (
            self.k_cache[:batch_size, :, :end_pos, :],
            self.v_cache[:batch_size, :, :end_pos, :],
        )

    def reset(self):
        """Resets the cache seen_tokens count and zeroes out buffers."""
        self.seen_tokens = 0
        self.k_cache.zero_()
        self.v_cache.zero_()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of x."""
    half_dim = x.shape[-1] // 2
    x1 = x[..., :half_dim]
    x2 = x[..., half_dim:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """
    Applies Rotary Position Embedding to x.
    x: (batch, n_heads, seq_len, head_dim)
    cos, sin: (batch, 1, seq_len, head_dim)
    """
    return (x * cos) + (rotate_half(x) * sin)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Expands KV heads from n_kv_heads to n_heads by repeating each KV head contiguously n_rep times.
    x: (batch, n_kv_heads, seq_len, head_dim)
    Returns: (batch, n_heads, seq_len, head_dim)
    """
    if n_rep == 1:
        return x
    return x.repeat_interleave(n_rep, dim=1)


class GroupedQueryAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int,
        max_seq_len: int = 2048,
        rope_theta: float = 10000.0,
        attention_dropout: float = 0.0,
        external_cos: Optional[torch.Tensor] = None,
        external_sin: Optional[torch.Tensor] = None,
        attn_temperature: float = 1.0,
        rope: Optional[nn.Module] = None,
    ):
        super().__init__()
        assert (
            d_model == n_heads * head_dim
        ), f"d_model ({d_model}) must equal n_heads * head_dim ({n_heads * head_dim})"
        assert (
            n_heads % n_kv_heads == 0
        ), f"n_heads ({n_heads}) must be divisible by n_kv_heads ({n_kv_heads})"

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.group_size = n_heads // n_kv_heads
        self.max_seq_len = max_seq_len
        self.rope_theta = rope_theta
        self.attention_dropout = attention_dropout
        self.attn_temperature = attn_temperature
        self.rope = rope

        if rope is not None and hasattr(rope, "attention_temperature"):
            self.attn_temperature = rope.attention_temperature()

        # Projections (no bias as requested)
        self.q_proj = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, d_model, bias=False)

        if external_cos is not None and external_sin is not None:
            # Use externally-supplied cos/sin (e.g. from YaRNRotaryEmbedding)
            self.register_buffer("cos_cached", external_cos, persistent=False)
            self.register_buffer("sin_cached", external_sin, persistent=False)
            inv_freq = 1.0 / (
                rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
            )
            self.register_buffer("inv_freq", inv_freq, persistent=False)
        elif rope is not None and hasattr(rope, "cos_cached"):
            self.register_buffer("cos_cached", rope.cos_cached, persistent=False)
            self.register_buffer("sin_cached", rope.sin_cached, persistent=False)
            self.register_buffer("inv_freq", rope.inv_freq, persistent=False)
        else:
            # Precompute RoPE rotation frequencies (original internal path)
            inv_freq = 1.0 / (
                rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
            )
            self.register_buffer("inv_freq", inv_freq, persistent=False)

            # Build cos/sin lookup table up to max_seq_len
            t = torch.arange(max_seq_len, dtype=torch.float32)
            freqs = torch.outer(t, inv_freq)  # (max_seq_len, head_dim // 2)
            emb = torch.cat((freqs, freqs), dim=-1)  # (max_seq_len, head_dim)
            self.register_buffer("cos_cached", emb.cos(), persistent=False)
            self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def _get_rope(
        self, position_ids: torch.Tensor, dtype: torch.dtype
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        position_ids: (batch, seq_len)
        Returns: cos, sin of shape (batch, 1, seq_len, head_dim) cast to input dtype
        """
        max_pos = int(position_ids.max().item()) + 1
        if max_pos > self.cos_cached.size(0):
            if self.rope is not None and hasattr(self.rope, "build_cache"):
                c, s = self.rope.build_cache(max_pos, self.cos_cached.device, dtype)
                self.cos_cached = c
                self.sin_cached = s
            else:
                t = torch.arange(max_pos, dtype=torch.float32, device=self.cos_cached.device)
                freqs = torch.outer(t, self.inv_freq.to(self.cos_cached.device))
                emb = torch.cat((freqs, freqs), dim=-1)
                self.cos_cached = emb.cos()
                self.sin_cached = emb.sin()

        cos = self.cos_cached[position_ids].to(dtype=dtype)  # (batch, seq_len, head_dim)
        sin = self.sin_cached[position_ids].to(dtype=dtype)  # (batch, seq_len, head_dim)
        return cos.unsqueeze(1), sin.unsqueeze(1)  # (batch, 1, seq_len, head_dim)

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
        attention_mask: Optional[torch.Tensor] = None,
        use_causal_mask: bool = True,
        rope_cos_sin: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        x: (batch, seq_len, d_model)
        position_ids: (batch, seq_len) — for RoPE
        kv_cache: optional cache object for incremental decoding
        attention_mask: optional attention mask tensor
        use_causal_mask: boolean flag for causal attention
        rope_cos_sin: optional pre-computed (cos, sin), each (batch, 1, seq_len, head_dim).
            TransformerModel computes these once per forward and shares them across layers
            (no per-layer table lookup / host sync). When omitted the layer looks them up
            itself, exactly as before.
        returns: (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape

        # 1. Linear projections
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # 2. RoPE on Q and K before KV expansion
        if rope_cos_sin is not None:
            cos, sin = rope_cos_sin
        else:
            cos, sin = self._get_rope(position_ids, dtype=x.dtype)
        q = apply_rotary_pos_emb(q, cos, sin)
        k = apply_rotary_pos_emb(k, cos, sin)

        # 3. Cache update (stores K/V at n_kv_heads size)
        if kv_cache is not None:
            k, v = kv_cache.update(k, v)

        total_seq_len = k.size(2)

        # 4. Expand K and V to n_heads by repeating each KV head group_size times
        k_expanded = repeat_kv(k, self.group_size)  # (batch, n_heads, total_seq_len, head_dim)
        v_expanded = repeat_kv(v, self.group_size)  # (batch, n_heads, total_seq_len, head_dim)

        # 5. Attention computation with SDPA
        # Build rectangular causal mask if use_causal_mask is True and seq_len > 1
        causal_mask = None
        if use_causal_mask and seq_len > 1:
            if seq_len == total_seq_len and attention_mask is None:
                # Optimized fast path for full-sequence prefill / training without explicit padding mask
                is_causal = True
                attn_mask = None
            else:
                # Rectangular causal mask: query i (at pos total_seq_len - seq_len + i) can attend to keys <= total_seq_len - seq_len + i
                q_idx = torch.arange(seq_len, device=x.device).unsqueeze(1)
                k_idx = torch.arange(total_seq_len, device=x.device).unsqueeze(0)
                causal_mask = k_idx <= (total_seq_len - seq_len + q_idx)
                is_causal = False
                attn_mask = causal_mask
        else:
            is_causal = False
            attn_mask = None

        # Combine with explicit attention_mask if provided
        if attention_mask is not None:
            is_causal = False
            if causal_mask is not None:
                if attention_mask.dtype == torch.bool:
                    attn_mask = attention_mask & causal_mask
                else:
                    causal_float = torch.zeros(
                        (seq_len, total_seq_len), device=x.device, dtype=x.dtype
                    )
                    causal_float = causal_float.masked_fill(~causal_mask, float("-inf"))
                    attn_mask = attention_mask + causal_float
            else:
                attn_mask = attention_mask

        attn_dropout_p = self.attention_dropout if self.training else 0.0

        # Scale factor incorporates head_dim and optional YaRN attention temperature
        if self.attn_temperature != 1.0:
            scale = (1.0 / math.sqrt(self.head_dim)) * self.attn_temperature
            attn_out = F.scaled_dot_product_attention(
                q,
                k_expanded,
                v_expanded,
                attn_mask=attn_mask,
                dropout_p=attn_dropout_p,
                is_causal=is_causal,
                scale=scale,
            )
        else:
            attn_out = F.scaled_dot_product_attention(
                q,
                k_expanded,
                v_expanded,
                attn_mask=attn_mask,
                dropout_p=attn_dropout_p,
                is_causal=is_causal,
            )


        # 6. Output projection
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.o_proj(attn_out)
        return output
