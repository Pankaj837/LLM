"""Transformer backbone for the reward model.

This is a small GPT-style decoder written to be *replaced*. The reward head
only needs `forward(input_ids, attention_mask) -> [B, T, d_model]`, so when the
team's own transformer is ready you swap the class and keep everything else.

Integration points for the rest of the team:
  - Shambhavi (positional): replace `_rope` / the position handling in Block.
  - Prem & Afnan (Dynamic NTK, YaRN): these rescale RoPE frequencies; they hook
    the same place, via `rope_base` and `rope_scale`.
  - Aryan (GQA): replace MultiHeadAttention with the grouped-query version.
    `n_kv_head` is already threaded through as a stub argument.
  - Srikar (pruning): operates on a trained model; nothing to change here, but
    see README for why the reward model must be re-evaluated after pruning.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TransformerConfig:
    vocab_size: int
    d_model: int = 128
    n_layer: int = 4
    n_head: int = 4
    n_kv_head: int | None = None   # None == MHA; set < n_head once GQA lands
    max_seq_len: int = 256
    dropout: float = 0.1
    rope_base: float = 10000.0
    rope_scale: float = 1.0        # Dynamic NTK / YaRN hook


def build_rope_cache(seq_len: int, head_dim: int, base: float, scale: float, device):
    """Standard RoPE tables. `scale` stretches positions, which is the knob
    Dynamic NTK and YaRN turn to extend context beyond the trained length."""
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float() / scale
    freqs = torch.outer(t, inv_freq)                       # [T, hd/2]
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(x, cos, sin):
    """x: [B, n_head, T, head_dim]"""
    T = x.size(2)
    cos, sin = cos[:T].unsqueeze(0).unsqueeze(0), sin[:T].unsqueeze(0).unsqueeze(0)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out


class MultiHeadAttention(nn.Module):
    """Causal MHA. Aryan's GQA replaces this; the interface stays the same."""

    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.head_dim = cfg.d_model // cfg.n_head
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.dropout = cfg.dropout

    def forward(self, x, cos, sin, attn_mask):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)

        # is_causal=True plus an additive padding mask: padded positions must
        # not contribute, or the pooled vector shifts with batch composition.
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class Block(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = MultiHeadAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, 4 * cfg.d_model),
            nn.GELU(),
            nn.Linear(4 * cfg.d_model, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x, cos, sin, attn_mask):
        x = x + self.attn(self.ln1(x), cos, sin, attn_mask)
        x = x + self.mlp(self.ln2(x))
        return x


class TransformerBackbone(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, input_ids, attention_mask=None):
        B, T = input_ids.shape
        assert T <= self.cfg.max_seq_len, f"seq {T} > max {self.cfg.max_seq_len}"

        x = self.drop(self.tok_emb(input_ids))
        cos, sin = build_rope_cache(
            T, self.cfg.d_model // self.cfg.n_head,
            self.cfg.rope_base, self.cfg.rope_scale, input_ids.device,
        )

        # Combine causal + padding into one additive float mask.
        causal = torch.ones(T, T, dtype=torch.bool, device=input_ids.device).tril()
        mask = causal.view(1, 1, T, T)
        if attention_mask is not None:
            mask = mask & attention_mask.view(B, 1, 1, T).bool()
        attn_bias = torch.zeros(mask.shape, dtype=x.dtype, device=x.device)
        attn_bias.masked_fill_(~mask, float("-inf"))

        for blk in self.blocks:
            x = blk(x, cos, sin, attn_bias)
        return self.ln_f(x)
