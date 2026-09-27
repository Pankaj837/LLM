"""Conditional Reward Model.

A standard reward model maps (prompt, response) -> scalar. It encodes a single,
fixed notion of "better", which is a problem: human preference is not one axis.
The same pair of responses can be ranked oppositely depending on what you are
optimising for. A thorough three-paragraph answer beats a one-liner on
helpfulness and loses to it on conciseness.

A *conditional* reward model takes the criterion as an explicit input:

    r = f(prompt, response, condition)

so one set of weights covers every criterion, and at RL time you choose (or mix)
conditions instead of retraining a separate reward model per axis.

Conditioning strategy — three options, this file implements the first two:

  1. FiLM (default). The condition produces per-feature scale and shift applied
     to the pooled representation: h' = gamma(c) * h + beta(c). Multiplicative,
     so the condition can gate features on and off rather than just adding a
     bias the head can learn to ignore. This is the failure mode that matters:
     with plain concatenation the head can drive the condition weights toward
     zero and still fit the training set, giving a model that scores well
     overall and ignores the condition entirely.

  2. Concat. h' = [h ; emb(c)] into an MLP. Simpler baseline, included so the
     writeup can show FiLM is actually earning its extra parameters.

  3. Text prefix (not implemented). Prepend "<|helpful|>" as a real token.
     Reuses the LM's semantics and generalises to unseen criteria phrased in
     natural language, but costs context length and forces a separate backbone
     pass per condition — which breaks the caching trick below.

Both implemented options are *late fusion*: the backbone never sees the
condition. That means one backbone pass can be scored under every condition,
which is what makes the condition-swap evaluation cheap.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .transformer import TransformerBackbone, TransformerConfig


class FiLMConditioner(nn.Module):
    """Produces per-feature (scale, shift) from a condition id."""

    def __init__(self, num_conditions: int, cond_dim: int, d_model: int):
        super().__init__()
        self.emb = nn.Embedding(num_conditions, cond_dim)
        self.to_gamma = nn.Linear(cond_dim, d_model)
        self.to_beta = nn.Linear(cond_dim, d_model)
        # Init so the model starts at gamma=1, beta=0 (i.e. identity). Starting
        # away from identity makes early training noisy for no benefit.
        nn.init.zeros_(self.to_gamma.weight); nn.init.zeros_(self.to_gamma.bias)
        nn.init.zeros_(self.to_beta.weight);  nn.init.zeros_(self.to_beta.bias)

    def forward(self, h, condition_ids):
        c = self.emb(condition_ids)
        gamma = 1.0 + self.to_gamma(c)
        beta = self.to_beta(c)
        return gamma * h + beta


class ConditionalRewardModel(nn.Module):
    def __init__(
        self,
        cfg: TransformerConfig,
        num_conditions: int,
        cond_dim: int = 64,
        conditioning: str = "film",   # "film" | "concat"
        pad_id: int = 0,
        backbone: Optional[nn.Module] = None,
    ):
        super().__init__()
        assert conditioning in ("film", "concat")
        self.conditioning = conditioning
        self.pad_id = pad_id
        self.num_conditions = num_conditions

        # `backbone` may be any module with forward(input_ids, attention_mask) -> [B, T, cfg.d_model]
        # (see LMBackboneAdapter for the team's LM). Default: the small standalone GPT.
        self.backbone = backbone if backbone is not None else TransformerBackbone(cfg)

        if conditioning == "film":
            self.film = FiLMConditioner(num_conditions, cond_dim, cfg.d_model)
            head_in = cfg.d_model
        else:
            self.cond_emb = nn.Embedding(num_conditions, cond_dim)
            head_in = cfg.d_model + cond_dim

        self.head = nn.Sequential(
            nn.Linear(head_in, cfg.d_model),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model, 1),
        )
        # Small init on the final layer keeps initial rewards near zero, so the
        # Bradley-Terry logit starts near 0.5 and gradients are well scaled.
        nn.init.normal_(self.head[-1].weight, std=0.01)
        nn.init.zeros_(self.head[-1].bias)

    # ------------------------------------------------------------------ pool

    def encode(self, input_ids, attention_mask):
        """Backbone pass + pooling. Condition-independent, so the result can be
        reused across every condition — see `score_all_conditions`."""
        h = self.backbone(input_ids, attention_mask)          # [B, T, D]
        # Pool at the last real token. With a causal backbone this is the only
        # position that has attended to the whole sequence; mean-pooling would
        # dilute it with prefix positions that cannot see the response.
        lengths = attention_mask.sum(dim=1)                   # [B]
        if (lengths < 1).any():
            raise ValueError("every row needs at least one real (unmasked) token")
        last_idx = lengths - 1                                # [B] (right padding assumed)
        return h[torch.arange(h.size(0), device=h.device), last_idx]

    def score_from_pooled(self, pooled, condition_ids):
        if self.conditioning == "film":
            z = self.film(pooled, condition_ids)
        else:
            z = torch.cat([pooled, self.cond_emb(condition_ids)], dim=-1)
        return self.head(z).squeeze(-1)                        # [B]

    def forward(self, input_ids, attention_mask, condition_ids):
        return self.score_from_pooled(self.encode(input_ids, attention_mask), condition_ids)

    @torch.no_grad()
    def score_all_conditions(self, input_ids, attention_mask):
        """[B, num_conditions] scores from a single backbone pass."""
        pooled = self.encode(input_ids, attention_mask)
        B = pooled.size(0)
        out = []
        for c in range(self.num_conditions):
            cid = torch.full((B,), c, dtype=torch.long, device=pooled.device)
            out.append(self.score_from_pooled(pooled, cid))
        return torch.stack(out, dim=1)


def preference_loss(r_chosen, r_rejected, l2_coef: float = 1e-3):
    """Bradley-Terry: P(chosen > rejected) = sigmoid(r_c - r_r).

    Only the *difference* is supervised, so the absolute scale is unidentified
    and will drift outward without the L2 term — which then destabilises
    whatever RL stage consumes these rewards. The penalty pins the scale
    without touching the ranking.
    """
    loss = -F.logsigmoid(r_chosen - r_rejected).mean()
    reg = l2_coef * (r_chosen.pow(2) + r_rejected.pow(2)).mean()
    return loss + reg, loss.detach()


class LMBackboneAdapter(nn.Module):
    """Lets the team's `TransformerModel` act as the reward-model backbone.

    Satisfies the backbone contract forward(input_ids, attention_mask) -> [B, T, D]
    by requesting final-norm hidden states instead of logits. Inputs must be
    RIGHT-padded (the pooling step reads the last real token).
    Usage:
        lm = TransformerModel(ModelConfig()); lm.load_pretrained(path)
        cfg = TransformerConfig(vocab_size=lm.config.vocab_size, d_model=lm.config.hidden_dim)
        crm = ConditionalRewardModel(cfg, num_conditions=3, backbone=LMBackboneAdapter(lm))
    """

    def __init__(self, lm: nn.Module):
        super().__init__()
        self.lm = lm
        self.d_model = lm.config.hidden_dim

    def forward(self, input_ids, attention_mask=None):
        return self.lm(input_ids, attention_mask=attention_mask, return_hidden=True)
