"""
Core sub-modules required by the pretrained checkpoint:
  - RMSNorm   (weight-only layer norm, no bias)
  - SwiGLUFeedForward  (gate / up / down projections, SiLU activation)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalisation (Zhang & Sennrich, 2019).

    The checkpoint stores a single `weight` vector per norm layer and no bias.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # float32 for numerical stability during norm computation
        dtype = x.dtype
        x = x.float()
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        x = x / rms
        return (self.weight * x).to(dtype)


class SwiGLUFeedForward(nn.Module):
    """SwiGLU feed-forward network (Shazeer, 2020).

    Checkpoint keys per layer:
        ffn.W_gate.weight  [intermediate_dim, hidden_dim]
        ffn.W_up.weight    [intermediate_dim, hidden_dim]
        ffn.W_down.weight  [hidden_dim, intermediate_dim]

    We name our parameters to match the checkpoint mapping in model.py.
    """

    def __init__(self, hidden_dim: int, intermediate_dim: int):
        super().__init__()
        self.W_gate = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.W_up = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.W_down = nn.Linear(intermediate_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.W_down(F.silu(self.W_gate(x)) * self.W_up(x))
