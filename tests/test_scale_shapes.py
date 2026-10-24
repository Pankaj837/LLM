"""The 1.51B / 2B presets run with their real widths, heads, KV heads and FFN sizes (one layer instead of 32 to fit in RAM)."""
import math

import pytest
import torch

from llm.model import TransformerModel
from llm.model_config import ModelConfig
from llm.train import PRESETS


@pytest.mark.parametrize("name", ["1p5b", "2b"])
def test_scale_preset_widths_forward_backward_and_long_context(name):
    cfg = ModelConfig(**{**PRESETS[name], "num_layers": 1, "vocab_size": 32000})
    m = TransformerModel(cfg)
    x = torch.randint(0, 32000, (1, 128))
    y = torch.randint(0, 32000, (1, 128))
    logits = m(x)
    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 32000).float(), y.reshape(-1))
    loss.backward()
    assert logits.shape == (1, 128, 32000)
    # at init the loss is ln(V) plus a little from the logit spread (0.02 * sqrt(2048) ~ 0.9 -> +~0.4)
    assert math.log(32000) - 0.2 < loss.item() < math.log(32000) + 1.0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
    with torch.no_grad():                                                       # the preset's full 4K context window
        assert torch.isfinite(m(torch.randint(0, 32000, (1, cfg.max_position_embeddings)))).all()
