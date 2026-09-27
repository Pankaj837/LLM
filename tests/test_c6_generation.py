"""C6 Autoregressive generation (sampling, KV-cache decode, stopping)."""
import pytest
import torch

from helpers import small_lm_config
from llm.model import TransformerModel

PROMPT = torch.tensor([[10, 25, 42, 99, 105]])
EOS, OTHER = 7, 9


@pytest.fixture(scope="module")
def lm():
    torch.manual_seed(0)
    return TransformerModel(small_lm_config()).eval()


def test_cached_equals_uncached_greedy(lm):
    a = lm.generate(PROMPT, max_new_tokens=8, temperature=0.0, use_cache=True)
    b = lm.generate(PROMPT, max_new_tokens=8, temperature=0.0, use_cache=False)
    assert torch.equal(a, b)


def test_cached_equals_uncached_greedy_batched(lm):
    p = torch.randint(0, 1722, (3, 6))
    assert torch.equal(lm.generate(p, max_new_tokens=6, temperature=0.0),
                       lm.generate(p, max_new_tokens=6, temperature=0.0, use_cache=False))


def test_output_extends_prompt_without_modifying_it(lm):
    out = lm.generate(PROMPT, max_new_tokens=5, temperature=0.0)
    assert out.shape == (1, 10) and torch.equal(out[:, :5], PROMPT)


def test_sampling_is_seed_deterministic(lm):
    torch.manual_seed(3); a = lm.generate(PROMPT, max_new_tokens=8, temperature=1.0)
    torch.manual_seed(3); b = lm.generate(PROMPT, max_new_tokens=8, temperature=1.0)
    assert torch.equal(a, b)


def test_top_k_1_equals_greedy(lm):
    assert torch.equal(lm.generate(PROMPT, max_new_tokens=6, top_k=1),
                       lm.generate(PROMPT, max_new_tokens=6, temperature=0.0))


def test_top_k_restricts_to_k_best(lm):
    with torch.no_grad():
        topk = set(torch.topk(lm(PROMPT)[0, -1], 3).indices.tolist())
    for seed in range(20):
        torch.manual_seed(seed)
        assert lm.generate(PROMPT, max_new_tokens=1, top_k=3)[0, -1].item() in topk


def test_top_p_keeps_at_least_the_argmax(lm):
    with torch.no_grad():
        best = lm(PROMPT)[0, -1].argmax().item()
    for seed in range(10):
        torch.manual_seed(seed)
        assert lm.generate(PROMPT, max_new_tokens=1, top_p=1e-9, top_k=0)[0, -1].item() == best


@pytest.mark.parametrize("cache", [pytest.param(True, marks=pytest.mark.finding("F-07")), False])
def test_zero_new_tokens_returns_prompt_unchanged(lm, cache):
    """Cached path returns 1 extra token (F-07); the uncached path is already correct."""
    assert lm.generate(PROMPT, max_new_tokens=0, temperature=0.0, use_cache=cache).shape == PROMPT.shape


@pytest.mark.finding("F-08")
def test_empty_prompt_raises_clear_error(lm):
    with pytest.raises(ValueError):
        lm.generate(torch.zeros((1, 0), dtype=torch.long), max_new_tokens=3)


@pytest.mark.finding("F-08")
def test_negative_temperature_rejected(lm):
    with pytest.raises(ValueError):
        lm.generate(PROMPT, max_new_tokens=3, temperature=-1.0)


def test_out_of_range_token_id_raises(lm):
    with pytest.raises(IndexError):
        lm(torch.tensor([[1722]]))


def _scripted(lm, monkeypatch, plan):
    """Replace the network with a script: plan(step, row) -> token to emit."""
    def fake_forward(ids, *a, **k):
        B, T = ids.shape
        step = T - PROMPT.shape[1]
        lg = torch.zeros(B, T, 1722)
        for r in range(B):
            lg[r, -1, plan(max(step, 0), r)] = 10.0
        return lg
    monkeypatch.setattr(lm, "forward", fake_forward)


@pytest.mark.finding("F-09")
def test_finished_rows_stay_finished(lm, monkeypatch):
    """Row 0 emits EOS at step 0; row 1 never does. Row 0 must keep emitting EOS, not resume generating."""
    _scripted(lm, monkeypatch, lambda step, row: EOS if (row == 0 and step == 0) else OTHER)
    p = PROMPT.repeat(2, 1)
    out = lm.generate(p, max_new_tokens=4, temperature=0.0, eos_token_id=EOS, use_cache=False)
    assert out[0, 5:].tolist() == [EOS] * 4
    assert out[1, 5:].tolist() == [OTHER] * 4


def test_stops_early_when_all_rows_hit_eos(lm, monkeypatch):
    _scripted(lm, monkeypatch, lambda step, row: EOS)
    out = lm.generate(PROMPT.repeat(2, 1), max_new_tokens=10, temperature=0.0, eos_token_id=EOS, use_cache=False)
    assert out.shape[1] == 6


@pytest.mark.finding("F-10")
def test_exceeding_trained_context_warns(lm):
    long_prompt = torch.randint(0, 1722, (1, 250))
    with pytest.warns(UserWarning, match="max_position_embeddings"):
        lm.generate(long_prompt, max_new_tokens=20, temperature=0.0)
