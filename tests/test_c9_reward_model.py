"""C9 Conditional reward model: backbone, FiLM / concat conditioning, pooling, Bradley-Terry loss."""
import pytest
import torch
import torch.nn.functional as F

from llm.reward.reward_model import ConditionalRewardModel, FiLMConditioner, preference_loss
from llm.reward.transformer import TransformerBackbone, TransformerConfig

CFG = TransformerConfig(vocab_size=50, d_model=32, n_layer=2, n_head=4, max_seq_len=32, dropout=0.0)


def _model(kind="film", seed=0):
    torch.manual_seed(seed)
    return ConditionalRewardModel(CFG, num_conditions=3, cond_dim=16, conditioning=kind).eval()


def _batch(B=3, T=8, lengths=None):
    torch.manual_seed(1)
    ids = torch.randint(3, 50, (B, T))
    mask = torch.ones(B, T, dtype=torch.long)
    for i, L in enumerate(lengths or [T] * B):
        mask[i, L:] = 0
        ids[i, L:] = 0
    return ids, mask


# ------------------------------------------------------------------ components
def test_film_is_identity_at_init():
    f = FiLMConditioner(3, 16, 32)
    h = torch.randn(4, 32)
    assert torch.allclose(f(h, torch.tensor([0, 1, 2, 0])), h)


def test_film_becomes_condition_dependent_after_a_step():
    f = FiLMConditioner(3, 16, 32)
    with torch.no_grad():
        f.to_gamma.weight.add_(0.1); f.to_beta.weight.add_(0.1)
    h = torch.randn(1, 32)
    a, b = f(h, torch.tensor([0])), f(h, torch.tensor([1]))
    assert not torch.allclose(a, b)


def test_backbone_is_causal():
    bb = TransformerBackbone(CFG).eval()
    ids, mask = _batch(1, 10)
    ids2 = ids.clone(); ids2[:, 6:] = torch.randint(3, 50, (1, 4))
    with torch.no_grad():
        a, b = bb(ids, mask), bb(ids2, mask)
    assert torch.allclose(a[:, :6], b[:, :6], atol=1e-6)


def test_backbone_rejects_over_length_input():
    with pytest.raises(AssertionError):
        TransformerBackbone(CFG)(torch.randint(0, 50, (1, 40)))


# ------------------------------------------------------------------ scoring semantics
@pytest.mark.parametrize("kind", ["film", "concat"])
def test_score_shape_and_finite(kind):
    m, (ids, mask) = _model(kind), _batch()
    with torch.no_grad():
        s = m(ids, mask, torch.tensor([0, 1, 2]))
    assert s.shape == (3,) and torch.isfinite(s).all()


@pytest.mark.parametrize("kind", ["film", "concat"])
def test_score_all_conditions_equals_per_condition_forward(kind):
    m, (ids, mask) = _model(kind), _batch()
    # perturb so conditions differ
    with torch.no_grad():
        for p in m.parameters():
            p.add_(torch.randn_like(p) * 0.02)
        allc = m.score_all_conditions(ids, mask)
        for c in range(3):
            assert torch.allclose(allc[:, c], m(ids, mask, torch.full((3,), c)), atol=1e-5)


def test_scores_do_not_depend_on_padding_or_batch_composition():
    m = _model()
    short, mk_s = _batch(1, 5)
    long_, mk_l = _batch(1, 12)
    T = 12
    pad_ids = torch.zeros(2, T, dtype=torch.long); pad_mask = torch.zeros(2, T, dtype=torch.long)
    pad_ids[0, :5], pad_mask[0, :5] = short[0], 1
    pad_ids[1], pad_mask[1] = long_[0], 1
    cond = torch.zeros(2, dtype=torch.long)
    with torch.no_grad():
        batched = m(pad_ids, pad_mask, cond)
        alone = torch.cat([m(short, mk_s, cond[:1]), m(long_, mk_l, cond[:1])])
    assert torch.allclose(batched, alone, atol=1e-5)


def test_pooling_reads_last_real_token_only():
    m = _model()
    ids, mask = _batch(1, 8, lengths=[5])
    with torch.no_grad():
        base = m(ids, mask, torch.tensor([0]))
        ids_pad_changed = ids.clone(); ids_pad_changed[:, 5:] = 49       # garbage in padded slots
        assert torch.allclose(base, m(ids_pad_changed, mask, torch.tensor([0])), atol=1e-6)
        ids_last_changed = ids.clone(); ids_last_changed[:, 4] = (ids[0, 4] % 40) + 5
        assert not torch.allclose(base, m(ids_last_changed, mask, torch.tensor([0])))


def test_eval_mode_is_deterministic():
    m, (ids, mask) = _model(), _batch()
    with torch.no_grad():
        assert torch.equal(m(ids, mask, torch.tensor([0, 1, 2])), m(ids, mask, torch.tensor([0, 1, 2])))


def test_out_of_range_condition_id_raises():
    m, (ids, mask) = _model(), _batch()
    with pytest.raises(IndexError):
        m(ids, mask, torch.tensor([0, 1, 7]))


@pytest.mark.finding("F-16")
def test_all_padding_row_is_rejected():
    """A fully-masked row silently pools position -1 (the last padded slot)."""
    m, (ids, mask) = _model(), _batch()
    mask[1] = 0
    with pytest.raises(ValueError):
        m(ids, mask, torch.tensor([0, 1, 2]))


# ------------------------------------------------------------------ loss
def test_bradley_terry_matches_manual_formula():
    rc, rr = torch.tensor([1.0, 0.2]), torch.tensor([0.0, 0.5])
    total, bt = preference_loss(rc, rr, l2_coef=0.0)
    manual = -torch.log(torch.sigmoid(rc - rr)).mean()
    assert torch.allclose(bt, manual) and torch.allclose(total, manual)


def test_bradley_terry_loss_is_lower_for_correct_ordering():
    good = preference_loss(torch.tensor([2.0]), torch.tensor([-2.0]))[1]
    bad = preference_loss(torch.tensor([-2.0]), torch.tensor([2.0]))[1]
    assert good < 0.05 < 2.0 < bad


def test_l2_term_penalises_scale_but_not_ranking():
    """Adding a constant to both rewards leaves the BT term unchanged but raises the total."""
    rc, rr = torch.tensor([1.0]), torch.tensor([0.0])
    t0, b0 = preference_loss(rc, rr)
    t1, b1 = preference_loss(rc + 10, rr + 10)
    assert torch.allclose(b0, b1) and t1 > t0


def test_gradients_reach_condition_parameters():
    """FiLM is zero-initialised (identity), so on step 1 only to_gamma/to_beta receive gradient and the
    condition embedding gets exactly zero; after one update the embedding must start learning too."""
    m, (ids, mask) = _model(), _batch()
    m.train()
    cond = torch.tensor([0, 1, 2])
    opt = torch.optim.SGD(m.parameters(), lr=0.1)

    def step():
        opt.zero_grad()
        loss, _ = preference_loss(m(ids, mask, cond), m(ids.flip(0), mask.flip(0), cond))
        loss.backward()

    step()
    assert m.film.to_gamma.weight.grad.abs().sum() > 0 and m.film.to_beta.weight.grad.abs().sum() > 0
    assert m.film.emb.weight.grad.abs().sum() == 0            # documented zero-init behaviour
    opt.step(); step()
    assert m.film.emb.weight.grad.abs().sum() > 0


# ------------------------------------------------------------------ the mechanism itself (mock task)
def _toy_train(shuffle_conditions: bool, steps=150, seed=0):
    """Two criteria that disagree on the SAME pair: cond 0 prefers response token 3, cond 1 prefers token 4.
    Only a model that reads the condition can fit both."""
    torch.manual_seed(seed)
    cfg = TransformerConfig(vocab_size=8, d_model=32, n_layer=1, n_head=2, max_seq_len=4, dropout=0.0)
    m = ConditionalRewardModel(cfg, num_conditions=2, cond_dim=8, conditioning="film")
    opt = torch.optim.Adam(m.parameters(), lr=5e-3)
    seq = lambda r: torch.tensor([[1, r, 2]])                       # prompt, response, EOS
    x3, x4 = seq(3).repeat(2, 1), seq(4).repeat(2, 1)
    mask, cond = torch.ones(2, 3, dtype=torch.long), torch.tensor([0, 1])
    chosen, rejected = torch.cat([x3[:1], x4[1:]]), torch.cat([x4[:1], x3[1:]])
    for _ in range(steps):
        c = cond[torch.randperm(2)] if shuffle_conditions else cond
        loss, _ = preference_loss(m(chosen, mask, c), m(rejected, mask, c))
        opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    with torch.no_grad():
        rc, rr = m(chosen, mask, cond), m(rejected, mask, cond)
    return (rc > rr).float().mean().item(), m


def test_conditional_model_learns_opposing_criteria_and_flips_ranking():
    acc, m = _toy_train(shuffle_conditions=False)
    assert acc == 1.0
    mask = torch.ones(2, 3, dtype=torch.long)
    with torch.no_grad():
        s = m.score_all_conditions(torch.tensor([[1, 3, 2], [1, 4, 2]]), mask)   # [2 responses, 2 conds]
    assert s[0, 0] > s[1, 0] and s[0, 1] < s[1, 1]                              # ranking flips with condition


def test_ablation_control_cannot_learn_opposing_criteria():
    acc, _ = _toy_train(shuffle_conditions=True)
    assert acc < 1.0
