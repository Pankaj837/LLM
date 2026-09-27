"""C11 Cross-component contracts: LM -> hidden states -> reward head, and the text hand-off between tokenizers."""
import pytest
import torch

from helpers import small_lm_config, to_checkpoint_format
from llm.model import TransformerModel
from llm.pipeline import LLMPipeline

CFG = small_lm_config()


@pytest.fixture(scope="module")
def lm():
    torch.manual_seed(0)
    return TransformerModel(CFG).eval()


@pytest.mark.finding("F-14")
def test_lm_exposes_final_hidden_states(lm):
    ids = torch.randint(0, 1722, (2, 7))
    with torch.no_grad():
        h = lm(ids, return_hidden=True)
    assert h.shape == (2, 7, CFG.hidden_dim)


@pytest.mark.finding("F-15")
def test_lm_accepts_2d_padding_mask(lm):
    """Reward-model style (B,T) 0/1 mask; right-padded row must match the unpadded run."""
    ids = torch.randint(0, 1722, (2, 7))
    mask = torch.ones(2, 7, dtype=torch.long); mask[1, 4:] = 0
    with torch.no_grad():
        padded = lm(ids, attention_mask=mask)
        alone = lm(ids[1:2, :4])
    assert torch.allclose(padded[1, :4], alone[0], atol=1e-5)


@pytest.mark.finding("F-14")
def test_lm_backbone_adapter_drives_reward_head_with_padding_invariance(lm):
    from llm.reward.reward_model import ConditionalRewardModel, LMBackboneAdapter
    from llm.reward.transformer import TransformerConfig
    cfg = TransformerConfig(vocab_size=CFG.vocab_size, d_model=CFG.hidden_dim, dropout=0.0)
    crm = ConditionalRewardModel(cfg, num_conditions=3, backbone=LMBackboneAdapter(lm)).eval()
    ids = torch.randint(3, 1722, (2, 9)); mask = torch.ones(2, 9, dtype=torch.long); mask[1, 5:] = 0
    cond = torch.tensor([0, 2])
    with torch.no_grad():
        batched = crm(ids, mask, cond)
        alone = crm(ids[1:2, :5], mask[1:2, :5], cond[1:])
    assert batched.shape == (2,) and torch.allclose(batched[1], alone[0], atol=1e-5)


def test_text_level_handoff_lm_to_reward_model(tmp_path, vocab_path):
    """The integration path that works today: LM text -> reward tokenizer -> reward model."""
    from llm.tokenizer import BPETokenizer as CrmTok
    from llm.reward.reward_model import ConditionalRewardModel
    from llm.reward.transformer import TransformerConfig

    ck = tmp_path / "lm.pt"
    torch.save(to_checkpoint_format(TransformerModel(CFG).state_dict()), ck)
    pipe = LLMPipeline.from_pretrained(str(ck), vocab_path, config=CFG)
    text = pipe.generate("How do I reset my password?", max_new_tokens=8, temperature=0.0)

    tok = CrmTok.from_file(vocab_path)
    ids = tok.encode(text)[-64:] + [tok.eos_id]
    crm = ConditionalRewardModel(TransformerConfig(vocab_size=tok.n_vocab, d_model=32, n_layer=1, n_head=2, max_seq_len=64),
                                 num_conditions=3).eval()
    x = torch.tensor([ids]); m = torch.ones_like(x)
    with torch.no_grad():
        scores = crm.score_all_conditions(x, m)
    assert scores.shape == (1, 3) and torch.isfinite(scores).all()
