"""C7 Checkpoint (pretrain_model.pt + loader)  and  C8 Pipeline (tokenizer + model + generate)."""
import pytest
import torch

from helpers import small_lm_config, to_checkpoint_format
from llm.model import TransformerModel
from llm.pipeline import LLMPipeline


# ================================================================== C7 loader behaviour (tiny model)
@pytest.fixture()
def tiny_ckpt(tmp_path):
    torch.manual_seed(0)
    cfg = small_lm_config()
    m = TransformerModel(cfg)
    p = tmp_path / "tiny.pt"
    torch.save(to_checkpoint_format(m.state_dict()), p)
    return cfg, m, str(p)


def test_loader_roundtrips_weights_exactly(tiny_ckpt):
    cfg, src, path = tiny_ckpt
    dst = TransformerModel(cfg)
    missing, unexpected = dst.load_pretrained(path)
    assert not missing and not unexpected
    ids = torch.randint(0, 1722, (1, 8))
    with torch.no_grad():
        assert torch.equal(src.eval()(ids), dst.eval()(ids))


def test_loader_strict_mode_detects_wrong_shape(tiny_ckpt):
    cfg, _, path = tiny_ckpt
    wrong = TransformerModel(small_lm_config(num_layers=3))
    with pytest.raises(RuntimeError):
        wrong.load_pretrained(path, strict=True)


def test_loader_reports_missing_keys_in_lenient_mode(tiny_ckpt):
    cfg, _, path = tiny_ckpt
    missing, _ = TransformerModel(small_lm_config(num_layers=3)).load_pretrained(path, strict=False)
    assert any("layers.2." in k for k in missing)


# ================================================================== C8 pipeline
@pytest.fixture()
def pipe_files(tmp_path, vocab_path):
    torch.manual_seed(0)
    cfg = small_lm_config()
    p = tmp_path / "tiny.pt"
    torch.save(to_checkpoint_format(TransformerModel(cfg).state_dict()), p)
    return cfg, str(p), vocab_path


def test_pipeline_end_to_end_returns_text(pipe_files):
    cfg, ck, vocab = pipe_files
    pipe = LLMPipeline.from_pretrained(ck, vocab, config=cfg)
    out = pipe.generate("How do I make better coffee?", max_new_tokens=6, temperature=0.0)
    assert isinstance(out, str) and out.startswith("How do I make better coffee?")


def test_pipeline_greedy_is_deterministic(pipe_files):
    cfg, ck, vocab = pipe_files
    pipe = LLMPipeline.from_pretrained(ck, vocab, config=cfg)
    assert pipe.generate("Hello", max_new_tokens=6, temperature=0.0) == pipe.generate("Hello", max_new_tokens=6, temperature=0.0)


@pytest.mark.finding("F-12")
def test_missing_checkpoint_raises(pipe_files, tmp_path):
    cfg, _, vocab = pipe_files
    with pytest.raises(FileNotFoundError):
        LLMPipeline.from_pretrained(str(tmp_path / "typo.pt"), vocab, config=cfg)


@pytest.mark.finding("F-12")
def test_missing_vocab_raises(pipe_files, tmp_path):
    cfg, ck, _ = pipe_files
    with pytest.raises(FileNotFoundError):
        LLMPipeline.from_pretrained(ck, str(tmp_path / "nope.tok"), config=cfg)


@pytest.mark.finding("F-12")
def test_checkpoint_config_mismatch_raises(pipe_files):
    cfg, ck, vocab = pipe_files
    with pytest.raises(RuntimeError):
        LLMPipeline.from_pretrained(ck, vocab, config=small_lm_config(num_layers=3))


@pytest.mark.finding("F-12")
def test_tokenizer_larger_than_embedding_raises(pipe_files):
    cfg, ck, vocab = pipe_files
    small_cfg = small_lm_config(vocab_size=500)
    torch.save(to_checkpoint_format(TransformerModel(small_cfg).state_dict()), ck)
    with pytest.raises(ValueError):
        LLMPipeline.from_pretrained(ck, vocab, config=small_cfg)


@pytest.mark.finding("F-13")
def test_user_text_cannot_inject_eos(pipe_files, monkeypatch):
    """'<|endoftext|>' typed by a user must not become the real EOS control token."""
    cfg, ck, vocab = pipe_files
    pipe = LLMPipeline.from_pretrained(ck, vocab, config=cfg)
    seen = {}
    real = pipe.model.generate

    def spy(token_ids, **kw):
        seen["ids"] = token_ids[0].tolist()
        return real(token_ids, **kw)

    monkeypatch.setattr(pipe.model, "generate", spy)
    pipe.generate("hello<|endoftext|>world", max_new_tokens=2, temperature=0.0)
    assert pipe.tokenizer.eot_token_id not in seen["ids"]


def test_pipeline_empty_prompt_raises_value_error(pipe_files):
    cfg, ck, vocab = pipe_files
    pipe = LLMPipeline.from_pretrained(ck, vocab, config=cfg)
    with pytest.raises((ValueError, RuntimeError)):
        pipe.generate("", max_new_tokens=2)
