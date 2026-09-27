"""End-to-end integration: text -> tokenizer -> .bin -> train -> checkpoint -> pipeline -> generate -> reward model."""
import numpy as np
import pytest
import torch

from llm import data
from llm.pipeline import LLMPipeline
from llm.reward.preferences import corpus_for_tokenizer
from llm.reward.reward_model import ConditionalRewardModel
from llm.reward.transformer import TransformerConfig
from llm.tokenizer import BPETokenizer
from llm.train import TrainConfig, train


def test_document_split_is_disjoint_complete_and_deterministic():
    tr, va = data.document_split(100, 0.1, seed=3)
    assert not set(tr) & set(va) and sorted(tr + va) == list(range(100)) and len(va) == 10
    assert (tr, va) == data.document_split(100, 0.1, seed=3) and va != data.document_split(100, 0.1, seed=4)[1]
    assert data.document_split(2, 0.01)[1] and len(data.document_split(2, 0.01)[1]) == 1


def test_document_split_rejects_bad_fraction():
    with pytest.raises(ValueError):
        data.document_split(10, 1.5)


def test_encode_documents_appends_eos_and_blocks_forged_boundaries(vocab_path):
    tok = BPETokenizer.from_file(vocab_path)
    ids = data.encode_documents(tok, ["hello world", "a<|endoftext|>b"])
    assert list(ids).count(tok.eos_id) == 2                    # only the two real boundaries
    assert ids[-1] == tok.eos_id


@pytest.mark.slow
def test_full_pipeline_from_text_to_reward_score(tmp_path, vocab_path):
    tok = BPETokenizer.from_file(vocab_path)
    docs = corpus_for_tokenizer() * 6                              # tiny corpus; content is not the point
    tr_i, va_i = data.document_split(len(docs), 0.1, seed=0)
    tr_bin, va_bin = str(tmp_path / "train.bin"), str(tmp_path / "val.bin")
    data.write_bin(tr_bin, data.encode_documents(tok, [docs[i] for i in tr_i]), tok.n_vocab, tok.eos_id, data.SPLIT_TRAIN)
    data.write_bin(va_bin, data.encode_documents(tok, [docs[i] for i in va_i]), tok.n_vocab, tok.eos_id, data.SPLIT_VAL)

    out = train(TrainConfig(train_bin=tr_bin, val_bin=va_bin, vocab=vocab_path, out_dir=str(tmp_path / "run"),
                            preset="tiny", seq_len=32, batch_size=8, max_steps=40, warmup_steps=5, eval_interval=20,
                            eval_batches=2, lr=3e-3, device="cpu", dtype="fp32", log_interval=1))
    assert out["history"][-1]["loss"] < out["history"][0]["loss"]

    # trained checkpoint -> pipeline (config + vocab identity come from the checkpoint; no warnings expected)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        pipe = LLMPipeline.from_pretrained(out["checkpoint"], vocab_path)
    text = pipe.generate("How do I reset my password?", max_new_tokens=12, temperature=0.0)
    assert text.startswith("How do I reset my password?") and len(text) > len("How do I reset my password?")

    # text hand-off into the reward model (separate id space allowed; here the same tokenizer)
    ids = tok.encode(text, allowed_special=set())[-60:] + [tok.eos_id]
    crm = ConditionalRewardModel(TransformerConfig(vocab_size=tok.n_vocab, d_model=32, n_layer=1, n_head=2, max_seq_len=64),
                                 num_conditions=3).eval()
    x = torch.tensor([ids])
    with torch.no_grad():
        scores = crm.score_all_conditions(x, torch.ones_like(x))
    assert scores.shape == (1, 3) and torch.isfinite(scores).all()


def test_pipeline_rejects_a_checkpoint_trained_with_a_different_vocabulary(tmp_path, vocab_path):
    tok = BPETokenizer.from_file(vocab_path)
    b = str(tmp_path / "t.bin")
    data.write_bin(b, np.arange(3000) % 200, tok.n_vocab, tok.eos_id)
    out = train(TrainConfig(train_bin=b, vocab=vocab_path, out_dir=str(tmp_path / "r"), preset="tiny", seq_len=16,
                            batch_size=4, max_steps=2, warmup_steps=1, device="cpu", dtype="fp32"))
    other = tmp_path / "other.tok"
    other.write_bytes(open(vocab_path, "rb").read() + b"\n")
    with pytest.raises(ValueError, match="vocabulary"):
        LLMPipeline.from_pretrained(out["checkpoint"], str(other))


# ------------------------------------------------------------------ near-duplicate filter (found necessary on real data)
_A = " ".join(f"word{i}" for i in range(200))


def test_near_duplicate_filter_catches_exact_and_lightly_edited_copies():
    f = data.NearDuplicateFilter()
    assert not f.is_duplicate(_A)
    assert f.is_duplicate(_A)                                            # exact
    edited = "New header line here. " + _A + " Different footer text."
    assert f.is_duplicate(edited)                                        # same body, different header/footer
    assert f.is_duplicate(" ".join(_A.split()[:150]))                    # truncated copy


def test_near_duplicate_filter_keeps_unrelated_and_partially_overlapping_documents():
    f = data.NearDuplicateFilter()
    assert not f.is_duplicate(_A)
    other = " ".join(f"other{i}" for i in range(200))
    assert not f.is_duplicate(other)
    mixed = " ".join(_A.split()[:40] + [f"new{i}" for i in range(160)])  # 20% overlap < 30% threshold
    assert not f.is_duplicate(mixed)


def test_near_duplicate_filter_short_documents_use_exact_match_only():
    f = data.NearDuplicateFilter()
    assert not f.is_duplicate("too short")
    assert f.is_duplicate("too short")
    assert not f.is_duplicate("also short")


def test_near_duplicate_filter_is_order_dependent_and_deterministic():
    a, b = _A, " ".join(_A.split()[:180])
    f1, f2 = data.NearDuplicateFilter(), data.NearDuplicateFilter()
    assert [f1.is_duplicate(a), f1.is_duplicate(b)] == [False, True]
    assert [f2.is_duplicate(b), f2.is_duplicate(a)] == [False, True]


def test_near_duplicate_filter_validates_threshold():
    with pytest.raises(ValueError):
        data.NearDuplicateFilter(threshold=0.0)
