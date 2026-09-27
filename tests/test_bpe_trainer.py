"""Pure-Python BPE trainer: format, invariants, determinism, agreement with the encoder."""
import random

import pytest

from llm.bpe_trainer import train_bpe, write_tok
from llm.tokenizer import BPETokenizer

CORPUS = [
    "the cat sat on the mat and the cat ate the rat",
    "the quick brown fox jumps over the lazy dog",
    "cells divide and the cell membrane regulates transport of proteins",
    "DNA encodes proteins; RNA carries the message from DNA to the ribosome",
    "they're not going; we've seen it and I'll say it's fine",
    "numbers like 12345 and 2026 appear with punctuation, commas... and dashes---",
    "café naïve 日本語 \U0001f600",
] * 20


def _tok(vocab, tmp_path, name="v.tok"):
    p = str(tmp_path / name)
    n = write_tok(vocab, p)
    return BPETokenizer.from_file(p), n


def test_byte_tokens_come_first_and_merges_are_unique():
    v = train_bpe(CORPUS, 400, min_count=2)
    assert v[:256] == [bytes([i]) for i in range(256)]
    assert len(v) == len(set(v)) and len(v) <= 399


def test_every_merge_is_a_concatenation_of_two_earlier_tokens():
    v = train_bpe(CORPUS, 400, min_count=2)
    index = {t: i for i, t in enumerate(v)}
    for i in range(256, len(v)):
        ok = any(v[i][:k] in index and v[i][k:] in index and index[v[i][:k]] < i and index[v[i][k:]] < i
                 for k in range(1, len(v[i])))
        assert ok, (i, v[i])


def test_is_deterministic():
    assert train_bpe(CORPUS, 400) == train_bpe(CORPUS, 400)


def test_stops_when_no_pair_repeats():
    v = train_bpe(["abcdefg"], 1000, min_count=2)
    assert len(v) == 256                                   # nothing occurs twice


def test_rejects_vocab_too_small_for_bytes():
    with pytest.raises(ValueError):
        train_bpe(CORPUS, 100)


def test_written_file_loads_and_reports_sizes(tmp_path):
    v = train_bpe(CORPUS, 400)
    tok, n = _tok(v, tmp_path)
    assert tok.n_vocab == n == len(v) + 1 and tok.eos_id == len(v)


def test_roundtrip_through_the_encoder_on_seen_and_unseen_text(tmp_path):
    tok, _ = _tok(train_bpe(CORPUS, 500), tmp_path)
    rng = random.Random(0)
    unseen = ["Completely unseen text: αβγ üñ \U0001f680", "", " ", "\n\n", "x" * 200,
              "".join(rng.choice("abc def\n\t.,'é日") for _ in range(300))]
    for s in CORPUS[:7] + unseen:
        assert tok.decode(tok.encode(s, allowed_special=set())) == s


def test_more_merges_never_lengthen_the_encoding(tmp_path):
    text = " ".join(CORPUS[:5])
    lens = []
    for size in (300, 400, 600):
        tok, _ = _tok(train_bpe(CORPUS, size), tmp_path, f"v{size}.tok")
        lens.append(len(tok.encode(text, allowed_special=set())))
    assert lens[0] >= lens[1] >= lens[2] and lens[0] > lens[2]


def test_common_words_become_single_tokens(tmp_path):
    tok, _ = _tok(train_bpe(CORPUS, 600), tmp_path)
    assert len(tok.encode(" the", allowed_special=set())) == 1
    assert len(tok.encode(" proteins", allowed_special=set())) <= 2


def test_frequency_ordering_first_merge_is_the_most_frequent_pair():
    v = train_bpe(["aaab aaab aaab xy"], 300, min_count=1)
    assert v[256] == b"aa"


def test_encoder_reproduces_training_compression(tmp_path):
    """Sanity: tokens/byte on the training corpus is well below 1 and near the training-time optimum."""
    tok, _ = _tok(train_bpe(CORPUS, 700), tmp_path)
    total_bytes = sum(len(s.encode()) for s in CORPUS)
    total_tokens = sum(len(tok.encode(s, allowed_special=set())) for s in CORPUS)
    assert total_tokens / total_bytes < 0.5
