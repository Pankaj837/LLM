"""C5 Tokenizer: byte-level BPE (Python port of the C++ tokenizer)."""
import random

import pytest

from llm.tokenizer import BPETokenizer, pretokenize

UNICODE = ["café résumé naïve", "日本語のテキスト", "emoji \U0001f600\U0001f680 ok",
           "mixed 日本English99 ü", "Ｆｕｌｌｗｉｄｔｈ"]
WHITESPACE = ["", " ", "   ", "\n", "\n\n\n", "\t", "a  b", "a\n\nb", "trailing   ", "   leading", "tab\tsep", "\r\nwin\r\n"]
PUNCT = ["Hello, world!", "don't they've we'll I'm you're he'd it's", "'standalone 'tis", "a.b,c;d:e", "!!!???...---***", "1234567.89", "x=1+2*3"]


@pytest.fixture(scope="module")
def tok(vocab_path):
    return BPETokenizer(vocab_path=vocab_path, vocab_size=1722)


@pytest.mark.parametrize("text", UNICODE + WHITESPACE + PUNCT)
def test_roundtrip_is_lossless(tok, text):
    assert tok.decode(tok.encode(text, allowed_special=set())) == text


def test_roundtrip_random_fuzz(tok):
    rng = random.Random(0)
    alphabet = "abc XYZ 019 .,!?'\n\té日\U0001f600 "
    for _ in range(300):
        s = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 80)))
        assert tok.decode(tok.encode(s, allowed_special=set())) == s, repr(s)


def test_pretokenize_is_lossless_concatenation():
    for s in UNICODE + WHITESPACE + PUNCT:
        assert b"".join(pretokenize(s)) == s.encode("utf-8")


def test_encode_is_deterministic_and_cache_safe(tok):
    a = tok.encode("Grind the beans right before you brew.")
    b = tok.encode("Grind the beans right before you brew.")
    assert a == b
    a.append(-1)                                    # mutating a result must not poison the cache
    assert tok.encode("Grind the beans right before you brew.") == b


def test_special_token_encodes_to_single_id(tok):
    ids = tok.encode("hi<|endoftext|>there")
    assert ids.count(tok.eot_token_id) == 1


def test_lone_surrogate_raises_a_clear_error(tok):
    with pytest.raises(ValueError):                 # UnicodeEncodeError is a ValueError
        tok.encode("a\ud800b", allowed_special=set())


def test_unknown_id_decodes_without_crashing(tok):
    assert "<unk:99999>" in tok.decode([99999])


def test_default_byte_vocab_when_no_file():
    t = BPETokenizer()
    assert len(t.token_to_id) == 256 and t.eot_token_id == 256
    assert t.decode(t.encode("plain text")) == "plain text"


def test_load_rejects_empty_vocab_file(tmp_path):
    p = tmp_path / "empty.tok"; p.write_text("")
    with pytest.raises(ValueError):
        BPETokenizer(vocab_path=str(p))


def test_save_load_roundtrip(tok, tmp_path):
    p = str(tmp_path / "v.tok")
    tok.save(p)
    t2 = BPETokenizer(vocab_path=p)
    for s in ["Grind the beans.", "don't stop", "日本"]:
        assert t2.encode(s, allowed_special=set()) == tok.encode(s, allowed_special=set())


def test_merges_actually_compress(tok):
    text = "How do I make better coffee at home?"
    assert len(tok.encode(text, allowed_special=set())) < len(text.encode()) * 0.6


@pytest.mark.finding("F-12")
def test_tokenizer_reports_id_space_size(tok):
    """Pipeline needs to check tokenizer ids fit the embedding table."""
    assert tok.n_vocab == 1722


# ------------------------------------------------------------------ one tokenizer for the whole stack
def test_from_file_reports_vocab_and_eos(vocab_path):
    t = BPETokenizer.from_file(vocab_path)
    assert (t.n_vocab, t.vocab_size, t.eos_id) == (1722, 1722, 1721)


def test_from_file_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        BPETokenizer.from_file(str(tmp_path / "nope.tok"))


def test_file_sha256_is_stable_and_content_sensitive(vocab_path, tmp_path):
    a = BPETokenizer.file_sha256(vocab_path)
    assert a == BPETokenizer.file_sha256(vocab_path) and len(a) == 64
    other = tmp_path / "other.tok"
    other.write_bytes(open(vocab_path, "rb").read() + b"\n")
    assert BPETokenizer.file_sha256(str(other)) != a


def test_reward_scripts_use_the_shared_tokenizer():
    import llm.reward.train_reward as tr
    assert tr.BPETokenizer is BPETokenizer


def test_lm_and_crm_id_spaces_are_not_interchangeable():
    """The LM (vocab 8000) and the reward model (vocab ~1.7k) must hand off via TEXT, never ids."""
    from llm.model_config import ModelConfig
    assert ModelConfig().vocab_size == 8000
