"""scripts/train_vocab.py: JSONL samples -> a valid .tok that round-trips arbitrary text."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import train_vocab as tv  # noqa: E402
from llm.tokenizer import BPETokenizer  # noqa: E402


def test_trains_a_valid_vocab_from_jsonl(tmp_path):
    docs = ["def add(a, b): return a + b", "The mitochondria produce ATP.", r"Let $x^2 \ge 0$ for real x."] * 40
    (tmp_path / "a.jsonl").write_text("\n".join(json.dumps({"t": d}) for d in docs), encoding="utf-8")
    out = tmp_path / "v.tok"
    assert tv.main([str(tmp_path / "*.jsonl"), "--vocab-size", "300", "--out", str(out)]) == 0
    tok = BPETokenizer.from_file(str(out))
    assert tok.n_vocab == 300 and tok.eos_id == 299
    for s in docs[:3] + ["héllo 日本語 \U0001f600", ""]:
        assert tok.decode(tok.encode(s, allowed_special=set())) == s


def test_split_digits_vocab_encodes_every_number_digit_by_digit(tmp_path):
    import re

    from llm.bpe_trainer import train_bpe
    docs = ["The answer is 12345 and 2026 plus 3.14159 gives 2029.14159 for H2O at 37 degrees. " * 3, "x = 99 + 100"] * 40
    plain = train_bpe(docs, 320)
    split = train_bpe(docs, 320, split_digits=True)
    assert any(re.search(rb"[0-9]{2}", t) for t in plain)                       # without the option digit chunks are learned
    assert not any(re.search(rb"[0-9]{2}", t) for t in split)                   # with it, never
    out = tmp_path / "s.tok"
    from llm.bpe_trainer import write_tok
    write_tok(split, str(out))
    tok = BPETokenizer.from_file(str(out))
    ids = tok.encode("2026 12345", allowed_special=set())
    assert [tok.decode([i]).strip() for i in ids if tok.decode([i]).strip()] == list("202612345")
    assert tok.decode(ids) == "2026 12345"


def test_reserved_special_tokens_are_named_in_order_and_never_produced_from_text(tmp_path):
    import json

    from llm.tokenizer import BPETokenizer
    docs = ["hello world, number 12345 and 2026. " * 5, "def f(x): return x + 1\n" * 5] * 30
    (tmp_path / "s.jsonl").write_text("\n".join(json.dumps({"t": d}) for d in docs), encoding="utf-8")
    out = tmp_path / "v.tok"
    assert tv.main([str(tmp_path / "s.jsonl"), "--vocab-size", "400", "--reserved-specials", "64", "--out", str(out)]) == 0
    t = BPETokenizer.from_file(str(out))
    assert len(t.special_to_id) == 64 and t.eos_id == min(t.special_to_id.values())          # EOS is the first special
    assert sorted(t.special_to_id.values()) == list(range(t.eos_id, t.eos_id + 64)) and t.n_vocab == t.eos_id + 64
    assert t.id_to_special[t.eos_id] == "<|endoftext|>" and t.id_to_special[t.n_vocab - 1] == "<|reserved_63|>"
    assert {"<|pad|>", "<|user|>", "<|assistant|>", "<|tool_call|>"} <= set(t.special_to_id)
    ids = t.encode("a <|user|> b <|endoftext|>", allowed_special=set())
    assert not any(i in t.id_to_special for i in ids)                                      # text cannot forge a special
    assert t.decode(ids) == "a <|user|> b <|endoftext|>"
