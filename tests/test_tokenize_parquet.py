"""scripts/tokenize_parquet.py: Parquet shards -> valid TOK1 bins, per-domain counts, dedup, split, merge."""
import sys
from pathlib import Path

import numpy as np
import pytest

pa = pytest.importorskip("pyarrow")
import pyarrow.parquet as pq  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import tokenize_parquet as tp  # noqa: E402
from llm.bpe_trainer import train_bpe, write_tok  # noqa: E402
from llm.data import read_header, read_tokens  # noqa: E402
from llm.tokenizer import BPETokenizer  # noqa: E402

DOCS = [
    ("a1", "code", "def f(x): return x * 2  # doubles the input"),
    ("a2", "math", "Let $x$ be real. Then $x^2 \\ge 0$ for every real number."),
    ("a3", "math", "Let $x$ be real. Then $x^2 \\ge 0$ for every real number."),
    ("a4", "bio", "The mitochondria produce ATP through oxidative phosphorylation."),
]


@pytest.fixture
def setup(tmp_path):
    vocab = tmp_path / "v.tok"
    write_tok(train_bpe([d[2] for d in DOCS] * 20, 300), str(vocab))
    shard = tmp_path / "s0.parquet"
    pq.write_table(pa.table({"doc_id": [d[0] for d in DOCS], "domain": [d[1] for d in DOCS],
                             "text": [d[2] for d in DOCS], "char_count": [len(d[2]) for d in DOCS]}), shard)
    return tmp_path, vocab, shard


def _job(shard, out, vocab, val_frac=0.0, dedup=True, min_chars=0, count_only=False):
    return (str(shard), str(out), str(vocab), val_frac, dedup, min_chars, count_only, True)


def test_tokenizes_valid_bin_dedups_and_round_trips(setup):
    tmp, vocab, shard = setup
    st = tp.process_file(_job(shard, tmp / "o", vocab))
    assert st["docs"] == 3 and st["dup_dropped"] == 1
    tok = BPETokenizer.from_file(str(vocab))
    kept = [DOCS[0][2], DOCS[1][2], DOCS[3][2]]
    expected = sum(len(tok.encode(t, allowed_special=set())) + 1 for t in kept)
    h = read_header(str(tmp / "o" / "s0.train.bin"))
    assert h.total_tokens == expected == st["train_tokens"] and h.eos_id == tok.eos_id and h.vocab_size == tok.n_vocab
    _, arr = read_tokens(str(tmp / "o" / "s0.train.bin"), mmap=False)
    first = arr[: int(np.argmax(arr == tok.eos_id))].tolist()
    assert tok.decode(first) == DOCS[0][2]
    assert set(st["per_domain"]) == {"code", "math", "bio"} and st["per_domain"]["math"]["docs"] == 1


def test_val_split_by_document_is_total_and_deterministic(setup):
    tmp, vocab, shard = setup
    all_val = tp.process_file(_job(shard, tmp / "v1", vocab, val_frac=1.0, dedup=False))
    assert all_val["train_tokens"] == 0 and all_val["val_tokens"] > 0
    none_val = tp.process_file(_job(shard, tmp / "v0", vocab, val_frac=0.0, dedup=False))
    assert none_val["val_tokens"] == 0 and none_val["train_tokens"] == all_val["val_tokens"]
    again = tp.process_file(_job(shard, tmp / "v2", vocab, val_frac=0.5, dedup=False))
    again2 = tp.process_file(_job(shard, tmp / "v3", vocab, val_frac=0.5, dedup=False))
    assert again["val_tokens"] == again2["val_tokens"]
    assert again["train_tokens"] + again["val_tokens"] == none_val["train_tokens"]


def test_count_only_reads_char_count_without_a_vocab(setup):
    tmp, _, shard = setup
    st = tp.process_file(_job(shard, tmp / "c", None, dedup=False, count_only=True))
    assert st["chars"] == sum(len(d[2]) for d in DOCS) and st["docs"] == 4
    assert st["per_domain"]["math"]["chars"] == 2 * len(DOCS[1][2])


def test_min_chars_filter_and_missing_text_column(setup):
    tmp, vocab, shard = setup
    st = tp.process_file(_job(shard, tmp / "m", vocab, dedup=False, min_chars=60))
    assert st["short_dropped"] >= 1
    bad = tmp / "bad.parquet"
    pq.write_table(pa.table({"name": ["x"]}), bad)
    with pytest.raises(ValueError, match="no 'text' column"):
        tp.process_file(_job(bad, tmp / "b", vocab))


def test_v4_schema_without_text_column(setup):
    import json as _json
    tmp, vocab, _ = setup
    props = [_json.dumps({"code_content": DOCS[0][2]}), _json.dumps({"question": "What is 2+2?", "answer": "4"}),
             _json.dumps({"other": 1}), "not json"]
    shard = tmp / "v4.parquet"
    pq.write_table(pa.table({"entity_id": ["e1", "e2", "e3", "e4"], "domain": ["code", "qa", "x", "y"],
                             "description": ["d1", "d2", "fallback body", "bad props body"], "properties": props}), shard)
    assert tp.v4_text("d", props[0]) == DOCS[0][2]
    assert tp.v4_text("d", props[1]) == "Question: What is 2+2?\n\nAnswer: 4"
    assert tp.v4_text("fallback", props[2]) == "fallback" and tp.v4_text("bad", props[3]) == "bad"
    cnt = tp.process_file(_job(shard, tmp / "c4", None, dedup=False, count_only=True))
    assert cnt["docs"] == 4 and cnt["chars"] == len(DOCS[0][2]) + len("Question: What is 2+2?\n\nAnswer: 4") + len("fallback body") + len("bad props body")
    st = tp.process_file(_job(shard, tmp / "o4", vocab, dedup=False))
    tok = BPETokenizer.from_file(str(vocab))
    texts = [DOCS[0][2], "Question: What is 2+2?\n\nAnswer: 4", "fallback body", "bad props body"]
    assert st["train_tokens"] + st["val_tokens"] == sum(len(tok.encode(t, allowed_special=set())) + 1 for t in texts)


def test_merge_bins_writes_a_valid_file(setup):
    tmp, vocab, shard = setup
    tp.process_file(_job(shard, tmp / "a", vocab, dedup=False))
    p = str(tmp / "a" / "s0.train.bin")
    n = tp.merge_bins([p, p], str(tmp / "merged.bin"), 300, 299, 0)
    h = read_header(str(tmp / "merged.bin"))
    assert n == h.total_tokens == 2 * read_header(p).total_tokens


def test_main_end_to_end_manifest(setup):
    tmp, vocab, shard = setup
    assert tp.main([str(shard), "--out", str(tmp / "e"), "--vocab", str(vocab), "--workers", "1", "--val-frac", "0"]) == 0
    import json
    man = json.loads((tmp / "e" / "manifest.json").read_text())
    assert man["tokenizer_sha256"] == BPETokenizer.file_sha256(str(vocab))
    assert read_header(str(tmp / "e" / "train.bin")).total_tokens == man["train_tokens"]
