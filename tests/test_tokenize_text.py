"""scripts/tokenize_text.py: .txt / .txt.gz corpora -> valid TOK1 bins (line and blank-line documents), dedup, resume."""
import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import tokenize_text as tt  # noqa: E402
from llm.bpe_trainer import train_bpe, write_tok  # noqa: E402
from llm.data import read_header, read_tokens  # noqa: E402
from llm.tokenizer import BPETokenizer  # noqa: E402

LINES = [
    "Let x be a real number. Then x squared is never negative.",
    "The mitochondria produce ATP through oxidative phosphorylation.",
    "def add(a, b): return a + b  # adds two numbers",
]


@pytest.fixture
def vocab(tmp_path):
    p = tmp_path / "v.tok"
    write_tok(train_bpe(LINES * 30, 300), str(p))
    return p


def _expected(vocab, texts):
    tok = BPETokenizer.from_file(str(vocab))
    return [i for t in texts for i in tok.encode(t, allowed_special=set()) + [tok.eos_id]]


def _run(tmp_path, inputs, vocab, *extra):
    out = tmp_path / "out"
    assert tt.main([*inputs, "--out", str(out), "--vocab", str(vocab), "--workers", "2", "--val-frac", "0", *extra]) == 0
    return out


def test_line_documents_plain_and_gzip_match_direct_encoding(tmp_path, vocab):
    (tmp_path / "a.txt").write_text("\n".join(LINES[:2]) + "\n\n", encoding="utf-8")
    with gzip.open(tmp_path / "b.txt.gz", "wt", encoding="utf-8") as f:
        f.write(LINES[2] + "\n")
    out = _run(tmp_path, [f"math={tmp_path / 'a.txt'}", f"code={tmp_path / 'b.txt.gz'}"], vocab)
    _, arr = read_tokens(str(out / "train.bin"), mmap=False)
    assert arr.tolist() == _expected(vocab, LINES)
    h = read_header(str(out / "train.bin"))
    assert h.total_tokens == len(arr) and h.eos_id == BPETokenizer.from_file(str(vocab)).eos_id
    m = json.loads((out / "manifest.json").read_text())
    assert m["docs"] == 3 and set(m["per_domain"]) == {"math", "code"}
    assert m["per_domain"]["math"]["docs"] == 2 and m["tokenizer_sha256"] == BPETokenizer.file_sha256(str(vocab))


def test_blank_line_documents_stay_whole_and_invalid_bytes_are_repaired(tmp_path, vocab):
    (tmp_path / "qa.txt").write_bytes(b"Question: 2 plus 2?\nAnswer: 4\n\nQuestion: bad \xff byte?\nAnswer: ok\n")
    out = _run(tmp_path, [f"multistep={tmp_path / 'qa.txt'}"], vocab, "--blank-domains", "multistep")
    _, arr = read_tokens(str(out / "train.bin"), mmap=False)
    docs = ["Question: 2 plus 2?\nAnswer: 4", "Question: bad � byte?\nAnswer: ok"]
    assert arr.tolist() == _expected(vocab, docs)


def test_dedup_across_inputs_and_resume_and_count_only(tmp_path, vocab):
    (tmp_path / "a.txt").write_text(LINES[0] + "\n" + LINES[1] + "\n", encoding="utf-8")
    (tmp_path / "b.txt").write_text("  " + LINES[0].upper() + "\n" + LINES[2] + "\n", encoding="utf-8")
    ins = [f"x={tmp_path / 'a.txt'}", f"y={tmp_path / 'b.txt'}"]
    out = _run(tmp_path, ins, vocab, "--dedup")
    m = json.loads((out / "manifest.json").read_text())
    assert m["docs"] == 3 and m["dup_dropped"] == 1
    _, arr = read_tokens(str(out / "train.bin"), mmap=False)
    assert arr.tolist() == _expected(vocab, [LINES[0], LINES[1], LINES[2]])
    stamp = (out / "x.a.train.bin").stat().st_mtime_ns
    _run(tmp_path, ins, vocab, "--dedup")                                   # finished inputs are skipped
    assert (out / "x.a.train.bin").stat().st_mtime_ns == stamp
    assert tt.main([*ins, "--out", str(tmp_path / "cnt"), "--count-only"]) == 0
    c = json.loads((tmp_path / "cnt" / "manifest.json").read_text())
    assert c["chars"] == sum(len(t) for t in LINES) + len("  " + LINES[0].upper()) and "estimated_tokens" in c


def test_validation_split_is_a_deterministic_share_of_documents(tmp_path, vocab):
    docs = [f"document {i} about cells and ATP number {i * 13}" for i in range(400)]
    (tmp_path / "d.txt").write_text("\n".join(docs) + "\n", encoding="utf-8")
    out = tmp_path / "o1"
    args = [f"bio={tmp_path / 'd.txt'}", "--vocab", str(vocab), "--workers", "2", "--val-frac", "0.25"]
    assert tt.main([*args, "--out", str(out)]) == 0
    out2 = tmp_path / "o2"
    assert tt.main([*args, "--out", str(out2)]) == 0
    m1, m2 = (json.loads((o / "manifest.json").read_text()) for o in (out, out2))
    assert m1["val_tokens"] == m2["val_tokens"] > 0 and m1["train_tokens"] + m1["val_tokens"] == len(_expected(vocab, docs))
    assert np.array_equal(read_tokens(str(out / "val.bin"), mmap=False)[1], read_tokens(str(out2 / "val.bin"), mmap=False)[1])
