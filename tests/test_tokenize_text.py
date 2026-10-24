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


def test_marker_documents_drop_the_marker_lines_and_keep_unterminated_tails(tmp_path, vocab):
    raw = ("junk before the first document\n<|document|>\nSource: Cosmopedia (stanford)\n\nFirst document, two lines.\nSecond line.\n"
           "<|endofdocument|>\n<|document|>\nSecond document.\n<|endofdocument|>\n<|document|>\nCut off at the end of the file\n")
    (tmp_path / "phys.txt").write_text(raw, encoding="utf-8")
    assert list(tt.iter_docs(str(tmp_path / "phys.txt"), False, True)) == [
        "Source: Cosmopedia (stanford)\n\nFirst document, two lines.\nSecond line.", "Second document.", "Cut off at the end of the file"]
    out = _run(tmp_path, [f"physics={tmp_path / 'phys.txt'}"], vocab, "--marker-domains", "physics")
    _, arr = read_tokens(str(out / "train.bin"), mmap=False)
    assert arr.tolist() == _expected(vocab, ["Source: Cosmopedia (stanford)\n\nFirst document, two lines.\nSecond line.",
                                             "Second document.", "Cut off at the end of the file"])
    assert json.loads((out / "manifest.json").read_text())["docs"] == 3


def test_spread_reads_evenly_spaced_whole_documents_not_a_prefix(tmp_path, vocab):
    docs = [f"document {i:04d} about cells and ATP, number {i * 13}" for i in range(2000)]
    path = tmp_path / "big.txt"
    path.write_text("\n".join(docs) + "\n", encoding="utf-8")
    got = list(tt.iter_docs(str(path), False, False, spread=4, slice_bytes=6000))
    assert got and all(g in set(docs) for g in got)                      # only whole documents, never a cut one
    ids = sorted(int(g.split()[1]) for g in got)
    assert ids[0] > 100 and ids[-1] < 1900 and (ids[-1] - ids[0]) > 1000     # spread over the file, not rows 0..N
    blocks = [f"Question: q{i}\nAnswer: a{i}" for i in range(1500)]
    (tmp_path / "qa.txt").write_text("\n\n".join(blocks) + "\n", encoding="utf-8")
    got_b = list(tt.iter_docs(str(tmp_path / "qa.txt"), True, False, spread=3, slice_bytes=4000))
    assert got_b and all(g in set(blocks) for g in got_b)
    marked = "".join(f"<|document|>\nSource: x\n\ntext {i}\n<|endofdocument|>\n" for i in range(800))
    (tmp_path / "m.txt").write_text(marked, encoding="utf-8")
    got_m = list(tt.iter_docs(str(tmp_path / "m.txt"), False, True, spread=3, slice_bytes=3000))
    assert got_m and all(g.startswith("Source: x") and g.rsplit("text ", 1)[1].isdigit() for g in got_m)
    out = _run(tmp_path, [f"bio={path}"], vocab, "--max-tokens", "3000", "--spread", "4")
    m = json.loads((out / "manifest.json").read_text())
    assert 0 < m["docs"] < 2000
