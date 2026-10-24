"""Failure-mode tests for the data scripts: stale resumes, half-written outputs, same-named inputs, row-group sizes."""
import json
import sys
from pathlib import Path

import pytest

pa = pytest.importorskip("pyarrow")
import pyarrow.parquet as pq  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import dedup_parquet as dp  # noqa: E402
import tokenize_parquet as tp  # noqa: E402
import tokenize_text as tt  # noqa: E402
from llm.bpe_trainer import train_bpe, write_tok  # noqa: E402
from llm.data import read_header  # noqa: E402
from llm.tokenizer import BPETokenizer  # noqa: E402

TEXTS = [f"document {i} about proteins, cells and the number {i * 17} in tissue {i % 5}" for i in range(120)]


@pytest.fixture
def two_vocabs(tmp_path):
    a, b = tmp_path / "a.tok", tmp_path / "b.tok"
    write_tok(train_bpe(TEXTS * 3, 300), str(a))
    write_tok(train_bpe(TEXTS * 3, 340, split_digits=True), str(b))          # a different vocabulary (different ids)
    assert BPETokenizer.file_sha256(str(a)) != BPETokenizer.file_sha256(str(b))
    return a, b


def _parquet(tmp_path):
    p = tmp_path / "in.parquet"
    pq.write_table(pa.table({"doc_id": [f"d{i}" for i in range(len(TEXTS))], "domain": ["bio"] * len(TEXTS), "text": TEXTS}), p)
    return p


def test_changing_the_vocabulary_never_reuses_old_parquet_shards(tmp_path, two_vocabs):
    va, vb = two_vocabs
    src = _parquet(tmp_path)
    out = tmp_path / "bins"
    args = [str(src), "--out", str(out), "--workers", "1", "--val-frac", "0"]
    assert tp.main([*args, "--vocab", str(va)]) == 0
    assert read_header(str(out / "train.bin")).vocab_size == BPETokenizer.from_file(str(va)).n_vocab
    assert tp.main([*args, "--vocab", str(vb)]) == 0                      # same folder, other tokenizer
    h = read_header(str(out / "train.bin"))
    assert h.vocab_size == BPETokenizer.from_file(str(vb)).n_vocab
    assert json.loads((out / "manifest.json").read_text())["tokenizer_sha256"] == BPETokenizer.file_sha256(str(vb))


def test_changing_the_vocabulary_never_reuses_old_text_files(tmp_path, two_vocabs):
    va, vb = two_vocabs
    (tmp_path / "c.txt").write_text("\n".join(TEXTS) + "\n", encoding="utf-8")
    out = tmp_path / "bins"
    args = [f"bio={tmp_path / 'c.txt'}", "--out", str(out), "--workers", "1", "--val-frac", "0"]
    assert tt.main([*args, "--vocab", str(va)]) == 0
    assert tt.main([*args, "--vocab", str(vb)]) == 0
    assert read_header(str(out / "train.bin")).vocab_size == BPETokenizer.from_file(str(vb)).n_vocab


def test_a_failed_job_leaves_no_bin_that_looks_finished(tmp_path, two_vocabs, monkeypatch):
    va, _ = two_vocabs
    src = _parquet(tmp_path)
    out = tmp_path / "bins"

    class Boom(BPETokenizer):
        def encode(self, *a, **k):
            raise RuntimeError("worker died")

    monkeypatch.setattr(tp, "BPETokenizer", type("T", (), {"from_file": staticmethod(lambda p: Boom.from_file(p)),
                                                           "file_sha256": staticmethod(BPETokenizer.file_sha256)}))
    with pytest.raises(RuntimeError):
        tp.process_file((str(src), str(out), str(va), 0.0, False, 0, False, False))
    assert not list(out.glob("*.bin")) and not list(out.glob("*.tmp")) and not list(out.glob("*.stats.json"))


def test_dedup_parquet_keeps_same_named_inputs_apart_and_writes_real_row_groups(tmp_path):
    for sub in ("x", "y"):
        (tmp_path / sub).mkdir()
        pq.write_table(pa.table({"text": [f"{sub} document number {i}" for i in range(3000)]}), tmp_path / sub / "part-0.parquet")
    r = dp.clean_shards([str(tmp_path / "x" / "part-0.parquet"), str(tmp_path / "y" / "part-0.parquet")], str(tmp_path / "o"))
    assert len(r["files"]) == 2 and r["total"]["kept"] == 6000
    outs = sorted((tmp_path / "o").glob("*.parquet"))
    assert len(outs) == 2
    assert all(pq.ParquetFile(p).metadata.num_row_groups == 1 for p in outs)       # 3000 rows, not one group per 2000-row batch
