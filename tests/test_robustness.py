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


def test_sample_rg_takes_evenly_spaced_row_groups_never_a_prefix(tmp_path, two_vocabs):
    va, _ = two_vocabs
    docs = [f"group{i // 10} document {i} about cells and the number {i}" for i in range(80)]
    p = tmp_path / "s.parquet"
    pq.write_table(pa.table({"doc_id": [f"d{i}" for i in range(80)], "domain": ["bio"] * 80, "text": docs}), p, row_group_size=10)
    assert pq.ParquetFile(p).metadata.num_row_groups == 8
    out = tmp_path / "o"
    assert tp.main([str(p), "--vocab", str(va), "--out", str(out), "--workers", "1", "--val-frac", "0", "--sample-rg", "0.25",
                    "--chunk-mb", "0.0001", "--no-merge"]) == 0
    m = json.loads((out / "manifest.json").read_text())
    assert m["docs"] == 20                                                   # 2 of 8 row groups x 10 documents
    taken = sorted(int(j["job"].split(".rg")[1]) for j in m["shards"])
    assert taken == [2, 6]                                                   # spread over the file, not rows 0-19
    with pytest.raises(SystemExit):
        tp.main([str(p), "--vocab", str(va), "--out", str(out), "--sample-rg", "0"])


def test_max_tokens_stops_reading_a_text_input_early(tmp_path, two_vocabs):
    va, _ = two_vocabs
    (tmp_path / "big.txt").write_text("\n".join(TEXTS * 400) + "\n", encoding="utf-8")
    out = tmp_path / "o"
    base = [f"bio={tmp_path / 'big.txt'}", "--vocab", str(va), "--workers", "1", "--val-frac", "0"]
    assert tt.main([*base, "--out", str(out / "all")]) == 0
    full = json.loads((out / "all" / "manifest.json").read_text())["train_tokens"]
    old = tt.BATCH_DOCS
    tt.BATCH_DOCS = 1000                       # small batches so the cap can bite on a small file
    try:
        assert tt.main([*base, "--out", str(out / "cap"), "--max-tokens", "5000"]) == 0
    finally:
        tt.BATCH_DOCS = old
    capped = json.loads((out / "cap" / "manifest.json").read_text())["train_tokens"]
    assert 5000 <= capped < full
    assert read_header(str(out / "cap" / "train.bin")).total_tokens == capped


def test_max_tokens_does_not_overshoot_by_the_whole_pipeline_depth(tmp_path, two_vocabs):
    va, _ = two_vocabs
    (tmp_path / "big.txt").write_text("\n".join(TEXTS * 3000) + "\n", encoding="utf-8")
    old = tt.BATCH_DOCS
    tt.BATCH_DOCS = 500                               # ~500 documents per batch -> 8 batches in flight with 2 workers
    try:
        assert tt.main([f"bio={tmp_path / 'big.txt'}", "--vocab", str(va), "--workers", "2", "--val-frac", "0", "--out", str(tmp_path / "o"),
                        "--max-tokens", "30000"]) == 0
    finally:
        tt.BATCH_DOCS = old
    got = json.loads((tmp_path / "o" / "manifest.json").read_text())["train_tokens"]
    assert 30000 <= got < 30000 * 1.6                 # the estimate of what is already in flight stops submitting in time
