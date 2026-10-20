"""scripts/dedup_text.py: exact dedup (across files), broken-encoding and short-document filters, .gz in and out."""
import gzip
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import dedup_text as dd  # noqa: E402
import dedup_parquet  # noqa: E402,F401

A = "Let x be a real number. Then x squared is never negative."


def test_dedup_filters_and_reports(tmp_path):
    f1, f2 = tmp_path / "a.txt", tmp_path / "b.txt.gz"
    f1.write_text("\n".join([A, "  " + A.upper() + "  ", "short", "bad ���� text " * 5,
                             "A different long enough document about mitochondria and ATP.", ""]), encoding="utf-8")
    with gzip.open(f2, "wt", encoding="utf-8") as f:
        f.write(A + "\nAnother unique document that is long enough to keep around.\n")
    out = tmp_path / "out"
    assert dd.main([str(f1), str(f2), "--out", str(out), "--min-chars", "20"]) == 0
    rep = json.loads((out / "dedup_report.json").read_text())
    assert rep["files"]["a.txt"] == {"in": 5, "kept": 2, "duplicate": 1, "broken": 1, "short": 1}
    assert rep["files"]["b.txt.gz"] == {"in": 2, "kept": 1, "duplicate": 1, "broken": 0, "short": 0}   # A seen in a.txt
    assert (out / "a.txt").read_text(encoding="utf-8").splitlines() == [A, "A different long enough document about mitochondria and ATP."]
    with gzip.open(out / "b.txt.gz", "rt", encoding="utf-8") as f:
        assert f.read().splitlines() == ["Another unique document that is long enough to keep around."]


def test_dedup_parquet_across_shards_text_and_v4(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    import dedup_parquet as dp
    pq.write_table(pa.table({"doc_id": ["1", "2"], "text": [A, "Second distinct document here."]}), tmp_path / "s1.parquet")
    pq.write_table(pa.table({"doc_id": ["3", "4"], "text": ["  " + A.upper(), "Third distinct document."]}), tmp_path / "s2.parquet")
    r = dp.clean_shards([str(tmp_path / "s1.parquet"), str(tmp_path / "s2.parquet")], str(tmp_path / "o"))
    assert r["total"] == {"in": 4, "kept": 3, "duplicate": 1}
    assert pq.read_table(tmp_path / "o" / "s2.parquet").column("doc_id").to_pylist() == ["4"]
    props = [json.dumps({"code_content": "x = 1", "sha256": "aa"}), json.dumps({"code_content": "y = 2", "sha256": "aa"}),
             json.dumps({"code_content": "z = 3", "sha256": "bb"})]
    pq.write_table(pa.table({"description": ["d"] * 3, "properties": props}), tmp_path / "v4.parquet")
    r2 = dp.clean_shards([str(tmp_path / "v4.parquet")], str(tmp_path / "o4"), key_mode="sha256")
    assert r2["total"] == {"in": 3, "kept": 2, "duplicate": 1}
