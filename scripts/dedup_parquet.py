#!/usr/bin/env python
"""Drop exact-duplicate documents ACROSS a set of Parquet shards (case/whitespace-insensitive), keeping the first copy.

Works on shards with a ``text`` column (Gokul's 18-column files) and on v4.0 entity files (body taken the same way as
``tokenize_parquet.py``). Writes the cleaned shards under ``--out`` with the same schema, plus ``dedup_report.json``.

    python scripts/dedup_parquet.py "biology/*.parquet" "earth_environment/*.parquet" --out cleaned
    python scripts/dedup_parquet.py "scientific_programming.parquet" --out cleaned --key sha256   # v4.0: dedup on properties.sha256

Memory: 8 bytes per kept document.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dedup_text import doc_key  # noqa: E402
from tokenize_parquet import v4_text  # noqa: E402


def clean_shards(paths, out_dir, key_mode="text", batch=2000) -> dict:
    import pyarrow as pa
    import pyarrow.parquet as pq

    os.makedirs(out_dir, exist_ok=True)
    seen: set = set()
    report = {"files": {}, "total": {"in": 0, "kept": 0, "duplicate": 0}}
    for p in paths:
        pf = pq.ParquetFile(p)
        names = set(pf.schema_arrow.names)
        v4 = "text" not in names and "properties" in names
        if "text" not in names and not v4:
            raise ValueError(f"{p}: no 'text' column and not a v4.0 file (columns: {sorted(names)})")
        st = {"in": 0, "kept": 0, "duplicate": 0}
        out_path = Path(out_dir) / Path(p).name
        with pq.ParquetWriter(out_path, pf.schema_arrow, compression="zstd") as w:
            for rb in pf.iter_batches(batch_size=batch):
                tb = pa.Table.from_batches([rb])
                if v4:
                    desc, props = tb.column("description").to_pylist(), tb.column("properties").to_pylist()
                    if key_mode == "sha256":
                        keys = [json.loads(x).get("sha256") if x else None for x in props]
                        keys = [k if k else doc_key(v4_text(d, x)) for k, d, x in zip(keys, desc, props)]
                    else:
                        keys = [doc_key(v4_text(d, x)) for d, x in zip(desc, props)]
                else:
                    keys = [doc_key(t or "") for t in tb.column("text").to_pylist()]
                keep = []
                for k in keys:
                    if k in seen:
                        keep.append(False)
                    else:
                        seen.add(k)
                        keep.append(True)
                st["in"] += len(keep)
                st["kept"] += sum(keep)
                st["duplicate"] += len(keep) - sum(keep)
                if any(keep):
                    w.write_table(tb.filter(pa.array(keep)))
        report["files"][Path(p).name] = st
        for k, v in st.items():
            report["total"][k] += v
    with open(os.path.join(out_dir, "dedup_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--key", choices=["text", "sha256"], default="text",
                    help="v4.0 files only: 'sha256' uses properties.sha256 when present (fast, exact file identity)")
    a = ap.parse_args(argv)
    paths = []
    for s in a.inputs:
        paths += sorted(glob.glob(s, recursive=True))
    paths = list(dict.fromkeys(paths))
    if not paths:
        ap.error("no Parquet files matched")
    t0 = time.time()
    r = clean_shards(paths, a.out, a.key)
    t = r["total"]
    print(f"in={t['in']:,} kept={t['kept']:,} duplicate={t['duplicate']:,} ({time.time() - t0:.0f}s) -> {a.out}/dedup_report.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
