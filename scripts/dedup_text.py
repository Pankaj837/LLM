#!/usr/bin/env python
"""Clean and de-duplicate text corpora (``.txt`` or ``.txt.gz``) in one streaming pass.

A document is one line (default) or, with ``--records blank``, a block of lines ended by a blank line
(e.g. ``Question:`` / ``Answer:`` pairs, or ``Title:`` / ``Abstract:`` records).

For each input file this writes ``<out>/<name>`` with:
  * exact duplicates removed (case/whitespace-insensitive hash of the whole document, across ALL input files),
  * documents containing more than ``--max-fffd`` U+FFFD replacement characters (broken encoding) removed,
  * documents shorter than ``--min-chars`` removed,
and a ``<out>/dedup_report.json`` with the counts (kept / duplicates / broken / short, per file and in total).

    python scripts/dedup_text.py maths_final_corpus.txt.gz --out cleaned_math
    python scripts/dedup_text.py multistep_calculations_corpus.txt quant_finance_corpus.txt --out cleaned

Memory: about 80 bytes per kept document (a Python set of hashes): ~5M documents ≈ 400 MB, 18M ≈ 1.5 GB. Exact duplicates only;
near-duplicate removal across datasets is a separate, later step.
"""
from __future__ import annotations

import argparse
import glob
import gzip
import hashlib
import json
import os
import sys
import time


def _open_in(path):
    return gzip.open(path, "rt", encoding="utf-8", errors="replace", newline="\n") if path.endswith(".gz") \
        else open(path, "r", encoding="utf-8", errors="replace", newline="\n")


def _open_out(path):
    return gzip.open(path, "wt", encoding="utf-8", newline="\n") if path.endswith(".gz") \
        else open(path, "w", encoding="utf-8", newline="\n")


def doc_key(text: str) -> int:
    return int.from_bytes(hashlib.blake2b(" ".join(text.lower().split()).encode("utf-8", "replace"), digest_size=8).digest(), "little")


def _docs(fi, records):
    if records == "line":
        for line in fi:
            yield line.rstrip("\r\n")
        return
    block = []
    for line in fi:
        line = line.rstrip("\r\n")
        if line.strip():
            block.append(line)
        elif block:
            yield "\n".join(block)
            block = []
    if block:
        yield "\n".join(block)


def clean_files(paths, out_dir, min_chars=0, max_fffd=2, records="line") -> dict:
    os.makedirs(out_dir, exist_ok=True)
    seen: set = set()
    report = {"files": {}, "total": {"in": 0, "kept": 0, "duplicate": 0, "broken": 0, "short": 0}}
    for p in paths:
        st = {"in": 0, "kept": 0, "duplicate": 0, "broken": 0, "short": 0}
        with _open_in(p) as fi, _open_out(os.path.join(out_dir, os.path.basename(p))) as fo:
            for doc in _docs(fi, records):
                if not doc.strip():
                    continue
                st["in"] += 1
                if doc.count("�") > max_fffd:
                    st["broken"] += 1
                    continue
                if len(doc) < min_chars:
                    st["short"] += 1
                    continue
                k = doc_key(doc)
                if k in seen:
                    st["duplicate"] += 1
                    continue
                seen.add(k)
                fo.write(doc + ("\n" if records == "line" else "\n\n"))
                st["kept"] += 1
        report["files"][os.path.basename(p)] = st
        for k, v in st.items():
            report["total"][k] += v
    with open(os.path.join(out_dir, "dedup_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="files or globs")
    ap.add_argument("--records", choices=["line", "blank"], default="line",
                    help="line: one document per line; blank: documents are blocks separated by blank lines")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-chars", type=int, default=50)
    ap.add_argument("--max-fffd", type=int, default=2, help="drop a document with more than this many U+FFFD characters")
    a = ap.parse_args(argv)
    paths = sorted({p for s in a.inputs for p in glob.glob(s)})
    if not paths:
        ap.error("no input files matched")
    t0 = time.time()
    r = clean_files(paths, a.out, a.min_chars, a.max_fffd, a.records)
    t = r["total"]
    print(f"in={t['in']:,} kept={t['kept']:,} duplicate={t['duplicate']:,} broken={t['broken']:,} short={t['short']:,} "
          f"({time.time() - t0:.0f}s) -> {a.out}/dedup_report.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
