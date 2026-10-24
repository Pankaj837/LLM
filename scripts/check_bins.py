#!/usr/bin/env python
"""Integrity check for TOK1 token files before any training run.

    python scripts/check_bins.py runs/bins/train.bin runs/bins/val.bin runs/bins/*.val.bin --vocab tokenizers/vocab32k_v1/vocab32k.tok

For every file: the header is valid and the file size matches ``total_tokens`` (no truncation); every id is below
the vocabulary size; the file ends on an EOS; documents (= EOS count) and mean document length are reported. Across
files: identical vocabulary size and EOS id, and (with ``--vocab``) both equal the frozen tokenizer's. Exit code 1 on
any problem, so it can gate a launch script.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from llm.data import read_tokens  # noqa: E402
from llm.tokenizer import BPETokenizer  # noqa: E402

CHUNK = 1 << 26


def check_file(path: str) -> dict:
    h, toks = read_tokens(path, mmap=True)
    info = {"path": path, "tokens": int(h.total_tokens), "vocab": h.vocab_size, "eos": h.eos_id, "problems": []}
    eos = docs = 0
    mx = 0
    for i in range(0, len(toks), CHUNK):
        c = np.asarray(toks[i:i + CHUNK])
        mx = max(mx, int(c.max())) if len(c) else mx
        eos += int((c == h.eos_id).sum())
    docs = eos
    if len(toks) and mx >= h.vocab_size:
        info["problems"].append(f"id {mx} >= vocab_size {h.vocab_size}")
    if len(toks) and int(toks[-1]) != h.eos_id:
        info["problems"].append("file does not end with EOS (cut mid-document?)")
    if len(toks) == 0:
        info["problems"].append("no tokens")
    info.update(max_id=mx, docs=docs, mean_doc=(len(toks) / docs if docs else float("nan")))
    return info


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bins", nargs="+", help="TOK1 files or globs")
    ap.add_argument("--vocab", help="the frozen .tok: vocabulary size and EOS id must match")
    a = ap.parse_args(argv)
    paths = sorted({p for s in a.bins for p in glob.glob(s)})
    if not paths:
        ap.error("no files matched")
    ref = None
    if a.vocab:
        t = BPETokenizer.from_file(a.vocab)
        ref = (t.n_vocab, t.eos_id)
        print(f"frozen tokenizer: n_vocab={ref[0]} eos_id={ref[1]} sha256={BPETokenizer.file_sha256(a.vocab)}")
    bad = 0
    first = None
    print(f"{'file':<48}{'tokens':>16}{'docs':>12}{'mean doc':>10}  status")
    for p in paths:
        try:
            r = check_file(p)
        except ValueError as e:
            print(f"{os.path.basename(p):<48}{'':>16}{'':>12}{'':>10}  FAIL {e}")
            bad += 1
            continue
        key = (r["vocab"], r["eos"])
        first = first or key
        if key != first:
            r["problems"].append(f"vocab/eos {key} differ from the first file {first}")
        if ref and key != ref:
            r["problems"].append(f"vocab/eos {key} differ from the frozen tokenizer {ref}")
        status = "ok" if not r["problems"] else "FAIL " + "; ".join(r["problems"])
        bad += bool(r["problems"])
        print(f"{os.path.basename(p):<48}{r['tokens']:>16,}{r['docs']:>12,}{r['mean_doc']:>10.0f}  {status}")
    print("RESULT:", "PASS" if not bad else f"FAIL ({bad} file(s))")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
