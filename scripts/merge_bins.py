#!/usr/bin/env python
"""Concatenate TOK1 files (e.g. all the per-chunk bins of one domain) into one valid TOK1 file.

    python scripts/merge_bins.py "runs/bins/code*.train.bin" --out bins/code.train.bin
    python scripts/merge_bins.py "runs/bins/code*.val.bin"   --out bins/code.val.bin --split val

``tokenize_parquet.py`` / ``tokenize_text.py --no-merge`` write one train and one validation file per job (a big input
is split into several jobs). The weighted mix (``llm.train --train-mix``) and the per-domain evaluation want ONE file per
domain, so merge each domain's files with this script. Inputs are taken in sorted name order; all must share the same
vocabulary size and EOS id (checked, so files from different tokenizers can never be glued together).
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tokenize_parquet as tp  # noqa: E402
from llm.data import SPLIT_TRAIN, SPLIT_VAL, read_header  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="TOK1 files or globs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", choices=["train", "val"], default="train")
    a = ap.parse_args(argv)
    paths = sorted({p for s in a.inputs for p in glob.glob(s)})
    paths = [p for p in paths if os.path.abspath(p) != os.path.abspath(a.out)]
    if not paths:
        ap.error("no input files matched")
    heads = [read_header(p) for p in paths]                    # also validates every header and file size
    ref = (heads[0].vocab_size, heads[0].eos_id)
    for p, h in zip(paths, heads):
        if (h.vocab_size, h.eos_id) != ref:
            ap.error(f"{p}: vocab_size/eos_id {(h.vocab_size, h.eos_id)} differ from {paths[0]} {ref}")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    n = tp.merge_bins(paths, a.out, ref[0], ref[1], SPLIT_TRAIN if a.split == "train" else SPLIT_VAL)
    assert n == sum(h.total_tokens for h in heads)
    print(f"merged {len(paths)} files -> {a.out}: {n:,} tokens")
    return 0


if __name__ == "__main__":
    sys.exit(main())
