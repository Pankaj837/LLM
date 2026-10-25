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

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tokenize_parquet as tp  # noqa: E402
from llm.data import SPLIT_TRAIN, SPLIT_VAL, read_header, read_tokens, write_bin  # noqa: E402


def capped_mix(paths, out: str, cap: int, split: int = SPLIT_VAL) -> dict:
    """One file from the first ``cap`` tokens of every input, each piece cut back to its last EOS (whole documents only).

    Used for the single validation file ``llm.train --val-bin`` needs: every domain contributes the same amount, so the
    running validation loss is not the loss of whichever domain happens to come first. Returns tokens taken per input.
    """
    pieces, taken, ref = [], {}, None
    for p in paths:
        h, toks = read_tokens(p)
        ref = ref or (h.vocab_size, h.eos_id)
        if (h.vocab_size, h.eos_id) != ref:
            raise ValueError(f"{p}: vocab_size/eos_id {(h.vocab_size, h.eos_id)} differ from {ref}")
        if h.total_tokens == 0:
            continue
        if h.total_tokens <= cap:
            piece = np.asarray(toks)
        else:
            head = np.asarray(toks[:cap])
            ends = np.flatnonzero(head == h.eos_id)
            if len(ends):
                piece = head[: ends[-1] + 1]
            else:                                              # one document longer than the cap: take it whole
                more = np.flatnonzero(np.asarray(toks[cap:cap + 4 * cap]) == h.eos_id)
                if not len(more):
                    continue
                piece = np.asarray(toks[: cap + more[0] + 1])
        pieces.append(piece.astype(np.int64))
        taken[p] = int(len(piece))
    if not pieces:
        raise ValueError("no input contributed any tokens")
    write_bin(out, np.concatenate(pieces), ref[0], ref[1], split)
    return taken


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="TOK1 files or globs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", choices=["train", "val"], default="train")
    ap.add_argument("--cap-per-input", type=int, default=0,
                    help="take only the first N tokens of each input (cut at a document end): a balanced validation file")
    a = ap.parse_args(argv)
    paths = sorted({p for s in a.inputs for p in glob.glob(s)})
    paths = [p for p in paths if os.path.abspath(p) != os.path.abspath(a.out)]
    if not paths:
        ap.error("no input files matched")
    if a.cap_per_input:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        taken = capped_mix(paths, a.out, a.cap_per_input, SPLIT_TRAIN if a.split == "train" else SPLIT_VAL)
        print(f"merged {len(taken)} files (at most {a.cap_per_input:,} tokens each) -> {a.out}: {sum(taken.values()):,} tokens")
        return 0
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
