#!/usr/bin/env python
"""Train a byte-level BPE vocabulary (``.tok``) from JSONL samples (one ``{"t": "<document text>"}`` per line).

    python scripts/train_vocab.py samples/*.jsonl --vocab-size 32000 --out tokenizers/vocab32k_v1/vocab32k.tok

Uses the repo's pure-Python trainer (``llm.bpe_trainer``): about 4-5 minutes for a 32K vocabulary on a ~220 MB mixed
sample (measured). Its output is format-compatible with, and was checked against, the C++ encoder
(see ``tokenizers/vocab32k_v1/README.md``). Streams the files, so memory stays low.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from llm.bpe_trainer import train_bpe, write_tok  # noqa: E402
from llm.tokenizer import BPETokenizer  # noqa: E402


def iter_texts(paths):
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)["t"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("samples", nargs="+", help="JSONL files or globs")
    ap.add_argument("--vocab-size", type=int, default=32000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split-digits", action="store_true",
                    help="no token contains two adjacent digits: numbers are encoded digit by digit")
    a = ap.parse_args(argv)
    paths = sorted({p for s in a.samples for p in glob.glob(s)})
    if not paths:
        ap.error("no sample files matched")
    t0 = time.time()
    vocab = train_bpe(iter_texts(paths), a.vocab_size, split_digits=a.split_digits)
    n = write_tok(vocab, a.out)
    print(f"files={len(paths)} merges={len(vocab)} n_vocab={n} {time.time() - t0:.0f}s sha256={BPETokenizer.file_sha256(a.out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
