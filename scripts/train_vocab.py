#!/usr/bin/env python
"""Train a byte-level BPE vocabulary (``.tok``) from JSONL samples (one ``{"t": "<document text>"}`` per line).

    python scripts/train_vocab.py samples/*.jsonl --vocab-size 32000 --out tokenizers/vocab32k_v2/vocab32k.tok

Uses the repo's pure-Python trainer (``llm.bpe_trainer``): about 4-5 minutes for a 32K vocabulary on a ~220 MB mixed
sample (measured). Its output is format-compatible with, and was checked against, the C++ encoder
(see ``tokenizers/vocab32k_v2/README.md``). Streams the files, so memory stays low.
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


# Named special tokens, in id order after the merges. The first is the document boundary the data pipeline appends; the
# next eight are the conventional markers for chat / tool-use / structured-output fine-tuning later; the rest are
# reserved so that those stages never need to resize the embedding of a frozen vocabulary.
SPECIAL_NAMES = ["<|endoftext|>", "<|pad|>", "<|bos|>", "<|system|>", "<|user|>", "<|assistant|>", "<|end_of_turn|>",
                 "<|tool_call|>", "<|tool_result|>"] + [f"<|reserved_{i}|>" for i in range(9, 64)]


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
    ap.add_argument("--reserved-specials", type=int, default=1,
                    help="number of special tokens (1 = only <|endoftext|>; 64 = the named + reserved set above)")
    a = ap.parse_args(argv)
    if not 1 <= a.reserved_specials <= len(SPECIAL_NAMES):
        ap.error(f"--reserved-specials must be between 1 and {len(SPECIAL_NAMES)}")
    paths = sorted({p for s in a.samples for p in glob.glob(s)})
    if not paths:
        ap.error("no sample files matched")
    t0 = time.time()
    vocab = train_bpe(iter_texts(paths), a.vocab_size, reserved_specials=a.reserved_specials, split_digits=a.split_digits)
    n = write_tok(vocab, a.out, specials=SPECIAL_NAMES[:a.reserved_specials])
    print(f"files={len(paths)} merges={len(vocab)} n_vocab={n} {time.time() - t0:.0f}s sha256={BPETokenizer.file_sha256(a.out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
