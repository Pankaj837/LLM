#!/usr/bin/env python
"""Integrity check for TOK1 token files before any training run.

    python scripts/check_bins.py runs/bins/train.bin runs/bins/val.bin runs/bins/*.val.bin --vocab tokenizers/vocab32k_v2/vocab32k.tok

For every file: the header is valid and the file size matches ``total_tokens`` (no truncation); every id is below
the vocabulary size; the file ends on an EOS; documents (= EOS count) and mean document length are reported. Across
files: identical vocabulary size and EOS id, and (with ``--vocab``) both equal the frozen tokenizer's. With ``--vocab`` it also
decodes and re-encodes 20 chunks spread through each file: the ids must come back unchanged, otherwise the file was tokenized
with a DIFFERENT vocabulary (the header cannot tell two same-sized vocabularies apart). Exit code 1 on
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

PROBE_CHUNKS = 20        # chunks decoded and re-encoded per file, spread evenly through it
PROBE_TOKENS = 1500      # tokens per chunk (starting at a document start; short documents are probed together)
PROBE_MAX_BAD = 0.2      # more than this share of mismatching documents => another vocabulary
PROBE_TAIL = 4           # tokens ignored at an artificial cut (a cut can split a word or a multi-byte character)


def probe_roundtrip(toks, eos_id: int, tokenizer, n: int = PROBE_CHUNKS) -> tuple:
    """(documents that do not survive decode -> encode, documents probed).

    Text tokenized with one vocabulary and decoded with another is not the original text; re-encoding that text does not give the
    ids back, for any text (plain ASCII included). Gokul's bio / earth files (25 Oct) were tokenized with ``vocab32k_digits_v2`` while
    the frozen one is ``vocab32k_v2``: same size, same EOS id, same file name ``vocab32k.tok``. 0 of 300 documents round-tripped.
    A document cut by the end of the chunk is compared up to its last few tokens only.
    """
    bad = total = 0
    for i in range(n):
        w = np.asarray(toks[i * len(toks) // n: i * len(toks) // n + 10 * PROBE_TOKENS])
        ends = np.flatnonzero(w == eos_id)
        if len(ends) < 1:
            continue
        chunk = w[int(ends[0]) + 1: int(ends[0]) + 1 + PROBE_TOKENS].tolist()
        docs, cur, complete = [], [], []
        for t in chunk:
            if t == eos_id:
                if cur:
                    docs.append(cur)
                    complete.append(True)
                cur = []
            else:
                cur.append(t)
        if cur:
            docs.append(cur)
            complete.append(False)                            # cut by the end of the chunk
        for ids, whole in zip(docs, complete):
            if len(ids) < 3:
                continue
            text = tokenizer.decode(ids)
            back = tokenizer.encode(text, allowed_special=set())
            keep = len(ids) if whole else len(ids) - PROBE_TAIL
            total += 1
            bad += (back[:keep] != ids[:keep]) or (whole and len(back) != len(ids))
    return bad, total


def check_file(path: str, tokenizer=None) -> dict:
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
    elif tokenizer is not None and not info["problems"]:
        bad, total = probe_roundtrip(toks, h.eos_id, tokenizer)
        info["probe"] = (bad, total)
        if total >= 5 and bad / total > PROBE_MAX_BAD:
            info["problems"].append(f"{bad} of {total} sampled documents do not survive decode -> encode with this vocabulary: tokenized with a "
                                    "DIFFERENT vocabulary (e.g. vocab32k_digits_v2 instead of the frozen vocab32k_v2)?")
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
    t = None
    if a.vocab:
        t = BPETokenizer.from_file(a.vocab)
        ref = (t.n_vocab, t.eos_id)
        print(f"frozen tokenizer: n_vocab={ref[0]} eos_id={ref[1]} sha256={BPETokenizer.file_sha256(a.vocab)}")
    bad = 0
    first = None
    print(f"{'file':<48}{'tokens':>16}{'docs':>12}{'mean doc':>10}  status")
    for p in paths:
        try:
            r = check_file(p, t)
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
