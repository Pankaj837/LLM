#!/usr/bin/env python
"""Verify a ``.tok`` vocabulary before freezing it: structure, lossless round-trip on held-out documents of every
domain, bytes/token per domain (and against cl100k if ``tiktoken`` is installed), and exact parity with the C++ encoder.

    python scripts/verify_vocab.py tokenizers/vocab32k_v2/vocab32k.tok --holdout "holdout/*.jsonl" \
        --cli native/tokenizer/build/encode_cli.exe

``--holdout`` files are JSONL (``{"t": text}``) that were NOT used to train the vocabulary. Exit code 1 on any failure.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from llm.tokenizer import BPETokenizer  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("vocab")
    ap.add_argument("--holdout", required=True)
    ap.add_argument("--cli", help="C++ encode_cli binary (native/tokenizer/tests/encode_cli.cpp); skips parity if absent")
    ap.add_argument("--bytes-per-domain", type=int, default=600_000)
    ap.add_argument("--parity-lines", type=int, default=3000)
    a = ap.parse_args(argv)
    ok = True
    tok = BPETokenizer.from_file(a.vocab)
    print(f"n_vocab={tok.n_vocab} eos_id={tok.eos_id} sha256={BPETokenizer.file_sha256(a.vocab)}")
    probe = "héllo 日本語 \U0001f600 \x00\x01 \t\n"
    ok &= tok.decode(tok.encode(probe, allowed_special=set())) == probe
    ok &= tok.encode("<|endoftext|>", allowed_special=set()) != [tok.eos_id]          # user text cannot forge EOS
    try:
        import tiktoken
        cl = tiktoken.get_encoding("cl100k_base")
    except Exception:
        cl = None
    lines = []
    print(f"{'domain':<14}{'bytes':>10}{'tokens':>10}{'B/tok':>8}" + ("  cl100k B/tok" if cl else ""))
    for f in sorted(glob.glob(a.holdout)):
        docs = [json.loads(x)["t"] for x in open(f, encoding="utf-8") if x.strip()]
        random.Random(f"2:{os.path.basename(f)}").shuffle(docs)       # per-domain seed: a domain's numbers never depend on the others
        b = t = c = 0
        for i, s in enumerate(docs):
            if b > a.bytes_per_domain:
                break
            ids = tok.encode(s, allowed_special=set())
            if tok.decode(ids) != s:
                print("ROUND-TRIP FAILURE in", f)
                ok = False
            b += len(s.encode())
            t += len(ids)
            c += len(cl.encode(s, disallowed_special=())) if cl else 0
            if i < 150 and len(lines) < a.parity_lines:
                lines += [x for x in s.split("\n") if x and "\r" not in x][:20]
        name = os.path.basename(f).rsplit(".", 1)[0]
        print(f"{name:<14}{b:>10}{t:>10}{b / max(t, 1):>8.2f}" + (f"{b / max(c, 1):>14.2f}" if cl else ""))
    if a.cli:
        lines = lines[:a.parity_lines] + ["café résumé 日本語 \U0001f600", "  leading and trailing  ", "tab\there"]
        out = subprocess.run([os.path.abspath(a.cli), os.path.abspath(a.vocab)], input="\n".join(lines), capture_output=True, text=True,
                             encoding="utf-8", timeout=900, check=True)
        res = out.stdout.strip("\n").split("\n")
        bad = sum(" ".join(map(str, tok.encode(x, allowed_special=set()))) != y.strip() for x, y in zip(lines, res))
        ok &= len(res) == len(lines) and bad == 0
        print(f"C++ parity: {len(lines) - bad}/{len(lines)} lines identical")
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
