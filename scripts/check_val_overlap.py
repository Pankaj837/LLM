#!/usr/bin/env python
"""Data checks on the training bins: does validation text also occur in training, and where are the leftover source tags?

    python scripts/check_val_overlap.py --bins bins --domains math physics ... --vocab tokenizers/vocab32k_v2/vocab32k.tok \
        --out results/val_overlap.json

For every domain's ``<name>.val.bin`` and ``<name>.train.bin``:

* ``exact_docs``: validation documents (split at the end-of-text token) that occur whole, token for token, in training.
* ``window_share``: of ``--windows`` ``--n``-token passages spread evenly through the validation file, the share that occurs anywhere
  in training. Passages can be repeated boilerplate, so this measures overlap, not how much a validation loss is inflated.

``tags`` counts each ``--tags`` string in every training file, with and without a leading space (the tokenizer encodes the two
differently, and inside a text the leading-space form is the usual one).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llm.data import read_tokens  # noqa: E402
from llm.tokenizer import BPETokenizer  # noqa: E402
from check_leakage import ngram_hashes  # noqa: E402

TAGS = ["Benchmark source: data/CoT", "GSM_AnsAug", "Expected Answer:"]


def doc_hashes(ids: np.ndarray, eos: int) -> dict:
    """Hash of every document that ends with an end-of-text token -> its length in tokens, and how many times it occurs."""
    ends = np.flatnonzero(ids == eos)
    out = {}
    for a, b in zip(np.r_[0, ends[:-1] + 1], ends):
        if b > a:
            k = hashlib.blake2b(ids[int(a):int(b)].tobytes(), digest_size=12).digest()
            n = out.get(k, (int(b - a), 0))[1]
            out[k] = (int(b - a), n + 1)
    return out


def overlap(val_path: str, train_path: str, n: int, windows: int, chunk: int) -> dict:
    hv, v = read_tokens(val_path)
    ht, t = read_tokens(train_path)
    va = np.asarray(v)
    want = doc_hashes(va, hv.eos_id)
    idx = np.linspace(0, max(0, len(va) - n - 1), windows).astype(int)
    samples = np.unique(np.stack([ngram_hashes(va[i:i + n], n)[0] for i in idx]))
    found, hit, prev = set(), set(), 0
    for s in range(0, len(t), chunk):
        a = np.asarray(t[s:s + chunk + n - 1])
        h = ngram_hashes(a, n)
        j = np.minimum(np.searchsorted(samples, h), len(samples) - 1)
        hit.update(h[samples[j] == h].tolist())
        for c in np.flatnonzero(a[:chunk] == ht.eos_id) + s:
            if c > prev:
                k = hashlib.blake2b(np.asarray(t[prev:int(c)]).tobytes(), digest_size=12).digest()
                if k in want:
                    found.add(k)
            prev = int(c) + 1
    docs = sum(c for _, c in want.values())
    same = sum(want[k][1] for k in found)
    return {"val_docs": docs, "exact_docs": same, "exact_doc_share": same / max(docs, 1),
            "exact_token_share": sum(want[k][0] * want[k][1] for k in found) / max(len(va), 1),
            "windows": len(samples), "window_share": len(hit) / len(samples)}


def count_tag(path: str, tok, text: str, chunk: int) -> int:
    _, t = read_tokens(path)
    total = 0
    for seq in {tuple(tok.encode(x, allowed_special=set())) for x in (text, " " + text)}:
        n, target = len(seq), ngram_hashes(list(seq), len(seq))[0]
        for s in range(0, len(t), chunk):
            total += int(np.count_nonzero(ngram_hashes(np.asarray(t[s:s + chunk + n - 1]), n) == target))
    return total


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bins", required=True)
    ap.add_argument("--domains", nargs="+", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--out", default="val_overlap.json")
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--windows", type=int, default=2000)
    ap.add_argument("--tags", nargs="*", default=TAGS)
    ap.add_argument("--chunk", type=int, default=50_000_000)
    a = ap.parse_args(argv)
    tok = BPETokenizer.from_file(a.vocab)
    res = {"n": a.n, "overlap": {}, "tags": {x: {} for x in a.tags}}
    for d in a.domains:
        val, train = os.path.join(a.bins, f"{d}.val.bin"), os.path.join(a.bins, f"{d}.train.bin")
        r = res["overlap"][d] = overlap(val, train, a.n, a.windows, a.chunk)
        for x in a.tags:
            res["tags"][x][d] = count_tag(train, tok, x, a.chunk)
        print(f"{d}: {r['exact_docs']}/{r['val_docs']} whole documents, {r['window_share']:.1%} of {r['windows']} passages; "
              f"tags {[res['tags'][x][d] for x in a.tags]}", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
