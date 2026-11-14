#!/usr/bin/env python
"""Train/evaluation leakage check: is any evaluation question present in the training ``.bin`` files?

    python scripts/check_leakage.py --evals evals --vocab tokenizers/vocab32k_v2/vocab32k.tok \
        --bins "bins/*.train.bin" --out results/leakage

Every question is tokenized with the frozen tokenizer and cut into overlapping n-grams of ``--n`` tokens (13 by default,
the usual contamination window; a question shorter than that is checked as one n-gram of its own length, down to
``--min-n``). Each training file is streamed in chunks (memory-mapped, overlap n−1 tokens so nothing is missed at a chunk
edge), every n-gram of the chunk is hashed (64-bit polynomial hash) and looked up among the question hashes.

A question is ``leaked`` when at least ``--threshold`` (default 0.2) of its n-grams (and at least ``--min-hits``, default 2,
so one shared stock phrase such as "Which of the following statements is true?" cannot flag a short question), or at least
``--min-matched`` (default 10) of them, occur in the training data: copies on forums often change punctuation ("4,096" / "4096", ’ / '),
so only part of a copied question matches token for token. It is ``partial`` when a few n-grams match (usually a common phrase), ``clean`` when none do, and ``too_short`` when it has fewer than
``--min-n`` tokens. ``flagged_ids`` (leaked questions) can be passed to ``scripts/eval_science.py --exclude``. For each
leaked question one matching training passage is decoded into the report as evidence.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llm.data import read_tokens  # noqa: E402
from llm.tokenizer import BPETokenizer  # noqa: E402
from eval_science import load_items  # noqa: E402

P = np.uint64(1_000_003)


def ngram_hashes(ids, n: int) -> np.ndarray:
    """Hash of every window of n consecutive tokens (uint64 arithmetic wraps, which is intended)."""
    a = np.asarray(ids, dtype=np.uint64)
    m = len(a) - n + 1
    if m <= 0:
        return np.zeros(0, dtype=np.uint64)
    h = np.zeros(m, dtype=np.uint64)
    with np.errstate(over="ignore"):
        for j in range(n):
            h = h * P + a[j:j + m]
    return h


def question_ngrams(items, tok, n: int, min_n: int) -> dict:
    """id -> (window length, array of its n-gram hashes); None for questions shorter than min_n tokens."""
    out = {}
    for it in items:
        ids = tok.encode(it["question"], allowed_special=set())
        k = min(n, len(ids))
        out[it["id"]] = None if k < min_n else (k, np.unique(ngram_hashes(ids, k)))
    return out


def scan(bins: list, qgrams: dict, chunk: int, log=print) -> dict:
    """hash -> (bin, token offset) of its first occurrence, for every question hash found in the bins."""
    by_n = {}
    for v in qgrams.values():
        if v is not None:
            by_n.setdefault(v[0], []).append(v[1])
    tables = {k: np.unique(np.concatenate(v)) for k, v in by_n.items()}
    found = {}
    for path in bins:
        _, toks = read_tokens(path)
        total, t0 = len(toks), time.time()
        maxn = max(tables) if tables else 1
        for start in range(0, total, chunk):
            a = np.asarray(toks[start:min(total, start + chunk + maxn - 1)])
            for k, table in tables.items():
                h = ngram_hashes(a, k)
                if len(h) == 0:
                    continue
                idx = np.minimum(np.searchsorted(table, h), len(table) - 1)
                for pos in np.flatnonzero(table[idx] == h):
                    found.setdefault(int(h[pos]), (os.path.basename(path), start + int(pos), k))
        log(f"  {os.path.basename(path)}: {total:,} tokens scanned in {time.time() - t0:.0f} s, {len(found)} question n-grams found so far")
    return found


def classify(qgrams: dict, found: dict, threshold: float, min_matched: int, min_hits: int = 2) -> dict:
    out = {}
    for qid, v in qgrams.items():
        if v is None:
            out[qid] = {"status": "too_short", "ngrams": 0, "matched": 0}
            continue
        hits = [int(x) for x in v[1] if int(x) in found]
        frac = len(hits) / len(v[1])
        out[qid] = {"status": "leaked" if (frac >= threshold and len(hits) >= min_hits) or len(hits) >= min_matched else ("partial" if hits else "clean"),
                    "ngrams": int(len(v[1])), "matched": len(hits), "fraction": round(frac, 3),
                    "first_hit": list(found[hits[0]]) if hits else None}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--evals", default="evals")
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--bins", nargs="+", required=True, help="training .bin files or glob patterns")
    ap.add_argument("--out", default="leakage")
    ap.add_argument("--n", type=int, default=13)
    ap.add_argument("--min-n", type=int, default=8)
    ap.add_argument("--threshold", type=float, default=0.2)
    ap.add_argument("--min-matched", type=int, default=10)
    ap.add_argument("--min-hits", type=int, default=2, help="matched n-grams needed besides --threshold")
    ap.add_argument("--chunk", type=int, default=50_000_000, help="tokens per chunk")
    a = ap.parse_args(argv)

    bins = sorted({p for pat in a.bins for p in (glob.glob(pat) if any(c in pat for c in "*?[") else [pat])})
    if not bins:
        print("no bins match")
        return 2
    tok = BPETokenizer.from_file(a.vocab)
    for p in bins:
        h, _ = read_tokens(p)
        if h.vocab_size != tok.n_vocab:
            print(f"{p}: vocab_size {h.vocab_size} != tokenizer {tok.n_vocab}")
            return 2
    items = load_items(a.evals)
    qgrams = question_ngrams(items, tok, a.n, a.min_n)
    print(f"{len(items)} questions, {sum(len(v[1]) for v in qgrams.values() if v)} distinct n-grams, {len(bins)} bin files")
    found = scan(bins, qgrams, a.chunk)
    per = classify(qgrams, found, a.threshold, a.min_matched, a.min_hits)

    by_item = {it["id"]: it for it in items}
    summary = {}
    for qid, r in per.items():
        it = by_item[qid]
        key = f"{it['domain']} | {it['source'].split(' (')[0]}"
        s = summary.setdefault(key, {"n": 0, "leaked": 0, "partial": 0, "clean": 0, "too_short": 0})
        s["n"] += 1
        s[r["status"]] += 1
    flagged = sorted(q for q, r in per.items() if r["status"] == "leaked")
    res = {"bins": [os.path.basename(p) for p in bins], "vocab_sha256": BPETokenizer.file_sha256(a.vocab),
           "n": a.n, "min_n": a.min_n, "threshold": a.threshold, "min_matched": a.min_matched, "min_hits": a.min_hits, "questions": len(items),
           "counts": {s: sum(1 for r in per.values() if r["status"] == s) for s in ("leaked", "partial", "clean", "too_short")},
           "by_domain_source": dict(sorted(summary.items())), "flagged_ids": flagged, "items": per}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out + ".json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)

    L = ["# Train / evaluation leakage check", "",
         f"- training files: {', '.join(res['bins'])}",
         f"- {len(items)} questions; window {a.n} tokens (min {a.min_n}); leaked = at least {a.threshold:.0%} (and {a.min_hits}) or {a.min_matched} of a question's n-grams found",
         f"- leaked {res['counts']['leaked']}, partial {res['counts']['partial']}, clean {res['counts']['clean']}, too short to check {res['counts']['too_short']}", "",
         "| domain | source | n | leaked | partial | clean | too short |", "|---|---|---|---|---|---|---|"]
    for key, s in res["by_domain_source"].items():
        d, src = key.split(" | ")
        L.append(f"| {d} | {src} | {s['n']} | {s['leaked']} | {s['partial']} | {s['clean']} | {s['too_short']} |")
    L += ["", "## Leaked questions with the matching training text", ""]
    files = {os.path.basename(p): p for p in bins}
    for qid in flagged[:50]:
        r = per[qid]
        bin_name, off, k = r["first_hit"]
        _, toks = read_tokens(files[bin_name])
        ctx = tok.decode(np.asarray(toks[max(0, off - 30):off + k + 30]).tolist()).replace("\n", " ")
        L += [f"**{qid}** ({r['matched']}/{r['ngrams']} n-grams) in `{bin_name}` at token {off:,}  ",
              f"Q: {by_item[qid]['question'][:250]}  ", f"Training text: …{ctx[:400]}…", ""]
    if len(flagged) > 50:
        L.append(f"... and {len(flagged) - 50} more in {os.path.basename(a.out)}.json")
    with open(a.out + ".md", "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print(f"wrote {a.out}.json and {a.out}.md: {res['counts']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
