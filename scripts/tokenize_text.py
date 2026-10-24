#!/usr/bin/env python
"""Tokenize plain-text corpora (``.txt`` / ``.txt.gz``) into TOK1 ``.bin`` files with a frozen tokenizer.

Companion of ``tokenize_parquet.py`` for the corpora that are not Parquet: the math corpus (one document per
line, ~34 GB unpacked), Multi-step / Astronomy / Finance (documents separated by blank lines), the chemistry text
files, OpenWebMath shards. One reader streams each file (gzip is decompressed on the fly, nothing is unpacked to
disk) and a pool of workers tokenizes batches, so all cores stay busy and memory stays small.

    # <domain>=<file> names the domain in the manifest; --blank-domains lists domains whose documents are
    # blocks of lines separated by blank lines (default: one document per line)
    python scripts/tokenize_text.py math=maths_final_corpus.clean.txt.gz multistep=multistep_corpus.txt \
        astronomy=astronomy_corpus.txt --blank-domains multistep,astronomy \
        --vocab tokenizers/vocab32k_v1/vocab32k.tok --out runs/bins_text --workers 8 --dedup

Per input it writes ``<domain>.<file>.train.bin`` / ``.val.bin`` (valid TOK1 files; the validation set is a
deterministic hash-based share of DOCUMENTS), merges them into ``train.bin`` / ``val.bin`` unless ``--no-merge``,
and writes ``manifest.json`` (tokenizer sha256, per-domain counts, inputs). EOS is appended after every document.
A finished input is skipped on re-run. ``--dedup`` drops exact duplicates across all inputs of this run (8 bytes per
kept document in memory); ``--count-only`` reads and counts characters without tokenizing.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tokenize_parquet as tp  # noqa: E402
from llm.data import HEADER_SIZE, SPLIT_TRAIN, SPLIT_VAL  # noqa: E402
from llm.tokenizer import BPETokenizer  # noqa: E402

BATCH_CHARS = 1_000_000
BATCH_DOCS = 20_000
_TOK = None
_THRESH = 0


def _open(path: str):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace", newline="\n")
    return open(path, "r", encoding="utf-8", errors="replace", newline="\n")


def iter_docs(path: str, blank: bool):
    """Yield documents: one per non-empty line, or blocks of lines separated by blank lines."""
    with _open(path) as f:
        if not blank:
            for line in f:
                line = line.rstrip("\r\n")
                if line.strip():
                    yield line
            return
        block: list = []
        for line in f:
            line = line.rstrip("\r\n")
            if line.strip():
                block.append(line)
            elif block:
                yield "\n".join(block)
                block = []
        if block:
            yield "\n".join(block)


def _init_worker(vocab: str, val_frac: float) -> None:
    global _TOK, _THRESH
    _TOK = BPETokenizer.from_file(vocab)
    _THRESH = int(round(val_frac * tp.MOD))


def _encode_batch(docs: list):
    """Tokenize one batch -> (train_bytes, val_bytes, n_train_tokens, n_val_tokens)."""
    train, val = [], []
    nt = nv = 0
    for text in docs:
        ids = _TOK.encode(tp.clean_text(text), allowed_special=set())
        ids.append(_TOK.eos_id)
        arr = np.asarray(ids, dtype="<u2")
        if (tp._h64(text) % tp.MOD) < _THRESH:
            val.append(arr.tobytes())
            nv += len(ids)
        else:
            train.append(arr.tobytes())
            nt += len(ids)
    return b"".join(train), b"".join(val), nt, nv


def _batches(path: str, blank: bool, seen, dedup: bool, min_chars: int, stats: dict):
    batch, chars = [], 0
    for doc in iter_docs(path, blank):
        stats["in"] += 1
        if len(doc) < min_chars:
            stats["short_dropped"] += 1
            continue
        if dedup:
            k = tp._h64(" ".join(doc.lower().split()))
            if k in seen:
                stats["dup_dropped"] += 1
                continue
            seen.add(k)
        stats["docs"] += 1
        stats["chars"] += len(doc)
        batch.append(doc)
        chars += len(doc)
        if chars >= BATCH_CHARS or len(batch) >= BATCH_DOCS:
            yield batch
            batch, chars = [], 0
    if batch:
        yield batch


def process_input(domain: str, path: str, name: str, a, ex, seen) -> dict:
    out = Path(a.out)
    stats_p = out / f"{name}.stats.json"
    if not a.count_only and stats_p.exists() and not a.force:
        return json.loads(stats_p.read_text())
    blank = domain in a.blank_set
    st = {"file": os.path.basename(path), "job": name, "domain": domain, "in": 0, "docs": 0, "dup_dropped": 0,
          "short_dropped": 0, "chars": 0, "train_tokens": 0, "val_tokens": 0}
    batches = _batches(path, blank, seen, a.dedup, a.min_chars, st)
    if a.count_only:
        for _ in batches:
            pass
    else:
        ftrain, fval = open(out / f"{name}.train.bin", "wb"), open(out / f"{name}.val.bin", "wb")
        for f in (ftrain, fval):
            f.write(b"\0" * HEADER_SIZE)
        pending: deque = deque()
        t0, last = time.time(), time.time()

        def drain(block: bool) -> None:
            while pending and (block or len(pending) >= 4 * a.workers):
                tb, vb, nt, nv = pending.popleft().result()
                ftrain.write(tb)
                fval.write(vb)
                st["train_tokens"] += nt
                st["val_tokens"] += nv

        try:
            for batch in batches:
                pending.append(ex.submit(_encode_batch, batch))
                drain(False)
                if time.time() - last > 30:
                    last = time.time()
                    print(f"  {name}: docs={st['docs']:,} tokens={st['train_tokens'] + st['val_tokens']:,} "
                          f"({(time.time() - t0):.0f}s)", flush=True)
            drain(True)
        finally:
            tok_id = BPETokenizer.from_file(a.vocab)
            tp._write_header(ftrain, tok_id.n_vocab, tok_id.eos_id, st["train_tokens"], SPLIT_TRAIN)
            tp._write_header(fval, tok_id.n_vocab, tok_id.eos_id, st["val_tokens"], SPLIT_VAL)
            ftrain.close()
            fval.close()
        stats_p.write_text(json.dumps(st))
    return st


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="[domain=]file.txt[.gz]; the domain defaults to the file stem")
    ap.add_argument("--out", required=True)
    ap.add_argument("--vocab", help="frozen .tok vocabulary (required unless --count-only)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--val-frac", type=float, default=0.001)
    ap.add_argument("--blank-domains", default="", help="comma-separated domains whose documents are blank-line separated blocks")
    ap.add_argument("--dedup", action="store_true", help="drop exact-duplicate documents across all inputs of this run")
    ap.add_argument("--min-chars", type=int, default=0)
    ap.add_argument("--count-only", action="store_true")
    ap.add_argument("--chars-per-token", type=float, default=3.6, help="only for the --count-only estimate")
    ap.add_argument("--no-merge", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    if not a.count_only and not a.vocab:
        ap.error("--vocab is required unless --count-only")
    a.blank_set = {x.strip() for x in a.blank_domains.split(",") if x.strip()}

    items = []
    for spec in a.inputs:
        domain, _, path = spec.partition("=") if "=" in spec and not os.path.exists(spec) else ("", "", spec)
        if not path:
            path = spec
        if not os.path.isfile(path):
            ap.error(f"input not found: {path}")
        stem = Path(path).name
        for suf in (".gz", ".txt"):
            if stem.endswith(suf):
                stem = stem[:-len(suf)]
        items.append((domain or stem, path))
    os.makedirs(a.out, exist_ok=True)
    names, used = [], set()
    for domain, path in items:
        base = f"{domain}.{Path(path).name.split('.')[0]}"
        name = base if base not in used else base + "-" + hashlib.blake2b(os.path.abspath(path).encode(), digest_size=3).hexdigest()
        used.add(name)
        names.append(name)

    seen: set = set()
    results = []
    t0 = time.time()
    ex = None if a.count_only else ProcessPoolExecutor(max_workers=a.workers, initializer=_init_worker, initargs=(a.vocab, a.val_frac))
    try:
        for (domain, path), name in zip(items, names):
            print(f"[{len(results) + 1}/{len(items)}] {path}", flush=True)
            results.append(process_input(domain, path, name, a, ex, seen))
    finally:
        if ex is not None:
            ex.shutdown()

    per_domain: dict = {}
    for r in results:
        t = per_domain.setdefault(r["domain"], {"docs": 0, "chars": 0, "tokens": 0})
        t["docs"] += r["docs"]
        t["chars"] += r["chars"]
        t["tokens"] += r["train_tokens"] + r["val_tokens"]
    manifest = {"files": len(items), "inputs": [{"path": p, "bytes": os.path.getsize(p)} for _, p in items],
                "docs": sum(r["docs"] for r in results), "chars": sum(r["chars"] for r in results),
                "dup_dropped": sum(r["dup_dropped"] for r in results), "per_domain": per_domain, "shards": results}
    if a.count_only:
        manifest["estimate_chars_per_token"] = a.chars_per_token
        manifest["estimated_tokens"] = int(manifest["chars"] / a.chars_per_token)
        for v in per_domain.values():
            v["estimated_tokens"] = int(v["chars"] / a.chars_per_token)
    else:
        tok = BPETokenizer.from_file(a.vocab)
        manifest.update(tokenizer_sha256=BPETokenizer.file_sha256(a.vocab), vocab_size=tok.n_vocab, eos_id=tok.eos_id,
                        train_tokens=sum(r["train_tokens"] for r in results), val_tokens=sum(r["val_tokens"] for r in results))
        if not a.no_merge:
            for split, which in ((SPLIT_TRAIN, "train"), (SPLIT_VAL, "val")):
                n = tp.merge_bins([os.path.join(a.out, nm + f".{which}.bin") for nm in names],
                                  os.path.join(a.out, f"{which}.bin"), tok.n_vocab, tok.eos_id, split)
                assert n == manifest[f"{which}_tokens"], (which, n, manifest[f"{which}_tokens"])
    Path(a.out, "manifest.json").write_text(json.dumps(manifest, indent=2))
    key = "estimated_tokens" if a.count_only else "train_tokens"
    print(f"docs={manifest['docs']:,} chars={manifest['chars']:,} {key}={manifest.get(key, 0):,} ({time.time() - t0:.0f}s)")
    for dom, v in sorted(per_domain.items(), key=lambda kv: -kv[1]["chars"]):
        print(f"  {dom:<20} docs={v['docs']:>12,} chars={v['chars']:>16,} " +
              (f"~tokens={v['estimated_tokens']:>14,}" if a.count_only else f"tokens={v['tokens']:>14,}"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
