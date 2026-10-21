#!/usr/bin/env python
"""Stream Parquet shards into TOK1 ``.bin`` files with a frozen tokenizer; or count tokens per domain.

Low memory: each shard is read in small batches and only the needed columns (``text``; ``doc_id``,
``domain``, ``char_count`` when present). Files in the v4.0 entity schema (no ``text``; body in the
``properties`` JSON, plus ``description``) are handled by ``v4_text`` -- a proposed rule the data owners
must confirm; ``--count-only`` on such files parses the JSON, so it is slower than the ``char_count`` path. One shard = one job, so shards run in parallel and a finished
shard is skipped on re-run (resumable).

  # fast token estimate without tokenizing (reads only char_count/domain, seconds not hours)
  python scripts/tokenize_parquet.py data/*.parquet --count-only --out runs/count

  # real tokenization (train/val split by document hash, exact-duplicate drop inside each shard)
  python scripts/tokenize_parquet.py data/*.parquet --vocab vocab32k.tok --out runs/bins --workers 8 --dedup

Writes per-shard ``<stem>.train.bin`` / ``<stem>.val.bin`` (valid TOK1 files), merges them into
``train.bin`` / ``val.bin`` unless ``--no-merge``, and writes ``manifest.json`` (tokenizer sha256, token
counts per domain and per shard). EOS is appended after every document; user text never forges one.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import struct
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from llm.data import HEADER_FMT, HEADER_SIZE, MAGIC, SPLIT_TRAIN, SPLIT_VAL  # noqa: E402
from llm.tokenizer import BPETokenizer  # noqa: E402

BATCH = 2000
MOD = 1_000_000


def _h64(s: str) -> int:
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8", "replace"), digest_size=8).digest(), "little")


def v4_text(description, properties) -> str:
    """Training text of one v4.0 entity row (no ``text`` column): the body lives in ``properties`` JSON.

    code_content / text / content if present; else Question + Answer; else CAD (description + cadquery_code + STEP
    snippet + Q&A); else ``description``.
    """
    if properties:
        try:
            p = json.loads(properties)
        except ValueError:
            p = None
        if isinstance(p, dict):
            for k in ("code_content", "text", "content"):
                if isinstance(p.get(k), str) and p[k].strip():
                    return p[k]
            q, a = p.get("question"), p.get("answer")
            if isinstance(q, str) and isinstance(a, str) and q.strip() and a.strip():
                return f"Question: {q}\n\nAnswer: {a}"
            if isinstance(p.get("cadquery_code"), str) and p["cadquery_code"].strip():      # CAD/STEP records
                parts = [description or "", p["cadquery_code"], p.get("step_file_snippet") or "", p.get("qa_pairs") or ""]
                return "\n\n".join(x for x in parts if isinstance(x, str) and x.strip())
    return description or ""


def _write_header(f, vocab_size: int, eos_id: int, total: int, split: int) -> None:
    f.seek(0)
    f.write(struct.pack(HEADER_FMT, MAGIC, 1, vocab_size, eos_id, total, split, 0))


def process_file(job) -> dict:
    """Tokenize (or just count) one Parquet shard. ``job`` = (path, out_dir, vocab, val_frac, dedup, min_chars, count_only, force)."""
    import pyarrow.parquet as pq

    path, out_dir, vocab, val_frac, dedup, min_chars, count_only, force = job
    os.makedirs(out_dir, exist_ok=True)
    stem = Path(path).stem
    stats_p = Path(out_dir) / f"{stem}.stats.json"
    if not count_only and stats_p.exists() and not force:
        return json.loads(stats_p.read_text())

    pf = pq.ParquetFile(path)
    names = set(pf.schema_arrow.names)
    v4 = "text" not in names and "properties" in names and "description" in names
    if "text" not in names and not v4 and not (count_only and "char_count" in names):
        raise ValueError(f"{path}: no 'text' column (columns: {sorted(names)})")
    use_chars_only = count_only and "char_count" in names and not dedup and not min_chars and not v4
    cols = [c for c in ("text", "doc_id", "domain", "char_count") if c in names]
    if v4:
        cols = [c for c in ("entity_id", "domain", "description", "properties") if c in names]
    if use_chars_only:
        cols = [c for c in ("char_count", "domain") if c in names]

    stats = {"file": os.path.basename(path), "docs": 0, "dup_dropped": 0, "short_dropped": 0, "chars": 0,
             "train_tokens": 0, "val_tokens": 0, "per_domain": {}}
    seen: set = set()
    thresh = int(round(val_frac * MOD))
    tok = None if count_only else BPETokenizer.from_file(vocab)
    ftrain = fval = None
    if not count_only:
        ftrain, fval = open(Path(out_dir) / f"{stem}.train.bin", "wb"), open(Path(out_dir) / f"{stem}.val.bin", "wb")
        for f in (ftrain, fval):
            f.write(b"\0" * HEADER_SIZE)
    try:
        for batch in pf.iter_batches(batch_size=BATCH, columns=cols):
            d = batch.to_pydict()
            n = batch.num_rows
            domains = d.get("domain") or ["unknown"] * n
            for i in range(n):
                dom = domains[i] or "unknown"
                pd_ = stats["per_domain"].setdefault(dom, {"docs": 0, "chars": 0, "tokens": 0})
                if use_chars_only:
                    stats["docs"] += 1
                    stats["chars"] += int(d["char_count"][i])
                    pd_["docs"] += 1
                    pd_["chars"] += int(d["char_count"][i])
                    continue
                text = v4_text(d["description"][i], d["properties"][i]) if v4 else (d["text"][i] or "")
                if len(text) < min_chars:
                    stats["short_dropped"] += 1
                    continue
                key = _h64(text)
                if dedup:
                    if key in seen:
                        stats["dup_dropped"] += 1
                        continue
                    seen.add(key)
                stats["docs"] += 1
                stats["chars"] += len(text)
                pd_["docs"] += 1
                pd_["chars"] += len(text)
                if count_only:
                    continue
                id_col = "doc_id" if "doc_id" in d else "entity_id"
                ident = d[id_col][i] if id_col in d and d[id_col][i] else key
                is_val = (_h64(str(ident)) % MOD) < thresh
                ids = tok.encode(text, allowed_special=set())
                ids.append(tok.eos_id)
                np.asarray(ids, dtype="<u2").tofile(fval if is_val else ftrain)
                stats["val_tokens" if is_val else "train_tokens"] += len(ids)
                pd_["tokens"] += len(ids)
    finally:
        if not count_only:
            _write_header(ftrain, tok.n_vocab, tok.eos_id, stats["train_tokens"], SPLIT_TRAIN)
            _write_header(fval, tok.n_vocab, tok.eos_id, stats["val_tokens"], SPLIT_VAL)
            ftrain.close()
            fval.close()
    if not count_only:
        stats_p.write_text(json.dumps(stats))
    return stats


def merge_bins(shards, out: str, vocab_size: int, eos_id: int, split: int) -> int:
    """Concatenate shard ``.bin`` files (headers stripped) into one valid TOK1 file. Returns total tokens."""
    total = 0
    with open(out, "wb") as fo:
        fo.write(b"\0" * HEADER_SIZE)
        for s in shards:
            with open(s, "rb") as fi:
                fi.seek(HEADER_SIZE)
                while True:
                    chunk = fi.read(64 << 20)
                    if not chunk:
                        break
                    fo.write(chunk)
                    total += len(chunk) // 2
        _write_header(fo, vocab_size, eos_id, total, split)
    return total


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="Parquet files, globs or directories")
    ap.add_argument("--out", required=True)
    ap.add_argument("--vocab", help="frozen .tok vocabulary (required unless --count-only)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--val-frac", type=float, default=0.001, help="fraction of DOCUMENTS held out (hash-based, deterministic)")
    ap.add_argument("--dedup", action="store_true", help="drop exact-duplicate texts inside each shard")
    ap.add_argument("--min-chars", type=int, default=0)
    ap.add_argument("--count-only", action="store_true")
    ap.add_argument("--chars-per-token", type=float, default=3.6, help="only used for the --count-only estimate")
    ap.add_argument("--no-merge", action="store_true")
    ap.add_argument("--delete-shards", action="store_true", help="remove per-shard bins after a successful merge")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    if not a.count_only and not a.vocab:
        ap.error("--vocab is required unless --count-only")

    files = []
    for p in a.inputs:
        files += sorted(glob.glob(os.path.join(p, "**", "*.parquet"), recursive=True)) if os.path.isdir(p) else sorted(glob.glob(p))
    files = sorted(set(files))
    if not files:
        ap.error("no Parquet files matched")
    os.makedirs(a.out, exist_ok=True)
    jobs = [(f, a.out, a.vocab, a.val_frac, a.dedup, a.min_chars, a.count_only, a.force) for f in files]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        results = []
        for i, r in enumerate(ex.map(process_file, jobs), 1):
            results.append(r)
            if i % 10 == 0 or i == len(jobs):
                print(f"[{i}/{len(jobs)}] shards done, {time.time() - t0:.0f}s", flush=True)

    per_domain: dict = {}
    for r in results:
        for dom, v in r["per_domain"].items():
            t = per_domain.setdefault(dom, {"docs": 0, "chars": 0, "tokens": 0})
            for k in t:
                t[k] += v[k]
    manifest = {"files": len(files), "docs": sum(r["docs"] for r in results), "chars": sum(r["chars"] for r in results),
                "dup_dropped": sum(r["dup_dropped"] for r in results), "per_domain": per_domain, "shards": results}
    if a.count_only:
        cpt = a.chars_per_token
        manifest["estimate_chars_per_token"] = cpt
        manifest["estimated_tokens"] = int(manifest["chars"] / cpt)
        for v in per_domain.values():
            v["estimated_tokens"] = int(v["chars"] / cpt)
    else:
        tok = BPETokenizer.from_file(a.vocab)
        manifest.update(tokenizer_sha256=BPETokenizer.file_sha256(a.vocab), vocab_size=tok.n_vocab, eos_id=tok.eos_id,
                        train_tokens=sum(r["train_tokens"] for r in results), val_tokens=sum(r["val_tokens"] for r in results))
        if not a.no_merge:
            for split, name in ((SPLIT_TRAIN, "train"), (SPLIT_VAL, "val")):
                shards = [os.path.join(a.out, Path(f).stem + f".{name}.bin") for f in files]
                n = merge_bins(shards, os.path.join(a.out, f"{name}.bin"), tok.n_vocab, tok.eos_id, split)
                assert n == manifest[f"{name}_tokens"], (name, n, manifest[f"{name}_tokens"])
                if a.delete_shards:
                    for s in shards:
                        os.remove(s)
    Path(a.out, "manifest.json").write_text(json.dumps(manifest, indent=2))
    key = "estimated_tokens" if a.count_only else "train_tokens"
    print(f"docs={manifest['docs']:,} chars={manifest['chars']:,} {key}={manifest.get(key, 0):,}")
    for dom, v in sorted(per_domain.items(), key=lambda kv: -kv[1]["chars"]):
        print(f"  {dom:<28} docs={v['docs']:>10,} chars={v['chars']:>16,}" + (f" tokens={v['tokens']:>14,}" if not a.count_only else f" ~tokens={v['estimated_tokens']:>14,}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
