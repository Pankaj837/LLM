#!/usr/bin/env python
"""Stream Parquet shards into TOK1 ``.bin`` files with a frozen tokenizer; or count tokens per domain.

Low memory: each shard is read in small batches and only the needed columns (``text``; ``doc_id``,
``domain``, ``char_count`` when present). Files in the v4.0 entity schema (no ``text``; body in the
``properties`` JSON, plus ``description``) are handled by ``v4_text`` (rule documented in
``docs/DATA_STATUS_2026-10-20.md``); ``--count-only`` on such files parses the JSON, so it is slower than the
``char_count`` path.

Parallelism: big files are split into jobs of whole row groups (about ``--chunk-mb`` of uncompressed data each),
so one 3 GB Parquet file keeps every core busy. A finished job is skipped on re-run (resumable). Exact-duplicate
removal (``--dedup``) is per job only; remove duplicates across files first with ``scripts/dedup_parquet.py``.
Text that cannot be encoded (lone surrogates from JSON escapes) is repaired, never fatal.

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
import re
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
CHUNK_MB = 256
_SURROGATE = re.compile("[\ud800-\udfff]")


def clean_text(text: str) -> str:
    """Make ``text`` encodable as UTF-8: lone surrogates (e.g. from JSON escapes) become U+FFFD."""
    try:
        text.encode("utf-8")
        return text
    except UnicodeEncodeError:
        return _SURROGATE.sub("�", text)


def _h64(s: str) -> int:
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8", "replace"), digest_size=8).digest(), "little")


def _s(x) -> str:
    return x.strip() if isinstance(x, str) else ""


def v4_text(description, properties) -> str:
    """Training text of one v4.0 entity row (no ``text`` column): the body lives in ``properties`` JSON.

    1. Code / documentation: ``code_content``, ``doc_content``, ``text``, ``content``.
    2. Q&A: Question, Options (if any), Answer, Explanation (skipping the ``Source origin:`` boilerplate).
    3. CAD: DeepCAD ``objective_prompt`` + ``sketch_extrude_completion``; otherwise description + ``cadquery_code`` +
       ``step_file_snippet`` + ``qa_pairs`` (list of dicts or a string).
    4. Fallback: ``description``.
    """
    if not properties:
        return description or ""
    try:
        p = json.loads(properties) if isinstance(properties, str) else properties
    except ValueError:
        return description or ""
    if not isinstance(p, dict):
        return description or ""
    for k in ("code_content", "doc_content", "text", "content"):
        if _s(p.get(k)):
            return p[k].strip()
    q, a = _s(p.get("question")), _s(p.get("answer"))
    if q and a:
        parts = [f"Question: {q}"]
        opts = p.get("options")
        if isinstance(opts, str):
            try:
                opts = json.loads(opts)
            except ValueError:
                opts = None
        if isinstance(opts, list):
            lines = [f"- {o}" for o in opts if o]
            if lines:
                parts.append("Options:\n" + "\n".join(lines))
        parts.append(f"Answer: {a}")
        exp = _s(p.get("explanation"))
        if exp and not exp.startswith("Source origin:"):
            parts.append(f"Explanation: {exp}")
        return "\n\n".join(parts)
    if _s(p.get("objective_prompt")) and _s(p.get("sketch_extrude_completion")):
        return f"Objective: {p['objective_prompt'].strip()}\n\nCompletion:\n{p['sketch_extrude_completion'].strip()}"
    cad = [_s(description)] if _s(description) else []
    for k in ("cadquery_code", "step_file_snippet"):
        if _s(p.get(k)):
            cad.append(p[k].strip())
    qa = p.get("qa_pairs")
    if isinstance(qa, list):
        for pair in qa:
            if isinstance(pair, dict):
                qq, aa = _s(pair.get("question") or pair.get("q")), _s(pair.get("answer") or pair.get("a"))
                if qq and aa:
                    cad.append(f"Q: {qq}\nA: {aa}")
    elif _s(qa):
        cad.append(qa.strip())
    if len(cad) > (1 if _s(description) else 0):
        return "\n\n".join(cad)
    return description or ""


def _write_header(f, vocab_size: int, eos_id: int, total: int, split: int) -> None:
    f.seek(0)
    f.write(struct.pack(HEADER_FMT, MAGIC, 1, vocab_size, eos_id, total, split, 0))


def process_file(job) -> dict:
    """Tokenize (or just count) one Parquet file, or a range of its row groups.

    ``job`` = (path, out_dir, vocab, val_frac, dedup, min_chars, count_only, force[, rg_lo, rg_hi, name]).
    """
    import pyarrow.parquet as pq

    path, out_dir, vocab, val_frac, dedup, min_chars, count_only, force = job[:8]
    rg_lo, rg_hi, name = (job[8:11] if len(job) >= 11 else (None, None, None))
    os.makedirs(out_dir, exist_ok=True)
    stem = name or Path(path).stem
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

    stats = {"file": os.path.basename(path), "job": stem, "docs": 0, "dup_dropped": 0, "short_dropped": 0, "chars": 0,
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
        rgs = list(range(rg_lo, rg_hi)) if rg_lo is not None else None
        for batch in pf.iter_batches(batch_size=BATCH, columns=cols, row_groups=rgs):
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
                text = clean_text(v4_text(d["description"][i], d["properties"][i]) if v4 else (d["text"][i] or ""))
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


def plan_jobs(files, a) -> list:
    """One job per file, or per group of whole row groups (~``chunk_mb`` uncompressed) for big files.

    Job names are the file stem, made unique (short path hash) when two inputs share a stem, plus ``.rgNNNNN``
    for the chunks of a split file. The list order is the merge order.
    """
    import pyarrow.parquet as pq

    stems: dict = {}
    for f in files:
        stems.setdefault(Path(f).stem, []).append(f)
    jobs = []
    for f in files:
        stem = Path(f).stem
        if len(stems[stem]) > 1:
            stem += "-" + hashlib.blake2b(os.path.abspath(f).encode(), digest_size=3).hexdigest()
        md = pq.ParquetFile(f).metadata
        n = md.num_row_groups
        target = max(1, int(a.chunk_mb * (1 << 20)))
        ranges, lo, acc = [], 0, 0
        for r in range(n):
            acc += md.row_group(r).total_byte_size
            if acc >= target:
                ranges.append((lo, r + 1))
                lo, acc = r + 1, 0
        if lo < n:
            ranges.append((lo, n))
        if len(ranges) <= 1:
            jobs.append((f, a.out, a.vocab, a.val_frac, a.dedup, a.min_chars, a.count_only, a.force, None, None, stem))
        else:
            for lo, hi in ranges:
                jobs.append((f, a.out, a.vocab, a.val_frac, a.dedup, a.min_chars, a.count_only, a.force, lo, hi,
                             f"{stem}.rg{lo:05d}"))
    return jobs


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
    ap.add_argument("--delete-shards", action="store_true",
                    help="remove per-job bins after a successful merge (this also removes the per-domain validation files)")
    ap.add_argument("--chunk-mb", type=float, default=CHUNK_MB, help="target uncompressed MB of row groups per parallel job")
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
    jobs = plan_jobs(files, a)
    names = [j[10] for j in jobs]
    assert len(set(names)) == len(names), "internal error: duplicate job names"
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        results = []
        for i, r in enumerate(ex.map(process_file, jobs), 1):
            results.append(r)
            if i % 10 == 0 or i == len(jobs):
                done = sum(x["train_tokens"] + x["val_tokens"] for x in results)
                print(f"[{i}/{len(jobs)}] jobs done, {time.time() - t0:.0f}s, {done:,} tokens", flush=True)

    per_domain: dict = {}
    for r in results:
        for dom, v in r["per_domain"].items():
            t = per_domain.setdefault(dom, {"docs": 0, "chars": 0, "tokens": 0})
            for k in t:
                t[k] += v[k]
    manifest = {"files": len(files), "jobs": len(jobs),
                "inputs": [{"path": f, "bytes": os.path.getsize(f)} for f in files], "docs": sum(r["docs"] for r in results), "chars": sum(r["chars"] for r in results),
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
                shards = [os.path.join(a.out, j[10] + f".{name}.bin") for j in jobs]
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
