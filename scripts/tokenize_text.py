#!/usr/bin/env python
"""Tokenize plain-text corpora (``.txt`` / ``.txt.gz``) into TOK1 ``.bin`` files with a frozen tokenizer.

Companion of ``tokenize_parquet.py`` for the corpora that are not Parquet: the math corpus (one document per
line, ~34 GB unpacked), Multi-step / Astronomy / Finance (documents separated by blank lines), the chemistry text
files, OpenWebMath shards. One reader streams each file (gzip is decompressed on the fly, nothing is unpacked to
disk) and a pool of workers tokenizes batches, so all cores stay busy and memory stays small.

    # <domain>=<file> names the domain in the manifest; --blank-domains lists domains whose documents are
    # blocks of lines separated by blank lines, --marker-domains those whose documents sit between a line
    # <|document|> and a line <|endofdocument|> (the physics file), --eot-domains those whose documents are
    # separated by a line <|endoftext|> (the engineering reasoning file); default: one document per line
    python scripts/tokenize_text.py math=maths_final_corpus.clean.txt.gz multistep=multistep_corpus.txt \
        astronomy=astronomy_corpus.txt --blank-domains multistep,astronomy \
        --vocab tokenizers/vocab32k_v2/vocab32k.tok --out runs/bins_text --workers 8 --dedup

Per input it writes ``<domain>.<file>.train.bin`` / ``.val.bin`` (valid TOK1 files; the validation set is a
deterministic hash-based share of DOCUMENTS), merges them into ``train.bin`` / ``val.bin`` unless ``--no-merge``,
and writes ``manifest.json`` (tokenizer sha256, per-domain counts, inputs). EOS is appended after every document.
``--count-only`` counts documents with the same delimiter flags as a real run (``--blank-domains`` / ``--marker-domains``);
without them every line is a document. A finished input is skipped on re-run, but only if the vocabulary, settings and input file are unchanged (a fingerprint
is stored with it). ``--dedup`` drops exact duplicates across all inputs of this run (about 80 bytes of memory per
kept document: 5M documents ~ 400 MB); ``--count-only`` reads and counts characters without tokenizing.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
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


DOC_START, DOC_END = "<|document|>", "<|endofdocument|>"
DOC_EOT = "<|endoftext|>"


def _docs_from_lines(lines, blank, marker, eot=False):
    """Documents from an iterable of lines: one per non-empty line, blank-line separated blocks (``blank``), or the text between
    ``<|document|>`` and ``<|endofdocument|>`` lines (``marker``; the marker lines are dropped, an unterminated tail is kept), or the text
    between lines that read ``<|endoftext|>`` (``eot``: reasoning corpora whose documents contain many blank lines; the separator lines are dropped)."""
    if eot:
        buf = []
        for line in lines:
            line = line.rstrip("\r\n")
            if line.strip() == DOC_EOT:
                doc = "\n".join(buf).strip()
                if doc:
                    yield doc
                buf = []
            else:
                buf.append(line)
        doc = "\n".join(buf).strip()
        if doc:
            yield doc
        return
    if marker:
        buf, inside = [], False
        for line in lines:
            line = line.rstrip("\r\n")
            if line.strip() == DOC_START:
                if inside and buf:
                    yield "\n".join(buf).strip()
                buf, inside = [], True
            elif line.strip() == DOC_END:
                if inside and buf:
                    yield "\n".join(buf).strip()
                buf, inside = [], False
            elif inside:
                buf.append(line)
        if inside and buf:
            yield "\n".join(buf).strip()
        return
    if not blank:
        for line in lines:
            line = line.rstrip("\r\n")
            if line.strip():
                yield line
        return
    block: list = []
    for line in lines:
        line = line.rstrip("\r\n")
        if line.strip():
            block.append(line)
        elif block:
            yield "\n".join(block)
            block = []
    if block:
        yield "\n".join(block)


def _trim_slice(text: str, blank: bool, marker: bool, eot: bool = False) -> str:
    """A slice cut out of the middle of a file starts and ends mid-document: keep only the whole documents inside it."""
    if eot:
        i, j = text.find("\n" + DOC_EOT), text.rfind("\n" + DOC_EOT)
        return text[i + 1 + len(DOC_EOT):j + 1 + len(DOC_EOT)] if 0 <= i < j else ""
    if marker:
        i, j = text.find("\n" + DOC_START), text.rfind(DOC_END)
        return text[i + 1:j + len(DOC_END)] if 0 <= i < j else ""
    sep = "\n\n" if blank else "\n"
    i, j = text.find(sep), text.rfind(sep)
    return text[i + len(sep):j + len(sep)] if 0 <= i < j else ""


def iter_docs(path: str, blank, marker: bool = False, spread: int = 0, slice_bytes: int = 0, eot: bool = False):
    """Yield documents of one file. With ``spread`` > 0 (plain-text files only) read ``spread`` evenly spaced slices of
    ``slice_bytes`` instead of the whole file, so a capped read is representative rather than a prefix."""
    if spread > 0 and slice_bytes > 0 and not path.endswith(".gz"):
        size = os.path.getsize(path)
        if size > spread * slice_bytes:
            with open(path, "rb") as f:
                for i in range(spread):
                    off = max(0, min(size - slice_bytes, int(size * (i + 0.5) / spread - slice_bytes / 2)))
                    f.seek(off)
                    raw = f.read(slice_bytes).decode("utf-8", errors="replace").replace("\r\n", "\n")     # Windows files
                    text = _trim_slice(raw, blank, marker, eot)
                    yield from _docs_from_lines(io.StringIO(text), blank, marker, eot)
            return
    with _open(path) as f:
        yield from _docs_from_lines(f, blank, marker, eot)


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


def _batches(path: str, blank: bool, marker: bool, seen, dedup: bool, min_chars: int, stats: dict, spread: int = 0, slice_bytes: int = 0, eot: bool = False):
    batch, chars = [], 0
    for doc in iter_docs(path, blank, marker, spread, slice_bytes, eot):
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
    st_in = os.stat(path)
    fp = tp.fingerprint(rule=tp.RULE_VERSION, vocab=BPETokenizer.file_sha256(a.vocab) if not a.count_only else None,
                        val_frac=a.val_frac, dedup=a.dedup, min_chars=a.min_chars, blank=domain in a.blank_set, marker=domain in a.marker_set, eot=domain in a.eot_set, max_tokens=a.max_tokens, spread=a.spread,
                        size=st_in.st_size, mtime=int(st_in.st_mtime))
    if not a.count_only and stats_p.exists() and not a.force:
        done = json.loads(stats_p.read_text())
        if done.get("fingerprint") == fp:
            return done
        print(f"[tokenize] {name}: settings, vocabulary or input changed since the earlier run -> recomputing", flush=True)
    blank = domain in a.blank_set
    marker = domain in a.marker_set
    eot = domain in a.eot_set
    st = {"file": os.path.basename(path), "job": name, "fingerprint": fp, "domain": domain, "in": 0, "docs": 0, "dup_dropped": 0,
          "short_dropped": 0, "chars": 0, "train_tokens": 0, "val_tokens": 0}
    slice_bytes = int(a.max_tokens * 4.0 / a.spread) if (a.spread and a.max_tokens) else 0
    batches = _batches(path, blank, marker, seen, a.dedup, a.min_chars, st, a.spread, slice_bytes, eot)
    if a.count_only:
        for _ in batches:
            pass
    else:
        tmp_t, tmp_v = out / f"{name}.train.bin.tmp", out / f"{name}.val.bin.tmp"
        ftrain, fval = open(tmp_t, "wb"), open(tmp_v, "wb")
        for f in (ftrain, fval):
            f.write(b"\0" * HEADER_SIZE)
        pending: deque = deque()                       # (future, characters in that batch)
        t0, last = time.time(), time.time()
        done_chars = 0

        def drain(block: bool) -> None:
            nonlocal done_chars
            while pending and (block or len(pending) >= 4 * a.workers):
                fut, nchars = pending.popleft()
                tb, vb, nt, nv = fut.result()
                done_chars += nchars
                ftrain.write(tb)
                fval.write(vb)
                st["train_tokens"] += nt
                st["val_tokens"] += nv

        ok = False
        try:
            for batch in batches:
                pending.append((ex.submit(_encode_batch, batch), sum(len(x) for x in batch)))
                drain(False)
                if a.max_tokens:
                    done_tok = st["train_tokens"] + st["val_tokens"]
                    ratio = done_tok / done_chars if done_chars else 0.3            # tokens per character so far
                    if done_tok + ratio * sum(c for _, c in pending) >= a.max_tokens:
                        break               # what is already submitted will reach the cap (no large overshoot)
                if time.time() - last > 30:
                    last = time.time()
                    print(f"  {name}: docs={st['docs']:,} tokens={st['train_tokens'] + st['val_tokens']:,} "
                          f"({(time.time() - t0):.0f}s)", flush=True)
            drain(True)
            ok = True
        finally:
            if ok:
                tok_id = BPETokenizer.from_file(a.vocab)
                tp._write_header(ftrain, tok_id.n_vocab, tok_id.eos_id, st["train_tokens"], SPLIT_TRAIN)
                tp._write_header(fval, tok_id.n_vocab, tok_id.eos_id, st["val_tokens"], SPLIT_VAL)
            ftrain.close()
            fval.close()
            if not ok:                      # a failed input leaves nothing that could be mistaken for a finished one
                tmp_t.unlink(missing_ok=True)
                tmp_v.unlink(missing_ok=True)
        os.replace(tmp_t, out / f"{name}.train.bin")
        os.replace(tmp_v, out / f"{name}.val.bin")
        tmp = stats_p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(st))
        os.replace(tmp, stats_p)
    return st


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="[domain=]file.txt[.gz]; the domain defaults to the file stem")
    ap.add_argument("--out", required=True)
    ap.add_argument("--vocab", help="frozen .tok vocabulary (required unless --count-only)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--val-frac", type=float, default=0.001)
    ap.add_argument("--blank-domains", default="", help="comma-separated domains whose documents are blank-line separated blocks")
    ap.add_argument("--marker-domains", default="", help="comma-separated domains whose documents sit between <|document|> and <|endofdocument|> lines")
    ap.add_argument("--eot-domains", default="", help="comma-separated domains whose documents are separated by lines reading <|endoftext|>")
    ap.add_argument("--dedup", action="store_true", help="drop exact-duplicate documents across all inputs of this run")
    ap.add_argument("--min-chars", type=int, default=0)
    ap.add_argument("--count-only", action="store_true")
    ap.add_argument("--chars-per-token", type=float, default=3.6, help="only for the --count-only estimate")
    ap.add_argument("--max-tokens", type=int, default=0,
                    help="stop reading an input after about this many tokens (a prefix: shuffle or sample first if the order carries meaning)")
    ap.add_argument("--spread", type=int, default=0,
                    help="with --max-tokens on a plain-text file: read this many evenly spaced slices instead of a prefix (not for .gz)")
    ap.add_argument("--no-merge", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    if not a.count_only and not a.vocab:
        ap.error("--vocab is required unless --count-only")
    a.blank_set = {x.strip() for x in a.blank_domains.split(",") if x.strip()}
    a.marker_set = {x.strip() for x in a.marker_domains.split(",") if x.strip()}
    a.eot_set = {x.strip() for x in a.eot_domains.split(",") if x.strip()}

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
