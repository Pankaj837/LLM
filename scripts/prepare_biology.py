"""Prepare the Biology dataset (7 gzipped JSONL shards inside a zip) for a real-data smoke test.

    python scripts/prepare_biology.py --zip "Biology Dataset.zip" --out data/bio --vocab-size 8000

Every decision is explicit and recorded in ``manifest.json``:

  * Sources: pes2o, pmc_oa, libretexts, qbio_full, qbio, bhl are TRAINING sources. ``gutenberg``
    (general books; bio-term density ~50x lower) is kept out of training and written as a CONTROL set
    to show the model is not simply modelling English.
  * Sampling: a fixed character budget per source, taken with a deterministic stride over the whole shard
    (so early/late documents are both represented) instead of "the first N documents".
  * Cleaning (minimal, documented): drop documents shorter than ``--min-doc-chars``; strip BOM; normalise
    CRLF; collapse 3+ blank lines to 2; exact AND near-duplicate documents (>=30% of 12-word shingles already
    seen; LibreTexts republishes chapters) are dropped globally before splitting; documents longer than ``--max-doc-chars`` are cut at a whitespace boundary
    so one book/paper cannot dominate.
  * Split: by DOCUMENT and stratified per source (never by chunk), so validation text is never seen in training.
  * Tokenizer: byte-level BPE trained on TRAINING documents only (so the vocabulary cannot leak validation text),
    using the pure-Python stand-in trainer (NOT the team's C++ vocabulary).
  * Checks recorded: round-trip exactness of every document, ids < vocab and < 65536, EOS count == document
    count, bytes/token per source, share of single-byte tokens (fragmentation), and the fraction of validation
    32-token windows that also occur in training (near-duplicate leakage indicator).
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import re
import sys
import time
import zipfile
from collections import Counter
from typing import Dict, Iterator, List, Tuple

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from llm import data  # noqa: E402
from llm.bpe_trainer import train_bpe, write_tok  # noqa: E402
from llm.tokenizer import BPETokenizer  # noqa: E402

TRAIN_BUDGET_MB = {"pes2o": 6.0, "pmc_oa": 5.0, "libretexts": 5.0, "qbio_full": 4.0, "qbio": 3.0, "bhl": 1.5}
CONTROL_SOURCE, CONTROL_BUDGET_MB = "gutenberg", 2.0
BLANKS = re.compile(r"\n{3,}")


def clean(text: str) -> str:
    text = text.replace("﻿", "").replace("\r\n", "\n").replace("\r", "\n")
    return BLANKS.sub("\n\n", text).strip()


def truncate(text: str, limit: int) -> Tuple[str, bool]:
    if len(text) <= limit:
        return text, False
    cut = text.rfind(" ", limit - 400, limit)
    return text[: cut if cut > 0 else limit].rstrip(), True


def iter_shard(zf: zipfile.ZipFile, source: str) -> Iterator[dict]:
    name = f"{source}_cleaned.jsonl.gz"
    with zf.open(name) as raw, gzip.open(raw, "rt", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def sample_source(zf, source: str, budget_chars: int, max_doc_chars: int, min_doc_chars: int,
                  dedupe: data.NearDuplicateFilter, seed: int) -> Tuple[List[str], Dict]:
    """Strided, deduplicated sample of one shard within a character budget."""
    lens = []
    for r in iter_shard(zf, source):                       # pass 1: how big is the shard (after truncation)?
        lens.append(min(len(r.get("text") or ""), max_doc_chars))
    total = sum(lens)
    stride = max(1, math.ceil(total / budget_chars))
    offset = seed % stride
    docs: List[str] = []
    st = Counter()
    chars = 0
    for i, r in enumerate(iter_shard(zf, source)):         # pass 2: take every stride-th document
        if i % stride != offset:
            continue
        st["candidates"] += 1
        text = clean(r.get("text") or "")
        if len(text) < min_doc_chars:
            st["dropped_short"] += 1
            continue
        text, cut = truncate(text, max_doc_chars)          # judge duplication on what would actually be used
        if dedupe.is_duplicate(text):
            st["dropped_duplicate"] += 1
            continue
        st["truncated"] += cut
        docs.append(text)
        chars += len(text)
        if chars >= budget_chars:
            break
    st.update(shard_docs=len(lens), stride=stride, kept=len(docs), chars=chars)
    return docs, dict(st)


def window_hashes(ids: np.ndarray, width: int = 32, stride: int = 1) -> np.ndarray:
    """64-bit hashes of every ``width``-token window (dot product with fixed random uint64 weights, wrapping)."""
    w = np.random.RandomState(1234).randint(1, 2**62, size=width, dtype=np.int64).astype(np.uint64)
    from numpy.lib.stride_tricks import sliding_window_view
    view = sliding_window_view(ids.astype(np.uint64), width)[::stride]
    out = np.empty(len(view), dtype=np.uint64)
    for s in range(0, len(view), 1_000_000):
        out[s:s + 1_000_000] = (view[s:s + 1_000_000] * w).sum(axis=1)
    return out


def sha256_file(path: str) -> str:
    return BPETokenizer.file_sha256(path)


def main(argv=None) -> Dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zip", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--vocab-size", type=int, default=8000)
    ap.add_argument("--scale", type=float, default=1.0, help="multiply every source budget (use <1 for a quick run)")
    ap.add_argument("--max-doc-chars", type=int, default=10_000)
    ap.add_argument("--min-doc-chars", type=int, default=200)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sources", nargs="*", default=list(TRAIN_BUDGET_MB))
    ap.add_argument("--no-control", action="store_true")
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    zf = zipfile.ZipFile(args.zip)

    # 1. sample
    dedupe = data.NearDuplicateFilter()
    per_source: Dict[str, List[str]] = {}
    manifest: Dict = {"args": vars(args), "sampling": {}}
    for s in args.sources:
        docs, st = sample_source(zf, s, int(TRAIN_BUDGET_MB[s] * 1e6 * args.scale), args.max_doc_chars, args.min_doc_chars, dedupe, args.seed)
        per_source[s], manifest["sampling"][s] = docs, st
        print(f"[prep] {s:11s} kept={st['kept']:5d} chars={st['chars']/1e6:5.2f}M stride={st['stride']:3d} "
              f"dup={st.get('dropped_duplicate', 0)} short={st.get('dropped_short', 0)} truncated={st.get('truncated', 0)}", flush=True)
    control: List[str] = []
    if not args.no_control:
        control, st = sample_source(zf, CONTROL_SOURCE, int(CONTROL_BUDGET_MB * 1e6 * args.scale), args.max_doc_chars, args.min_doc_chars, dedupe, args.seed)
        manifest["sampling"][CONTROL_SOURCE] = st
        print(f"[prep] {CONTROL_SOURCE:11s} kept={st['kept']:5d} chars={st['chars']/1e6:5.2f}M (CONTROL, not trained on)", flush=True)

    # 2. stratified document-level split
    train_docs: List[str] = []
    val_docs: Dict[str, List[str]] = {}
    for s, docs in per_source.items():
        tr, va = data.document_split(len(docs), args.val_frac, args.seed)
        train_docs += [docs[i] for i in tr]
        val_docs[s] = [docs[i] for i in va]
    n_val = sum(len(v) for v in val_docs.values())
    print(f"[prep] split: train_docs={len(train_docs)} val_docs={n_val}", flush=True)

    # 3. tokenizer on TRAIN documents only
    t1 = time.time()
    vocab = train_bpe(train_docs, args.vocab_size, min_count=2)
    vocab_path = os.path.join(args.out, "vocab.tok")
    n_vocab = write_tok(vocab, vocab_path)
    tok = BPETokenizer.from_file(vocab_path)
    assert tok.n_vocab == n_vocab
    print(f"[prep] tokenizer: {len(vocab)} tokens + EOS = n_vocab {n_vocab} in {time.time()-t1:.0f}s (train docs only)", flush=True)

    # 4. encode + checks
    t2 = time.time()
    fail_roundtrip = 0
    tok_stats: Dict[str, Dict] = {}

    def encode(docs: List[str], name: str) -> np.ndarray:
        nonlocal fail_roundtrip
        chunks, n_bytes, single = [], 0, 0
        for d in docs:
            ids = tok.encode(d, allowed_special=set())
            if tok.decode(ids) != d:
                fail_roundtrip += 1
            chunks.append(np.asarray(ids + [tok.eos_id], dtype=np.int64))
            n_bytes += len(d.encode("utf-8"))
            single += sum(1 for i in ids if i < 256)
        arr = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.int64)
        n_tok = len(arr) - len(docs)
        tok_stats[name] = {"docs": len(docs), "tokens": int(len(arr)), "bytes": n_bytes,
                           "bytes_per_token": round(n_bytes / max(n_tok, 1), 3),
                           "single_byte_token_pct": round(100 * single / max(n_tok, 1), 2)}
        return arr

    train_ids = encode(train_docs, "train")
    val_ids_by_source = {s: encode(v, f"val_{s}") for s, v in val_docs.items()}
    val_ids = np.concatenate(list(val_ids_by_source.values()))
    control_ids = encode(control, "control") if control else None
    print(f"[prep] encoded in {time.time()-t2:.0f}s; round-trip failures={fail_roundtrip}", flush=True)

    # 5. write .bin files
    def write(name, ids, split):
        p = os.path.join(args.out, name)
        data.write_bin(p, ids, tok.n_vocab, tok.eos_id, split)
        return p
    paths = {"train": write("train.bin", train_ids, data.SPLIT_TRAIN), "val": write("val.bin", val_ids, data.SPLIT_VAL)}
    for s, ids in val_ids_by_source.items():
        paths[f"val_{s}"] = write(f"val_{s}.bin", ids, data.SPLIT_VAL)
    if control_ids is not None:
        paths["control"] = write("control.bin", control_ids, data.SPLIT_VAL)

    # 6. integrity + leakage indicators
    all_ids = np.concatenate([train_ids, val_ids] + ([control_ids] if control_ids is not None else []))
    n_docs_total = len(train_docs) + n_val + len(control)
    train_h = np.unique(window_hashes(train_ids, 32, 1))
    val_h = window_hashes(val_ids, 32, 16)
    overlap = float(np.isin(val_h, train_h).mean()) if len(val_h) else 0.0
    checks = {
        "roundtrip_failures": fail_roundtrip,
        "max_id": int(all_ids.max()), "min_id": int(all_ids.min()),
        "ids_below_vocab": bool(all_ids.max() < tok.n_vocab), "ids_below_65536": bool(all_ids.max() < 65536),
        "eos_count_matches_docs": int((all_ids == tok.eos_id).sum()) == n_docs_total,
        "val_windows_seen_in_train_pct": round(100 * overlap, 3),
        "vocab_tokens_used_in_train_pct": round(100 * len(np.unique(train_ids)) / tok.n_vocab, 1),
    }
    manifest.update({"tokenization": tok_stats, "checks": checks, "vocab_sha256": sha256_file(vocab_path),
                     "files": {k: {"path": v, "sha256": sha256_file(v)} for k, v in paths.items()},
                     "n_vocab": tok.n_vocab, "eos_id": tok.eos_id, "seconds": round(time.time() - t0, 1)})
    with open(os.path.join(args.out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print("[prep] " + json.dumps(checks), flush=True)
    print(f"[prep] train tokens={tok_stats['train']['tokens']:,} val tokens={len(val_ids):,} "
          f"bytes/token={tok_stats['train']['bytes_per_token']}  total {time.time()-t0:.0f}s", flush=True)
    return manifest


if __name__ == "__main__":
    main()
