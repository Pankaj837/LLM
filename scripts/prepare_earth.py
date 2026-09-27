"""Adapt the pre-processed "Earth and Environment" dataset into this repo's `.bin` format and verify it.

That dataset ships its OWN `train.bin` / `val.bin`, already tokenized with a Hugging-Face byte-level BPE
tokenizer (vocab 8192) — but with a **different binary header** than the tokenizer team's spec used elsewhere
in this repo:

    Earth/Env header (24 bytes, magic b"GNRP"):  <IIIIQ>  magic, version, vocab_size, eos_id, total_tokens
    This repo's header (36 bytes, magic b"TOK1"): <IIIIQIQ> magic, version, vocab_size, eos_id, total_tokens, split, reserved

Same information, incompatible byte layout — `llm.data.read_header` cannot open the delivered files directly.
This script reads the delivered format, independently re-verifies its own claims, and re-writes the token
stream through `llm.data.write_bin` so the rest of this repo's tooling (`llm.train`, `scripts/eval_bio.py`-style
evaluation) can use it unchanged. No re-tokenization happens: the ids are copied as-is.

    python scripts/prepare_earth.py --zip "Earth and Environment Processed Dataset.zip" --out data/earth
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
import sys
import zipfile
from typing import Tuple

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from llm import data  # noqa: E402

SRC_HEADER_FMT = "<IIIIQ"          # magic, version, vocab_size, eos_id, total_tokens
SRC_HEADER_SIZE = struct.calcsize(SRC_HEADER_FMT)
SRC_MAGIC = b"GNRP"


def read_source_bin(raw: bytes) -> Tuple[dict, np.ndarray]:
    if len(raw) < SRC_HEADER_SIZE:
        raise ValueError("file shorter than the 24-byte header")
    magic_b, version, vocab_size, eos_id, total_tokens = struct.unpack(SRC_HEADER_FMT, raw[:SRC_HEADER_SIZE])
    magic = magic_b.to_bytes(4, "little")
    if magic != SRC_MAGIC:
        raise ValueError(f"unexpected magic {magic!r} (expected {SRC_MAGIC!r})")
    expected = SRC_HEADER_SIZE + 2 * total_tokens
    if len(raw) != expected:
        raise ValueError(f"size {len(raw)} != header-declared {expected}")
    ids = np.frombuffer(raw, dtype="<u2", count=total_tokens, offset=SRC_HEADER_SIZE).astype(np.int64)
    return {"version": version, "vocab_size": vocab_size, "eos_id": eos_id, "total_tokens": int(total_tokens)}, ids


def window_hashes(ids: np.ndarray, width: int = 32, stride: int = 1) -> np.ndarray:
    from numpy.lib.stride_tricks import sliding_window_view
    if len(ids) < width:
        return np.zeros(0, dtype=np.uint64)
    w = np.random.RandomState(1234).randint(1, 2**62, size=width, dtype=np.int64).astype(np.uint64)
    view = sliding_window_view(ids.astype(np.uint64), width)[::stride]
    out = np.empty(len(view), dtype=np.uint64)
    for s in range(0, len(view), 1_000_000):
        out[s:s + 1_000_000] = (view[s:s + 1_000_000] * w).sum(axis=1)
    return out


def main() -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zip", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sample-docs", type=int, default=3)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    zf = zipfile.ZipFile(args.zip)
    names = {os.path.basename(n): n for n in zf.namelist()}
    root = next(n for n in zf.namelist() if n.endswith("dataset/train.bin"))[: -len("dataset/train.bin")]

    manifest: dict = {"source_zip": args.zip, "checks": {}}
    written = {}
    for split_name, split_id in (("train", data.SPLIT_TRAIN), ("val", data.SPLIT_VAL)):
        raw = zf.read(f"{root}dataset/{split_name}.bin")
        hdr, ids = read_source_bin(raw)
        eos_count = int((ids == hdr["eos_id"]).sum())
        checks = {
            "declared_total_tokens": hdr["total_tokens"], "actual_tokens": len(ids),
            "file_size_matches_header": len(raw) == SRC_HEADER_SIZE + 2 * hdr["total_tokens"],
            "ids_below_vocab": bool(len(ids) == 0 or ids.max() < hdr["vocab_size"]),
            "ids_below_65536": bool(len(ids) == 0 or ids.max() < 65536),
            "ids_non_negative": bool(len(ids) == 0 or ids.min() >= 0),
            "eos_count": eos_count,
        }
        manifest["checks"][split_name] = checks
        out_path = os.path.join(args.out, f"{split_name}.bin")
        data.write_bin(out_path, ids, hdr["vocab_size"], hdr["eos_id"], split_id)
        written[split_name] = {"path": out_path, "vocab_size": hdr["vocab_size"], "eos_id": hdr["eos_id"],
                               "tokens": len(ids), "documents": eos_count}
        print(f"[earth] {split_name:5s} tokens={len(ids):9,d} docs(eos)={eos_count:6d} checks={checks}", flush=True)

    assert written["train"]["vocab_size"] == written["val"]["vocab_size"], "train/val vocab_size mismatch"
    vocab_size = written["train"]["vocab_size"]

    _, tr_ids = data.read_tokens(written["train"]["path"], mmap=False)
    _, va_ids = data.read_tokens(written["val"]["path"], mmap=False)
    tr_ids, va_ids = np.asarray(tr_ids, dtype=np.int64), np.asarray(va_ids, dtype=np.int64)
    train_h = np.unique(window_hashes(tr_ids, 32, 1))
    val_h = window_hashes(va_ids, 32, 8)
    overlap = float(np.isin(val_h, train_h).mean()) if len(val_h) else 0.0
    manifest["val_windows_seen_in_train_pct"] = round(100 * overlap, 3)
    print(f"[earth] val/train 32-token window overlap: {manifest['val_windows_seen_in_train_pct']}%", flush=True)

    # decode round-trip with the delivered Hugging Face tokenizer (independent re-check of their own claim)
    extracted = zf.extract(names.get("tokenizer.json", f"{root}tokenizer/tokenizer.json"), args.out)
    tok_path = os.path.join(args.out, "tokenizer.json")   # flat, predictable location for eval_earth.py
    shutil.copy(extracted, tok_path)
    try:
        from tokenizers import Tokenizer
        hf = Tokenizer.from_file(tok_path)
        eos = written["train"]["eos_id"]
        doc_starts = np.where(np.concatenate([[True], tr_ids[:-1] == eos]))[0][: args.sample_docs]
        samples = []
        for s in doc_starts:
            e = s + list(tr_ids[s:s + 400]).index(eos) if eos in tr_ids[s:s + 400].tolist() else s + 60
            text = hf.decode(tr_ids[s:e].tolist())
            samples.append(text)
            print(f"[earth] sample doc: {text[:160]!r}", flush=True)
        manifest["decode_check"] = {"ok": True, "samples": samples}
    except Exception as e:  # pragma: no cover - best-effort diagnostic only
        manifest["decode_check"] = {"ok": False, "error": str(e)}
        print(f"[earth] decode check skipped: {e}", flush=True)

    manifest["vocab_size"] = vocab_size
    manifest["eos_id"] = written["train"]["eos_id"]
    manifest["files"] = written
    manifest["source_summary"] = None
    summary_name = names.get("PROCESSING_SUMMARY.md")
    if summary_name:
        text = zf.read(summary_name).decode("utf-8", "replace")
        for line in text.splitlines():
            if "Verbalized documents" in line or "Train tokens" in line or "Val tokens" in line:
                (manifest.setdefault("source_claims", [])).append(line.strip())

    with open(os.path.join(args.out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print("[earth] " + json.dumps({k: v for k, v in manifest.items() if k not in ("decode_check", "files")}))
    return manifest


if __name__ == "__main__":
    main()
