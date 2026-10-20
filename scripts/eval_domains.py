#!/usr/bin/env python
"""Per-domain held-out evaluation of a checkpoint: one validation ``.bin`` per domain -> loss, perplexity, tokens.

Typical use after ``scripts/tokenize_parquet.py --no-merge`` (it writes ``<shard>.val.bin`` per input file, so a
domain's validation set is just its own shard's file):

    python scripts/eval_domains.py --ckpt runs/1p5b/ckpt_last.pt --vocab vocab32k.tok \
        --val math=bins/maths.val.bin code=bins/code.val.bin physics=bins/physics.val.bin \
        --seq-len 2048 --batches 50 --device cuda --out runs/1p5b/eval_domains.json

Each ``name=path`` is evaluated on the same deterministic non-overlapping windows every time, so numbers from
different checkpoints are comparable. Exits non-zero if any loss is non-finite. A domain with fewer than one full
window is reported as ``windows: 0`` rather than silently skipped.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from llm.checkpoint import load_checkpoint, read_checkpoint  # noqa: E402
from llm.data import TokenDataset  # noqa: E402
from llm.model import TransformerModel  # noqa: E402
from llm.tokenizer import BPETokenizer  # noqa: E402
from llm.train import evaluate, resolve_device, resolve_dtype  # noqa: E402


def evaluate_domains(ckpt: str, vals: dict, seq_len: int, batch_size: int, batches: int, device: str = "auto",
                     dtype: str = "auto", vocab: str | None = None) -> dict:
    dev = resolve_device(device)
    amp = resolve_dtype(dtype, dev)
    _, cfg, meta, _ = read_checkpoint(ckpt)
    if cfg is None:
        raise ValueError("checkpoint has no embedded config; cannot rebuild the model")
    model = TransformerModel(cfg).to(dev)
    load_checkpoint(ckpt, model, expected_vocab_sha256=BPETokenizer.file_sha256(vocab) if vocab else None)
    model.eval()
    out = {"checkpoint": ckpt, "step": meta.get("step"), "seq_len": seq_len, "domains": {}}
    for name, path in vals.items():
        try:
            ds = TokenDataset(path, seq_len, cfg.vocab_size)
        except ValueError as e:
            if "too few" not in str(e):
                raise
            out["domains"][name] = {"windows": 0, "tokens": 0, "val_loss": None, "val_ppl": None}
            continue
        windows = len(range(0, len(ds), seq_len))
        r = evaluate(model, ds, batch_size, batches, dev, amp)
        used = min(windows, batch_size * batches)
        out["domains"][name] = {"windows": used, "tokens": used * seq_len, **r}
    losses = [d["val_loss"] for d in out["domains"].values() if d["val_loss"] is not None]
    out["mean_val_loss"] = sum(losses) / len(losses) if losses else None
    out["ok"] = all(math.isfinite(x) for x in losses)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--vocab", help=".tok file; its sha256 must match the one stored in the checkpoint")
    ap.add_argument("--val", nargs="+", required=True, metavar="NAME=PATH")
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--batches", type=int, default=50, help="max batches per domain")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="auto", choices=["auto", "fp32", "bf16"])
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    vals = {}
    for item in a.val:
        if "=" not in item:
            ap.error(f"--val expects NAME=PATH, got {item!r}")
        k, v = item.split("=", 1)
        vals[k] = v
    res = evaluate_domains(a.ckpt, vals, a.seq_len, a.batch_size, a.batches, a.device, a.dtype, a.vocab)
    print(f"checkpoint step {res['step']}")
    for k, d in res["domains"].items():
        if d["val_loss"] is None:
            print(f"  {k:<24} no full window (file too small)")
        else:
            print(f"  {k:<24} loss={d['val_loss']:.4f} ppl={d['val_ppl']:.2f} tokens={d['tokens']:,}")
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
