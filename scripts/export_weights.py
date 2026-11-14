#!/usr/bin/env python
"""Slim a training checkpoint down to the model weights only (for evaluation or a slow download).

    python scripts/export_weights.py --ckpt run/ckpt_best.pt --out ckpt_best_weights_bf16.pt            # bf16, about 3 GB for 1.5B
    python scripts/export_weights.py --ckpt run/ckpt_best.pt --out ckpt_best_weights_fp32.pt --dtype fp32  # exact weights, about 6 GB

A training checkpoint of the 1.5B model is 18.1 GB: the weights (fp32, about 6 GB) plus the two AdamW moments (about 12 GB).
Evaluation (scripts/evaluate.py, eval_science.py, eval_domains.py, sample_outputs.py) only needs the weights, config and
metadata. The slim file keeps the same format (``format_version``, ``model``, ``config``, ``meta``) so ``load_checkpoint``
and every script that takes ``--ckpt`` read it unchanged. It cannot resume training (no optimizer state): keep the full
checkpoint for that. ``--dtype bf16`` rounds the weights (relative error about 0.4%), which is what the bf16 forward pass
already does; use ``--dtype fp32`` when the exact weights are needed.

The output is written atomically, read back, and checked (same keys and shapes, values equal to the source within the
rounding of the chosen dtype); the exit code is non-zero if anything differs. The SHA-256 of the output is printed.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from llm.checkpoint import FORMAT_VERSION  # noqa: E402

DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32}


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def _load(path: str) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=True, mmap=True)   # no full copy in RAM
    except Exception:
        return torch.load(path, map_location="cpu", weights_only=True)


def export(ckpt: str, out: str, dtype: str = "bf16") -> dict:
    """Write the weights-only copy of ``ckpt`` to ``out`` and return a small report. Raises ValueError on any mismatch."""
    if dtype not in DTYPES:
        raise ValueError(f"dtype must be one of {sorted(DTYPES)}")
    target = DTYPES[dtype]
    raw = _load(ckpt)
    if not (isinstance(raw, dict) and "format_version" in raw and "model" in raw and "config" in raw):
        raise ValueError("not a checkpoint written by llm.checkpoint.save_checkpoint (no format_version/model/config)")
    if raw["format_version"] > FORMAT_VERSION:
        raise ValueError(f"checkpoint format {raw['format_version']} is newer than supported ({FORMAT_VERSION})")

    model = {k: (v.detach().to(target) if v.is_floating_point() else v.detach().clone()) for k, v in raw["model"].items()}
    meta = dict(raw.get("meta", {}))
    meta["exported"] = {"dtype": dtype, "source_file": os.path.basename(ckpt), "source_bytes": os.path.getsize(ckpt),
                        "source_step": meta.get("step"), "optimizer_dropped": "optimizer" in raw}
    payload = {"format_version": raw["format_version"], "model": model, "config": raw["config"], "meta": meta}

    tmp = f"{out}.tmp"
    torch.save(payload, tmp)
    os.replace(tmp, out)

    # read back and compare with the source
    back = torch.load(out, map_location="cpu", weights_only=True)
    if "optimizer" in back:
        raise ValueError("optimizer state is still in the output")
    if set(back["model"]) != set(raw["model"]):
        raise ValueError("parameter names differ between source and output")
    worst = 0.0
    for k, src in raw["model"].items():
        dst = back["model"][k]
        if tuple(dst.shape) != tuple(src.shape):
            raise ValueError(f"shape differs for {k}")
        if src.is_floating_point():
            if dst.dtype != target:
                raise ValueError(f"dtype differs for {k}: {dst.dtype}")
            ref = src.detach().to(target).float()
            diff = (dst.float() - ref).abs().max().item()
            worst = max(worst, diff)
            if diff != 0.0:
                raise ValueError(f"values differ for {k} (max abs {diff})")
        elif not torch.equal(dst, src):
            raise ValueError(f"values differ for {k}")
    if back["config"] != raw["config"]:
        raise ValueError("config differs")
    return {"out": out, "bytes": os.path.getsize(out), "source_bytes": os.path.getsize(ckpt), "step": meta.get("step"),
            "params": sum(v.numel() for v in back["model"].values()), "dtype": dtype, "sha256": sha256(out)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="full training checkpoint (ckpt_best.pt, ckpt_250m.pt, ...)")
    ap.add_argument("--out", required=True, help="slim output file")
    ap.add_argument("--dtype", choices=sorted(DTYPES), default="bf16")
    a = ap.parse_args(argv)
    if os.path.abspath(a.ckpt) == os.path.abspath(a.out):
        print("refusing to overwrite the source checkpoint", file=sys.stderr)
        return 2
    try:
        r = export(a.ckpt, a.out, a.dtype)
    except ValueError as e:
        print(f"FAILED: {e}", file=sys.stderr)
        return 1
    print(f"wrote {r['out']}: {r['bytes'] / 1e9:.2f} GB (source {r['source_bytes'] / 1e9:.2f} GB), step {r['step']}, "
          f"{r['params'] / 1e6:.1f}M parameters, {r['dtype']}")
    print(f"sha256 {r['sha256']}  {os.path.basename(r['out'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
