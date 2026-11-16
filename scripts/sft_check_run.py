#!/usr/bin/env python
"""Pass/fail check of one SFT run directory (the "gate" after the first short run on the real GPU, and after every full run).

    python scripts/sft_check_run.py --run sft/gate --base 1B_pilot/checkpoints/ckpt_1b_full_best.pt \
        --base-sha256 e6f9b32bb058d24794691482a91371478a0bc11ef2ad02cc10505697eba7b5b2 [--hash-base] \
        [--require-improvement] [--min-tokens-per-s 15000] [--max-mem-gb 75]

Checks (each printed as PASS / FAIL, exit code 1 when any FAIL):
  * ``sft_report.json`` and ``sft_metrics.jsonl`` exist; every logged loss, gradient norm and validation loss is finite
  * the report's base SHA-256 equals ``--base-sha256`` (and, with ``--hash-base``, the base file on disk still hashes to it: it was not touched)
  * ``ckpt_last.pt`` opens, all weights are finite, it has no optimizer state, its step equals the steps done, its configuration and
    vocabulary hash equal the base checkpoint's, and its recorded provenance names the base SHA-256
  * peak memory and throughput are inside the limits given (when recorded: CUDA runs only)
  * with ``--require-improvement``: the validation loss after training is lower than before it
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import List, Tuple

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from llm.checkpoint import read_checkpoint  # noqa: E402
from llm.sft import sha256_file  # noqa: E402


def _finite(x) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x)


def check_run(run: str, base: str, base_sha256: str = None, hash_base: bool = False, require_improvement: bool = False,
              min_tokens_per_s: float = 0.0, max_mem_gb: float = 0.0) -> List[Tuple[bool, str]]:
    out: List[Tuple[bool, str]] = []

    def add(ok: bool, msg: str) -> None:
        out.append((bool(ok), msg))

    rep_path, met_path, ck_path = (os.path.join(run, n) for n in ("sft_report.json", "sft_metrics.jsonl", "ckpt_last.pt"))
    for p in (rep_path, met_path, ck_path):
        add(os.path.isfile(p), f"{os.path.basename(p)} exists")
    if not all(ok for ok, _ in out):
        return out
    rep = json.load(open(rep_path, encoding="utf-8"))
    recs = [json.loads(line) for line in open(met_path, encoding="utf-8") if line.strip()]
    vals = [r[k] for r in recs for k in ("loss", "grad_norm", "val_loss") if k in r]
    add(recs and all(_finite(v) for v in vals), f"all {len(vals)} logged losses / gradient norms are finite")
    add(rep["steps_done"] >= 1, f"{rep['steps_done']} of {rep['steps_planned']} steps done" + (f" ({rep['stopped']})" if rep.get("stopped") else ""))

    if base_sha256:
        add(rep["base"].get("sha256") == base_sha256.lower(), "report's base SHA-256 equals the expected one")
    if hash_base:
        add(sha256_file(base) == (base_sha256 or rep["base"].get("sha256")), "base file on disk still has that SHA-256 (not modified)")

    state, cfg, meta, raw = read_checkpoint(ck_path)
    add("optimizer" not in raw, "ckpt_last.pt has no optimizer state")
    add(all(torch.isfinite(v).all() for v in state.values() if v.is_floating_point()), "all weights in ckpt_last.pt are finite")
    add(meta.get("step") == rep["steps_done"], f"checkpoint step {meta.get('step')} equals steps done {rep['steps_done']}")
    try:
        braw = torch.load(base, map_location="cpu", weights_only=True, mmap=True)
    except Exception:
        braw = torch.load(base, map_location="cpu", weights_only=True)
    add(braw["config"] == raw["config"], "configuration equals the base checkpoint's")
    add(braw.get("meta", {}).get("vocab_sha256") == meta.get("vocab_sha256"), "vocabulary hash equals the base checkpoint's")
    sft = (meta.get("extra") or {}).get("sft") or {}
    add(sft.get("base_sha256") == rep["base"].get("sha256") and sft.get("base_sha256") is not None, "checkpoint provenance names the base SHA-256")

    mem = [r["peak_mem_gb"] for r in recs if "peak_mem_gb" in r]
    if max_mem_gb and mem:
        add(max(mem) <= max_mem_gb, f"peak GPU memory {max(mem):.1f} GB <= {max_mem_gb} GB")
    tps = [r["tokens_per_s"] for r in recs if "tokens_per_s" in r]
    if min_tokens_per_s and tps:
        add(tps[-1] >= min_tokens_per_s, f"throughput {tps[-1]:,.0f} tokens/s >= {min_tokens_per_s:,.0f}")
    if require_improvement:
        b, a = (rep.get("before") or {}).get("val"), (rep.get("after") or {}).get("val")
        add(bool(b and a and a["loss"] < b["loss"]), f"validation loss {b and round(b['loss'], 4)} -> {a and round(a['loss'], 4)} improved")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--base-sha256")
    ap.add_argument("--hash-base", action="store_true")
    ap.add_argument("--require-improvement", action="store_true")
    ap.add_argument("--min-tokens-per-s", type=float, default=0.0)
    ap.add_argument("--max-mem-gb", type=float, default=0.0)
    a = ap.parse_args(argv)
    results = check_run(a.run, a.base, a.base_sha256, a.hash_base, a.require_improvement, a.min_tokens_per_s, a.max_mem_gb)
    for ok, msg in results:
        print(("PASS  " if ok else "FAIL  ") + msg)
    failed = sum(1 for ok, _ in results if not ok)
    print(f"RESULT: {'PASS' if not failed else 'FAIL'} ({len(results) - failed} passed, {failed} failed)")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
