#!/usr/bin/env python
"""Throughput / memory / cost benchmark of the real training step on synthetic tokens (no dataset needed).

    python scripts/benchmark.py --presets 100m 300m 1p5b --batch-sizes 1 2 4 8 16 --seq-len 2048 --rupees-per-hour 330

For every preset and micro-batch size it builds the model, runs a few full optimizer steps (forward, loss, backward,
AdamW) in the same precision as ``llm.train`` and reports tokens/s, peak GPU memory, an MFU estimate and what that means in
GPU hours and rupees per 1B tokens. An out-of-memory result is reported (and larger batches of that preset are skipped), it
never aborts the sweep. Results go to ``--out`` as JSON. Spend GPU minutes on this first: it replaces the estimates in
``docs/TRAINING_PLAN_1p5B.md`` with measured numbers, and each row takes seconds.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from llm.model import TransformerModel  # noqa: E402
from llm.model_config import ModelConfig  # noqa: E402
from llm.train import PRESETS, TrainConfig, build_optimizer, resolve_device, resolve_dtype, run_info  # noqa: E402


def flops_per_token(n_params: int, cfg: ModelConfig, seq_len: int) -> float:
    """Training FLOPs per token: 6N for the matmuls plus the attention score/value products (12 * layers * seq * d)."""
    return 6.0 * n_params + 12.0 * cfg.num_layers * seq_len * cfg.hidden_dim


def is_oom(e: BaseException) -> bool:
    oom = getattr(torch.cuda, "OutOfMemoryError", ())
    return isinstance(e, oom) or "out of memory" in str(e).lower()


def bench_one(preset: str, batch: int, seq_len: int, steps: int, warmup: int, device: torch.device, amp, vocab: int) -> dict:
    cfg = ModelConfig(**{**PRESETS[preset], "vocab_size": vocab})
    if seq_len > cfg.max_position_embeddings:
        return {"preset": preset, "batch": batch, "status": f"seq_len {seq_len} > max_position_embeddings"}
    model = opt = None
    try:
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        model = TransformerModel(cfg).to(device)
        model.train()
        opt = build_optimizer(model, TrainConfig())
        n_params = sum(p.numel() for p in model.parameters())
        g = torch.Generator().manual_seed(0)
        t_start = None
        for i in range(steps):
            if i == warmup:
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                t_start = time.time()
            x = torch.randint(0, vocab, (batch, seq_len), generator=g).to(device)
            y = torch.randint(0, vocab, (batch, seq_len), generator=g).to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=amp, enabled=amp is not None):
                logits = model(x)
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), y.reshape(-1))
            if not torch.isfinite(loss):
                return {"preset": preset, "batch": batch, "status": "non-finite loss"}
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        dt = time.time() - t_start
        timed = steps - warmup
        tps = batch * seq_len * timed / dt
        peak = torch.cuda.max_memory_allocated(device) / 1e9 if device.type == "cuda" else None
        return {"preset": preset, "batch": batch, "status": "ok", "params_M": round(n_params / 1e6, 1), "tokens_per_s": round(tps),
                "step_s": round(dt / timed, 3), "peak_mem_GB": round(peak, 1) if peak is not None else None,
                "flops_per_token": flops_per_token(n_params, cfg, seq_len)}
    except Exception as e:  # noqa: BLE001
        if is_oom(e):
            return {"preset": preset, "batch": batch, "status": "OOM"}
        raise
    finally:
        del model, opt
        if device.type == "cuda":
            torch.cuda.empty_cache()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--presets", nargs="+", default=["100m", "300m", "1p5b"], choices=sorted(PRESETS))
    ap.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 2, 4, 8, 16])
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--steps", type=int, default=10, help="optimizer steps per row, including warm-up")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--vocab", type=int, default=32000)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="auto", choices=["auto", "fp32", "bf16"])
    ap.add_argument("--rupees-per-hour", type=float, default=330.0)
    ap.add_argument("--peak-tflops", type=float, default=989.0, help="dense bf16 peak of the GPU (H100 SXM = 989) for the MFU estimate")
    ap.add_argument("--out", help="write the results as JSON")
    a = ap.parse_args(argv)
    if a.steps <= a.warmup:
        ap.error("--steps must be larger than --warmup")
    dev = resolve_device(a.device)
    amp = resolve_dtype(a.dtype, dev)
    info = run_info()
    print(f"device={dev} gpu={info['gpu']} torch={info['torch']} cuda={info['cuda']} amp={amp} seq_len={a.seq_len} commit={str(info['git_commit'])[:10]}")
    rows = []
    print(f"{'preset':<7}{'batch':>6}{'tokens/s':>12}{'step s':>9}{'peak GB':>9}{'MFU':>7}{'h per 1B tok':>14}{'Rs per 1B tok':>15}  status")
    for preset in a.presets:
        for b in sorted(a.batch_sizes):
            r = bench_one(preset, b, a.seq_len, a.steps, a.warmup, dev, amp, a.vocab)
            if r["status"] == "ok":
                hours = 1e9 / r["tokens_per_s"] / 3600
                r.update(mfu=round(r["tokens_per_s"] * r["flops_per_token"] / (a.peak_tflops * 1e12), 3),
                         hours_per_1B_tokens=round(hours, 2), rupees_per_1B_tokens=round(hours * a.rupees_per_hour))
                print(f"{preset:<7}{b:>6}{r['tokens_per_s']:>12,}{r['step_s']:>9}{str(r['peak_mem_GB']):>9}{r['mfu']:>7.1%}"
                      f"{r['hours_per_1B_tokens']:>14}{r['rupees_per_1B_tokens']:>15,}  ok", flush=True)
            else:
                print(f"{preset:<7}{b:>6}{'':>12}{'':>9}{'':>9}{'':>7}{'':>14}{'':>15}  {r['status']}", flush=True)
            rows.append(r)
            if r["status"] == "OOM":
                break                                             # larger batches of this preset cannot fit either
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump({"run_info": info, "seq_len": a.seq_len, "rupees_per_hour": a.rupees_per_hour, "rows": rows}, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
