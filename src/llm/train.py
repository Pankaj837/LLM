"""Reference language-model trainer (next-token prediction on ``.bin`` token files).

    python -m llm.train --train-bin data/train.bin --val-bin data/val.bin --vocab data/vocab.tok \
        --out runs/base --preset base --max-steps 20000 --batch-size 32 --seq-len 512

Design choices (kept deliberately conventional so results are comparable across runs):
  * AdamW(betas=(0.9, 0.95)), weight decay only on >=2-D tensors (not on norm weights).
  * Linear warmup, then cosine decay to ``min_lr_ratio * lr``.
  * Global-norm gradient clipping (1.0); gradient accumulation for large effective batches.
  * bf16 autocast on CUDA when supported (parameters and optimizer state stay fp32).
  * Batches are a pure function of (seed, step, micro-step), so a resumed run reproduces
    the uninterrupted run exactly.
  * Checkpoints carry config, step, losses and the vocabulary hash (see ``checkpoint.py``).
Training fails fast on a non-finite loss.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F

from .checkpoint import load_checkpoint, read_checkpoint, save_checkpoint
from .data import TokenDataset
from .model import TransformerModel
from .model_config import ModelConfig
from .tokenizer import BPETokenizer

PRESETS: Dict[str, Dict[str, Any]] = {
    "base": {},   # ModelConfig() defaults: 51.5M parameters
    "small": dict(hidden_dim=256, num_query_heads=8, num_kv_heads=2, head_dim=32, intermediate_dim=704,
                  num_layers=6, max_position_embeddings=512, yarn_original_max_position=512),   # ~6M params
    "tiny": dict(hidden_dim=128, num_query_heads=4, num_kv_heads=2, head_dim=32, intermediate_dim=352,
                 num_layers=4, max_position_embeddings=512, yarn_original_max_position=512),
}


@dataclass
class TrainConfig:
    train_bin: str = ""
    val_bin: Optional[str] = None
    vocab: Optional[str] = None            # .tok file; only its hash is recorded
    out_dir: str = "runs/default"
    preset: str = "base"
    seq_len: int = 256
    batch_size: int = 16
    grad_accum: int = 1
    max_steps: int = 1000
    lr: float = 6e-4
    min_lr_ratio: float = 0.1
    warmup_steps: int = 100
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    eval_interval: int = 100
    eval_batches: int = 20
    save_interval: int = 500
    log_interval: int = 10
    seed: int = 0
    device: str = "auto"
    dtype: str = "auto"                    # auto | fp32 | bf16
    resume: Optional[str] = None
    stop_step: Optional[int] = None        # exit early (e.g. time-limited job) WITHOUT changing the LR schedule
    model_overrides: Dict[str, Any] = field(default_factory=dict)


def lr_at(step: int, cfg: TrainConfig) -> float:
    """Learning rate for optimizer step ``step`` (0-based)."""
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    if step >= cfg.max_steps:
        return cfg.lr * cfg.min_lr_ratio
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    return cfg.lr * (cfg.min_lr_ratio + (1 - cfg.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))


def build_optimizer(model: torch.nn.Module, cfg: TrainConfig) -> torch.optim.Optimizer:
    decay = [p for p in model.parameters() if p.requires_grad and p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.ndim < 2]
    groups = [{"params": decay, "weight_decay": cfg.weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=cfg.lr, betas=(cfg.beta1, cfg.beta2))


def resolve_device(name: str) -> torch.device:
    return torch.device("cuda" if name == "auto" and torch.cuda.is_available() else "cpu" if name == "auto" else name)


def resolve_dtype(name: str, device: torch.device) -> Optional[torch.dtype]:
    if name == "fp32":
        return None
    if name == "bf16" or (name == "auto" and device.type == "cuda" and torch.cuda.is_bf16_supported()):
        return torch.bfloat16
    return None


def loss_fn(model: TransformerModel, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    logits = model(x)
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), y.reshape(-1))


@torch.no_grad()
def evaluate(model: TransformerModel, ds: TokenDataset, batch_size: int, max_batches: int,
             device: torch.device, amp_dtype: Optional[torch.dtype]) -> Dict[str, float]:
    """Mean next-token cross-entropy over deterministic non-overlapping windows, and perplexity."""
    was_training = model.training
    model.eval()
    total, count = 0.0, 0
    for x, y in ds.sequential_batches(batch_size, max_batches, device):
        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            total += loss_fn(model, x, y).item() * x.numel()
        count += x.numel()
    model.train(was_training)
    loss = total / max(count, 1)
    return {"val_loss": loss, "val_ppl": math.exp(min(loss, 50.0))}


def train(cfg: TrainConfig) -> Dict[str, Any]:
    """Run training; returns ``{"history": [...], "final": {...}, "checkpoint": path}``."""
    if cfg.preset not in PRESETS:
        raise ValueError(f"unknown preset {cfg.preset!r}; choose from {sorted(PRESETS)}")
    os.makedirs(cfg.out_dir, exist_ok=True)
    device = resolve_device(cfg.device)
    amp_dtype = resolve_dtype(cfg.dtype, device)

    train_ds = TokenDataset(cfg.train_bin, cfg.seq_len)
    model_cfg = ModelConfig(**{**PRESETS[cfg.preset], "vocab_size": train_ds.header.vocab_size, **cfg.model_overrides})
    if cfg.seq_len > model_cfg.max_position_embeddings:
        raise ValueError(f"seq_len {cfg.seq_len} exceeds max_position_embeddings {model_cfg.max_position_embeddings}")
    val_ds = TokenDataset(cfg.val_bin, cfg.seq_len, model_cfg.vocab_size) if cfg.val_bin else None
    vocab_sha = BPETokenizer.file_sha256(cfg.vocab) if cfg.vocab else None

    torch.manual_seed(cfg.seed)
    model = TransformerModel(model_cfg).to(device)
    optimizer = build_optimizer(model, cfg)
    start_step = 0
    if cfg.resume:
        info = load_checkpoint(cfg.resume, model, expected_vocab_sha256=vocab_sha)
        _, _, _, raw = read_checkpoint(cfg.resume)
        if "optimizer" not in raw:
            raise ValueError("resume checkpoint has no optimizer state")
        optimizer.load_state_dict(raw["optimizer"])
        start_step = int(info["step"])
    model.train()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] device={device} amp={amp_dtype} params={n_params/1e6:.2f}M vocab={model_cfg.vocab_size} "
          f"train_tokens={train_ds.header.total_tokens:,} steps {start_step}->{cfg.max_steps}", flush=True)
    with open(os.path.join(cfg.out_dir, "train_config.json"), "w", encoding="utf-8") as f:
        json.dump({"train": asdict(cfg), "model": model_cfg.to_dict()}, f, indent=2)

    metrics_path = os.path.join(cfg.out_dir, "metrics.jsonl")
    history: List[Dict[str, Any]] = []
    best_val = float("inf")
    last_loss: Optional[float] = None
    t0, tokens_seen = time.time(), 0
    gen = torch.Generator()

    def checkpoint(name: str, step: int, val: Optional[float]) -> str:
        path = os.path.join(cfg.out_dir, name)
        save_checkpoint(path, model, model_cfg, step=step, train_loss=last_loss, val_loss=val,
                        vocab_sha256=vocab_sha, optimizer=optimizer, extra={"seed": cfg.seed, "preset": cfg.preset})
        return path

    end_step = min(cfg.max_steps, cfg.stop_step) if cfg.stop_step is not None else cfg.max_steps
    done = start_step
    for step in range(start_step, end_step):
        lr = lr_at(step, cfg)
        for g in optimizer.param_groups:
            g["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        for micro in range(cfg.grad_accum):
            gen.manual_seed(cfg.seed * 1_000_003 + step * cfg.grad_accum + micro)
            x, y = train_ds.get_batch(cfg.batch_size, gen, device)
            with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                loss = loss_fn(model, x, y) / cfg.grad_accum
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at step {step}: {loss.item()}")
            loss.backward()
            step_loss += loss.item()
            tokens_seen += x.numel()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip).item()
        optimizer.step()
        last_loss = step_loss

        done = step + 1
        record: Dict[str, Any] = {}
        if done % cfg.log_interval == 0 or done == end_step:
            record = {"step": done, "loss": step_loss, "lr": lr, "grad_norm": grad_norm,
                      "tokens_per_s": tokens_seen / max(time.time() - t0, 1e-9)}
        if val_ds is not None and (done % cfg.eval_interval == 0 or done == end_step):
            record.update({"step": done, **evaluate(model, val_ds, cfg.batch_size, cfg.eval_batches, device, amp_dtype)})
            if record["val_loss"] < best_val:
                best_val = record["val_loss"]
                checkpoint("ckpt_best.pt", done, best_val)
        if record:
            history.append(record)
            with open(metrics_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
            print("[train] " + " ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in record.items()), flush=True)
        if done % cfg.save_interval == 0 and done != end_step:
            checkpoint("ckpt_last.pt", done, None)

    final_path = checkpoint("ckpt_last.pt", done, best_val if best_val < float("inf") else None)
    return {"history": history, "final": history[-1] if history else {}, "checkpoint": final_path, "model_config": model_cfg}


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    defaults = TrainConfig()
    ap.add_argument("--train-bin", required=True)
    ap.add_argument("--val-bin")
    ap.add_argument("--vocab", help=".tok file (its sha256 is stored in checkpoints)")
    ap.add_argument("--out", dest="out_dir", default=defaults.out_dir)
    ap.add_argument("--preset", default=defaults.preset, choices=sorted(PRESETS))
    for name in ("seq_len", "batch_size", "grad_accum", "max_steps", "warmup_steps", "eval_interval", "eval_batches",
                 "save_interval", "log_interval", "seed"):
        ap.add_argument("--" + name.replace("_", "-"), type=int, default=getattr(defaults, name))
    for name in ("lr", "min_lr_ratio", "weight_decay", "grad_clip"):
        ap.add_argument("--" + name.replace("_", "-"), type=float, default=getattr(defaults, name))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="auto", choices=["auto", "fp32", "bf16"])
    ap.add_argument("--resume")
    ap.add_argument("--stop-step", type=int, default=None, help="exit early without changing the LR schedule")
    args = ap.parse_args(argv)
    train(TrainConfig(**{k: v for k, v in vars(args).items()}))


if __name__ == "__main__":
    main()
