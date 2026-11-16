"""Supervised fine-tuning (SFT) of a pretrained checkpoint: chat format, response-only loss, full or LoRA training.

    python scripts/sft_train.py --base ckpt_1b_full_best.pt --vocab tokenizers/vocab32k_v2/vocab32k.tok \
        --train sft/train.jsonl --val sft/val.jsonl --out sft/run_full --mode full --epochs 2

What this module guarantees (each point has a test in ``tests/test_sft.py``):
  * One chat format, built by one function (:func:`prompt_ids`) for training AND generation::

        <|system|> system text      (only when a system text is given)
        <|user|> prompt text
        <|assistant|> response text <|endoftext|>

    The role markers are the frozen tokenizer's real special tokens (``<|system|>`` 31939, ``<|user|>`` 31940,
    ``<|assistant|>`` 31941); there is no ``<|im_start|>``. Text is always encoded with ``allowed_special=set()``, so a
    special-token string inside an example is plain text and can never forge a boundary. The sequence ends with the
    pretraining end-of-document token ``<|endoftext|>`` (the one the model already knows and every generation script stops on);
    ``end_token`` can be changed.
  * Loss is computed on the response tokens and the end token only. Everything else is ``-100`` (ignored). The loss of a step
    is the mean over its supervised tokens (not over sequences), however the step is split into micro-batches.
  * An example longer than ``max_len`` is dropped and counted, never silently truncated (a cut response would teach the
    model to stop mid-sentence or never to stop).
  * ``mode="full"`` updates every weight; ``mode="lora"`` freezes the base and trains low-rank adapters on the attention and
    feed-forward projections. Checkpoints of both modes are ordinary full checkpoints (LoRA is merged on the way out), so
    ``scripts/evaluate.py``, ``sample_outputs.py``, ``export_weights.py`` and :func:`load_base_weights` read them unchanged.
  * The base checkpoint is verified (config, vocabulary hash, optional SHA-256) and is only ever read, never written.
  * Optimizer state is not saved: an SFT run is short (about an hour on one H100 for 20-50k examples); a failed run is repeated.
  * Training stops on a non-finite loss. The base model's loss on the validation responses and on pretraining validation files
    (``retain_val``: forgetting check) is measured before training and again at the end.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .checkpoint import remap_legacy_key, save_checkpoint
from .data import TokenDataset
from .model import TransformerModel
from .model_config import ModelConfig
from .tokenizer import BPETokenizer
from .train import evaluate as evaluate_bin
from .train import resolve_device, resolve_dtype, run_info

IGNORE = -100
ROLE_SYSTEM, ROLE_USER, ROLE_ASSISTANT = "<|system|>", "<|user|>", "<|assistant|>"
DEFAULT_END = "<|endoftext|>"
LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "W_gate", "W_up", "W_down")


# ------------------------------------------------------------------ format
def special_ids(tok: BPETokenizer, end_token: str = DEFAULT_END) -> Dict[str, int]:
    """Token ids of the chat markers and the end token. Raises if the tokenizer lacks one of them."""
    out: Dict[str, int] = {}
    for name in (ROLE_SYSTEM, ROLE_USER, ROLE_ASSISTANT, end_token):
        if name not in tok.special_to_id:
            raise ValueError(f"tokenizer has no special token {name!r}; SFT needs {ROLE_SYSTEM}, {ROLE_USER}, {ROLE_ASSISTANT} "
                             f"and the end token {end_token!r}")
        out[name] = tok.special_to_id[name]
    out["pad"] = tok.special_to_id.get("<|pad|>", tok.special_to_id[end_token])
    return out


def encode_plain(tok: BPETokenizer, text: str) -> List[int]:
    """Text as ordinary tokens: special-token strings inside it are NOT turned into special ids."""
    return tok.encode(text, allowed_special=set())


def prompt_ids(tok: BPETokenizer, prompt: str, system: Optional[str] = None, end_token: str = DEFAULT_END) -> List[int]:
    """Everything before the response: [system marker, system text,] user marker, prompt, assistant marker."""
    sp = special_ids(tok, end_token)
    ids: List[int] = []
    if system:
        ids += [sp[ROLE_SYSTEM]] + encode_plain(tok, system)
    ids += [sp[ROLE_USER]] + encode_plain(tok, prompt) + [sp[ROLE_ASSISTANT]]
    return ids


def encode_example(tok: BPETokenizer, ex: Dict[str, Any], max_len: int, end_token: str = DEFAULT_END) -> Optional[Dict[str, Any]]:
    """``{"x", "y"}`` for one example, or ``None`` when it is longer than ``max_len`` tokens.

    ``x`` is the sequence without its last token, ``y`` the sequence without its first token, with ``-100`` wherever the
    target is not a response token (so ``y[i]`` is supervised exactly when it is part of the response or the end token).
    """
    sp = special_ids(tok, end_token)
    p = prompt_ids(tok, ex["prompt"], ex.get("system"), end_token)
    r = encode_plain(tok, ex["response"])
    if not r:
        raise ValueError("empty response")
    ids = p + r + [sp[end_token]]
    if len(ids) - 1 > max_len:
        return None
    x, y = ids[:-1], ids[1:]
    for i in range(len(p) - 1):
        y[i] = IGNORE
    return {"x": x, "y": y, "n_prompt": len(p), "n_response": len(r) + 1}


class SFTData:
    """A tokenised SFT file: a list of ``(x, y)`` plus counts."""

    def __init__(self, rows: Sequence[Dict[str, Any]], tok: BPETokenizer, max_len: int, end_token: str = DEFAULT_END):
        self.items: List[Tuple[List[int], List[int]]] = []
        self.rows: List[Dict[str, Any]] = []
        self.dropped_too_long = 0
        self.supervised_tokens = 0
        self.total_tokens = 0
        for ex in rows:
            e = encode_example(tok, ex, max_len, end_token)
            if e is None:
                self.dropped_too_long += 1
                continue
            self.items.append((e["x"], e["y"]))
            self.rows.append(ex)
            self.supervised_tokens += sum(1 for t in e["y"] if t != IGNORE)
            self.total_tokens += len(e["x"])

    def __len__(self) -> int:
        return len(self.items)

    def stats(self) -> Dict[str, Any]:
        return {"examples": len(self), "dropped_too_long": self.dropped_too_long, "supervised_tokens": self.supervised_tokens,
                "total_tokens": self.total_tokens,
                "mean_tokens_per_example": round(self.total_tokens / max(len(self), 1), 1)}


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            ex = json.loads(line)
            if not isinstance(ex.get("prompt"), str) or not isinstance(ex.get("response"), str) or not ex["prompt"] or not ex["response"]:
                raise ValueError(f"{path}:{n}: each line needs non-empty string fields 'prompt' and 'response'")
            rows.append(ex)
    return rows


def collate(items: Sequence[Tuple[List[int], List[int]]], pad_id: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Right-pad to the longest sequence of the batch. Padding sits after the real tokens, so causal attention never sees it,
    and its labels are -100."""
    t = max(len(x) for x, _ in items)
    xb = torch.full((len(items), t), pad_id, dtype=torch.long)
    yb = torch.full((len(items), t), IGNORE, dtype=torch.long)
    for i, (x, y) in enumerate(items):
        xb[i, :len(x)] = torch.tensor(x, dtype=torch.long)
        yb[i, :len(y)] = torch.tensor(y, dtype=torch.long)
    return xb.to(device), yb.to(device)


def sum_loss(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Cross-entropy SUMMED over the supervised tokens (divide by their count outside)."""
    logits = model(x)
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), y.reshape(-1), ignore_index=IGNORE, reduction="sum")


@torch.no_grad()
def evaluate_sft(model: nn.Module, data: SFTData, batch_size: int, pad_id: int, device: torch.device,
                 amp_dtype: Optional[torch.dtype]) -> Dict[str, float]:
    """Mean cross-entropy per supervised (response) token over the whole file, in file order."""
    was = model.training
    model.eval()
    total, n = 0.0, 0
    for i in range(0, len(data), batch_size):
        chunk = data.items[i:i + batch_size]
        x, y = collate(chunk, pad_id, device)
        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            total += sum_loss(model, x, y).item()
        n += int((y != IGNORE).sum())
    model.train(was)
    loss = total / max(n, 1)
    return {"loss": loss, "ppl": math.exp(min(loss, 50.0)), "tokens": n}


# ------------------------------------------------------------------ LoRA
class LoRALinear(nn.Module):
    """``y = base(x) + (alpha / r) * B(A(dropout(x)))``; ``base`` is frozen, ``B`` starts at zero (the model starts unchanged)."""

    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        self.base = base
        self.r, self.scaling = r, alpha / r
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features, device=base.weight.device, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r, device=base.weight.device, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.scaling * F.linear(F.linear(self.dropout(x), self.lora_A), self.lora_B)

    def merged_weight(self) -> torch.Tensor:
        return self.base.weight.detach().float() + self.scaling * (self.lora_B.detach() @ self.lora_A.detach())


def apply_lora(model: nn.Module, r: int = 16, alpha: float = 32.0, dropout: float = 0.0,
               targets: Iterable[str] = LORA_TARGETS) -> Dict[str, int]:
    """Freeze every parameter and wrap the target ``nn.Linear`` layers with adapters. Returns trainable/total counts."""
    targets = tuple(targets)
    for p in model.parameters():
        p.requires_grad = False
    wrapped = 0
    for parent in list(model.modules()):
        for name, child in list(parent.named_children()):
            if name in targets and isinstance(child, nn.Linear):
                setattr(parent, name, LoRALinear(child, r, alpha, dropout))
                wrapped += 1
    if wrapped == 0:
        raise ValueError(f"no layer named {targets} found to adapt")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"wrapped_layers": wrapped, "trainable": trainable, "total": sum(p.numel() for p in model.parameters())}


def merged_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """The model's state dict on the CPU with every LoRA adapter folded into its base weight, under the ORIGINAL key names
    (``...q_proj.weight``), so the result loads into a plain :class:`TransformerModel`. A model without adapters is returned as is."""
    lora = {n: m for n, m in model.named_modules() if isinstance(m, LoRALinear)}
    out: Dict[str, torch.Tensor] = {}
    for k, v in model.state_dict().items():
        stem = k.rsplit(".", 1)[0] if "." in k else ""
        if k.endswith(".base.weight") and k[:-len(".base.weight")] in lora:
            name = k[:-len(".base.weight")]
            out[name + ".weight"] = lora[name].merged_weight().cpu()
        elif (k.endswith(".lora_A") or k.endswith(".lora_B")) and stem in lora:
            continue
        else:
            out[k] = v.detach().cpu()
    return out


class _StateShim:
    """``save_checkpoint`` only calls ``state_dict()``; this lets it write a merged (LoRA-free) state dict."""

    def __init__(self, sd: Dict[str, torch.Tensor]):
        self._sd = sd

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return self._sd


# ------------------------------------------------------------------ base model
def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def _open_raw(path: str) -> Dict[str, Any]:
    try:
        raw = torch.load(path, map_location="cpu", weights_only=True, mmap=True)     # no full copy in RAM
    except Exception:
        raw = torch.load(path, map_location="cpu", weights_only=True)
    if not (isinstance(raw, dict) and "format_version" in raw and "model" in raw and "config" in raw):
        raise ValueError(f"{path}: not a checkpoint written by llm.checkpoint.save_checkpoint")
    return raw


def load_base_model(path: str, device: torch.device, expected_vocab_sha256: Optional[str] = None) -> Tuple[TransformerModel, Dict[str, Any]]:
    """Build the model from the checkpoint's own config and load its weights (fp32 model; bf16 exports are widened).

    The file is only read. Raises if the stored vocabulary hash differs from ``expected_vocab_sha256``.
    """
    raw = _open_raw(path)
    cfg = ModelConfig.from_dict(raw["config"])
    meta = raw.get("meta", {})
    stored = meta.get("vocab_sha256")
    if expected_vocab_sha256 and stored and stored != expected_vocab_sha256:
        raise ValueError("checkpoint was trained with a different vocabulary file (sha256 mismatch)")
    state = {remap_legacy_key(k): v for k, v in raw["model"].items()}
    dtypes = sorted({str(v.dtype).replace("torch.", "") for v in state.values() if v.is_floating_point()})
    model = TransformerModel(cfg)
    model.load_state_dict(state, strict=True)
    model.to(device)
    info = {"config": cfg.to_dict(), "step": meta.get("step"), "weights_dtype": dtypes, "vocab_sha256": stored,
            "had_optimizer": "optimizer" in raw}
    return model, info


# ------------------------------------------------------------------ training
@dataclass
class SFTConfig:
    base: str = ""
    vocab: str = ""
    train: str = ""
    val: Optional[str] = None
    out_dir: str = "runs/sft"
    mode: str = "full"                     # full | lora
    lora_r: int = 16
    lora_alpha: float = 32.0
    lora_dropout: float = 0.0
    lr: float = 1.5e-5
    min_lr_ratio: float = 0.1
    warmup_ratio: float = 0.05
    weight_decay: float = 0.0
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    epochs: float = 2.0
    max_steps: Optional[int] = None        # overrides epochs
    batch_size: int = 4                    # sequences per micro-batch
    grad_accum: int = 16
    max_len: int = 2048
    eval_interval: int = 50
    save_interval: int = 0                 # 0 = only ckpt_best.pt and ckpt_last.pt; N = also ckpt_step_<N>.pt every N steps
    log_interval: int = 10
    patience: int = 0                      # stop after this many evaluations without a better val loss (0 = never)
    seed: int = 0
    device: str = "auto"
    dtype: str = "auto"
    end_token: str = DEFAULT_END
    base_sha256: Optional[str] = None      # expected SHA-256 of the base checkpoint file
    hash_base: bool = True                 # compute and record the SHA-256 of the base file (reads it once)
    retain_val: List[str] = field(default_factory=list)   # NAME=PATH pretraining validation .bin files (forgetting check)
    retain_batches: int = 20
    retain_seq_len: int = 512


def lr_at(step: int, total: int, warmup: int, lr: float, min_ratio: float) -> float:
    if step < warmup:
        return lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return lr * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0))))


class _Stream:
    """Endless, reproducible stream of example indices: epoch ``e`` is a shuffle seeded by ``(seed, e)``."""

    def __init__(self, n: int, seed: int):
        self.n, self.seed, self.epoch, self.order, self.pos = n, seed, 0, [], 0

    def take(self, k: int) -> List[int]:
        out: List[int] = []
        while len(out) < k:
            if self.pos >= len(self.order):
                self.order = list(range(self.n))
                random.Random(self.seed * 1_000_003 + self.epoch).shuffle(self.order)
                self.epoch += 1
                self.pos = 0
            out.append(self.order[self.pos])
            self.pos += 1
        return out


def _retain_losses(model, specs: List[str], cfg: SFTConfig, vocab_size: int, device, amp_dtype) -> Dict[str, float]:
    out = {}
    for spec in specs:
        name, sep, path = spec.partition("=")
        if not sep:
            raise ValueError(f"retain_val expects NAME=PATH, got {spec!r}")
        ds = TokenDataset(path, cfg.retain_seq_len, vocab_size)
        out[name] = evaluate_bin(model, ds, 4, cfg.retain_batches, device, amp_dtype)["val_loss"]
    return out


def sft_train(cfg: SFTConfig) -> Dict[str, Any]:
    """Run one SFT job; returns the report that is also written to ``<out_dir>/sft_report.json``."""
    if cfg.mode not in ("full", "lora"):
        raise ValueError("mode must be 'full' or 'lora'")
    device = resolve_device(cfg.device)
    amp_dtype = resolve_dtype(cfg.dtype, device)
    os.makedirs(cfg.out_dir, exist_ok=True)
    tok = BPETokenizer.from_file(cfg.vocab)
    vocab_sha = BPETokenizer.file_sha256(cfg.vocab)
    sp = special_ids(tok, cfg.end_token)

    base_sha = None
    if cfg.base_sha256 or cfg.hash_base:
        base_sha = sha256_file(cfg.base)
        if cfg.base_sha256 and base_sha != cfg.base_sha256.lower():
            raise ValueError(f"base checkpoint SHA-256 mismatch: file {base_sha}, expected {cfg.base_sha256}")
    torch.manual_seed(cfg.seed)
    model, base_info = load_base_model(cfg.base, device, vocab_sha)
    vocab_size = model.config.vocab_size
    if max(sp.values()) >= vocab_size:
        raise ValueError("a special token id is outside the model's vocabulary")
    if cfg.max_len > model.config.max_position_embeddings:
        raise ValueError(f"max_len {cfg.max_len} exceeds the model's context {model.config.max_position_embeddings}")
    if "bfloat16" in base_info["weights_dtype"]:
        print("[sft] WARNING: the base weights are bf16 (a weights-only export). Initial weights are rounded; start from the fp32 "
              "training checkpoint when exactness matters.", flush=True)

    lora_info = None
    if cfg.mode == "lora":
        lora_info = apply_lora(model, cfg.lora_r, cfg.lora_alpha, cfg.lora_dropout)
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_train_params = sum(p.numel() for p in trainable)
    model.train()

    train_data = SFTData(load_jsonl(cfg.train), tok, cfg.max_len, cfg.end_token)
    val_data = SFTData(load_jsonl(cfg.val), tok, cfg.max_len, cfg.end_token) if cfg.val else None
    if len(train_data) == 0:
        raise ValueError("no usable training examples")
    per_step = cfg.batch_size * cfg.grad_accum
    total_steps = cfg.max_steps or max(1, math.ceil(cfg.epochs * len(train_data) / per_step))
    warmup = max(1, round(cfg.warmup_ratio * total_steps))
    decay = [p for p in trainable if p.ndim >= 2]
    no_decay = [p for p in trainable if p.ndim < 2]
    optimizer = torch.optim.AdamW([{"params": decay, "weight_decay": cfg.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
                                  lr=cfg.lr, betas=(cfg.beta1, cfg.beta2))

    print(f"[sft] mode={cfg.mode} device={device} amp={amp_dtype} trainable={n_train_params/1e6:.2f}M "
          f"train_examples={len(train_data)} (dropped too long: {train_data.dropped_too_long}) supervised_tokens={train_data.supervised_tokens:,} "
          f"steps={total_steps} ({per_step} sequences per step)", flush=True)

    before: Dict[str, Any] = {}
    if val_data is not None:
        before["val"] = evaluate_sft(model, val_data, cfg.batch_size, sp["pad"], device, amp_dtype)
    if cfg.retain_val:
        before["retain"] = _retain_losses(model, cfg.retain_val, cfg, vocab_size, device, amp_dtype)
    print(f"[sft] before training: {before}", flush=True)

    sft_meta = {"base_checkpoint": os.path.basename(cfg.base), "base_sha256": base_sha, "base_step": base_info["step"], "mode": cfg.mode,
                "lora": {"r": cfg.lora_r, "alpha": cfg.lora_alpha, "dropout": cfg.lora_dropout} if cfg.mode == "lora" else None,
                "end_token": cfg.end_token, "special_ids": sp, "train_file": os.path.basename(cfg.train),
                "train_sha256": sha256_file(cfg.train), "epochs": cfg.epochs, "lr": cfg.lr, "seed": cfg.seed}

    def write(name: str, step: int, train_loss: Optional[float], val_loss: Optional[float]) -> str:
        path = os.path.join(cfg.out_dir, name)
        save_checkpoint(path, _StateShim(merged_state_dict(model)), model.config, step=step, train_loss=train_loss, val_loss=val_loss,
                        vocab_sha256=vocab_sha, extra={"sft": sft_meta})
        return path

    stream = _Stream(len(train_data), cfg.seed)
    metrics_path = os.path.join(cfg.out_dir, "sft_metrics.jsonl")
    history: List[Dict[str, Any]] = []
    best_val, bad_evals, last_loss, last_val = float("inf"), 0, None, None
    t0, tokens_seen = time.time(), 0
    done = 0
    stopped = None
    for step in range(total_steps):
        lr = lr_at(step, total_steps, warmup, cfg.lr, cfg.min_lr_ratio)
        for g in optimizer.param_groups:
            g["lr"] = lr
        idx = stream.take(per_step)
        n_sup = sum(sum(1 for t in train_data.items[i][1] if t != IGNORE) for i in idx)
        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        for m in range(cfg.grad_accum):
            chunk = [train_data.items[i] for i in idx[m * cfg.batch_size:(m + 1) * cfg.batch_size]]
            x, y = collate(chunk, sp["pad"], device)
            with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                loss = sum_loss(model, x, y) / n_sup                 # token mean over the whole step
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at step {step}: {loss.item()}")
            loss.backward()
            step_loss += loss.item()
            tokens_seen += int(x.numel())
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, cfg.grad_clip).item()
        optimizer.step()
        last_loss, done = step_loss, step + 1

        record: Dict[str, Any] = {}
        if done % cfg.log_interval == 0 or done == total_steps:
            record = {"step": done, "loss": step_loss, "lr": lr, "grad_norm": grad_norm,
                      "tokens_per_s": tokens_seen / max(time.time() - t0, 1e-9)}
            if device.type == "cuda":
                record["peak_mem_gb"] = torch.cuda.max_memory_allocated(device) / 1e9
        if val_data is not None and (done % cfg.eval_interval == 0 or done == total_steps):
            v = evaluate_sft(model, val_data, cfg.batch_size, sp["pad"], device, amp_dtype)
            record.update({"step": done, "val_loss": v["loss"], "val_ppl": v["ppl"]})
            last_val = v["loss"]
            if v["loss"] < best_val:
                best_val, bad_evals = v["loss"], 0
                write("ckpt_best.pt", done, last_loss, best_val)
            else:
                bad_evals += 1
        if cfg.save_interval and done % cfg.save_interval == 0 and done != total_steps:
            write(f"ckpt_step_{done:06d}.pt", done, last_loss, last_val)
        if record:
            history.append(record)
            with open(metrics_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
            print("[sft] " + " ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in record.items()), flush=True)
        if cfg.patience and val_data is not None and bad_evals >= cfg.patience and done < total_steps:
            stopped = f"early stop: no better validation loss for {bad_evals} evaluations"
            print(f"[sft] {stopped}", flush=True)
            break

    last_path = write("ckpt_last.pt", done, last_loss, last_val)
    after: Dict[str, Any] = {}
    if val_data is not None:
        after["val"] = evaluate_sft(model, val_data, cfg.batch_size, sp["pad"], device, amp_dtype)
    if cfg.retain_val:
        after["retain"] = _retain_losses(model, cfg.retain_val, cfg, vocab_size, device, amp_dtype)
    report = {"config": asdict(cfg), "base": {**base_info, "sha256": base_sha}, "lora": lora_info, "trainable_parameters": n_train_params,
              "train_data": train_data.stats(), "val_data": val_data.stats() if val_data else None,
              "steps_done": done, "steps_planned": total_steps, "stopped": stopped,
              "before": before, "after": after, "best_val_loss": None if best_val == float("inf") else best_val,
              "seconds": round(time.time() - t0, 1), "tokens_processed": tokens_seen, "final_checkpoint": last_path,
              "run_info": run_info()}
    with open(os.path.join(cfg.out_dir, "sft_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[sft] done: before {before}  after {after}", flush=True)
    return report


# ------------------------------------------------------------------ generation
def _banned_by_ngram(ids: List[int], n: int) -> set:
    """Tokens that would complete an n-gram already present in ``ids`` (no_repeat_ngram_size)."""
    if n < 2 or len(ids) < n - 1:
        return set()
    prefix = tuple(ids[len(ids) - (n - 1):])
    banned = set()
    for i in range(len(ids) - n + 1):
        if tuple(ids[i:i + n - 1]) == prefix:
            banned.add(ids[i + n - 1])
    return banned


@torch.no_grad()
def generate_ids(model: nn.Module, prompt: Sequence[int], *, max_new_tokens: int, eos_id: int, device: torch.device,
                 temperature: float = 0.0, top_k: int = 0, top_p: float = 1.0, repetition_penalty: float = 1.0,
                 no_repeat_ngram: int = 0, seed: int = 0, amp_dtype: Optional[torch.dtype] = None) -> Tuple[List[int], bool]:
    """Greedy (temperature 0) or sampled continuation of ``prompt``; returns ``(new token ids without the end token, finished)``.

    ``finished`` is True when the model emitted ``eos_id`` itself (as opposed to running into ``max_new_tokens``).
    Repetition penalty (as in the CTRL paper: logits of tokens already in the context are divided by the penalty when positive,
    multiplied when negative) and ``no_repeat_ngram`` are applied here so every decoding experiment goes through one code path.
    No KV cache: simple and exact, fine for the short answers of an evaluation.
    """
    ctx = model.config.max_position_embeddings
    if len(prompt) + max_new_tokens > ctx:
        raise ValueError(f"prompt ({len(prompt)}) + max_new_tokens ({max_new_tokens}) exceeds the context {ctx}")
    was = model.training
    model.eval()
    ids = list(prompt)
    gen = torch.Generator().manual_seed(seed)
    finished = False
    for _ in range(max_new_tokens):
        x = torch.tensor([ids], dtype=torch.long, device=device)
        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            logits = model(x)[0, -1].float()
        if repetition_penalty != 1.0:
            for t in set(ids):
                logits[t] = logits[t] / repetition_penalty if logits[t] > 0 else logits[t] * repetition_penalty
        for t in _banned_by_ngram(ids, no_repeat_ngram):
            logits[t] = float("-inf")
        if temperature <= 0:
            nxt = int(torch.argmax(logits))
        else:
            logits = logits / temperature
            if top_k and top_k > 0:
                kth = torch.topk(logits, min(top_k, logits.numel())).values[-1]
                logits = torch.where(logits < kth, torch.full_like(logits, float("-inf")), logits)
            probs = torch.softmax(logits, -1)
            if 0 < top_p < 1.0:
                sp, si = torch.sort(probs, descending=True)
                keep = torch.cumsum(sp, 0) - sp < top_p
                probs = torch.zeros_like(probs).scatter_(0, si[keep], sp[keep])
                probs = probs / probs.sum()
            nxt = int(torch.multinomial(probs.cpu(), 1, generator=gen))
        if nxt == eos_id:
            finished = True
            break
        ids.append(nxt)
    model.train(was)
    return ids[len(prompt):], finished
