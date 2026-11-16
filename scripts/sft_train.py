#!/usr/bin/env python
"""Supervised fine-tuning of a pretrained checkpoint (full-parameter or LoRA). See ``src/llm/sft.py`` for the exact format and guarantees.

    # data first:  python scripts/sft_prepare.py --input raw/*.jsonl --vocab <tok> --out-dir sft/v1 --evals evals
    python scripts/sft_train.py --base 1B_pilot/checkpoints/ckpt_1b_full_best.pt --vocab tokenizers/vocab32k_v2/vocab32k.tok \
        --train sft/v1/train.jsonl --val sft/v1/val.jsonl --out sft/run_full --mode full --lr 1.5e-5 --epochs 2 \
        --base-sha256 e6f9b32bb058d24794691482a91371478a0bc11ef2ad02cc10505697eba7b5b2 \
        --retain-val math=bins/math.val.bin code=bins/code.val.bin

    python scripts/sft_train.py ... --out sft/run_lora --mode lora --lora-r 16 --lora-alpha 32 --lr 2e-4     # the LoRA arm

The base checkpoint is read only. Prefer the fp32 training checkpoint as ``--base`` (the weights-only bf16 export is rounded). Outputs in
``--out``: ``ckpt_best.pt`` (lowest validation loss), ``ckpt_last.pt``, optional ``ckpt_step_*.pt`` (``--save-interval``), ``sft_metrics.jsonl``
and ``sft_report.json`` (configuration, data counts, before/after validation losses, forgetting check, commit and hardware). Selecting a
checkpoint by validation loss alone is not enough: compare candidates with ``scripts/sft_generate.py`` and ``scripts/evaluate.py``.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from llm.sft import DEFAULT_END, SFTConfig, sft_train  # noqa: E402


def main(argv=None) -> int:
    d = SFTConfig()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True, help="pretrained checkpoint (fp32 training checkpoint preferred); never modified")
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--train", required=True, help="train.jsonl from sft_prepare.py")
    ap.add_argument("--val", help="val.jsonl from sft_prepare.py (strongly recommended)")
    ap.add_argument("--out", dest="out_dir", required=True)
    ap.add_argument("--mode", choices=["full", "lora"], default=d.mode)
    ap.add_argument("--lora-r", type=int, default=d.lora_r)
    ap.add_argument("--lora-alpha", type=float, default=d.lora_alpha)
    ap.add_argument("--lora-dropout", type=float, default=d.lora_dropout)
    for name in ("lr", "min_lr_ratio", "warmup_ratio", "weight_decay", "beta1", "beta2", "grad_clip", "epochs"):
        ap.add_argument("--" + name.replace("_", "-"), type=float, default=getattr(d, name))
    for name in ("batch_size", "grad_accum", "max_len", "eval_interval", "save_interval", "log_interval", "patience", "seed",
                 "retain_batches", "retain_seq_len"):
        ap.add_argument("--" + name.replace("_", "-"), type=int, default=getattr(d, name))
    ap.add_argument("--max-steps", type=int, default=None, help="overrides --epochs")
    ap.add_argument("--device", default=d.device)
    ap.add_argument("--dtype", default=d.dtype, choices=["auto", "fp32", "bf16"])
    ap.add_argument("--end-token", default=DEFAULT_END)
    ap.add_argument("--no-init-markers", dest="init_markers", action="store_false",
                    help="keep the role-marker embeddings as they are in the base (default: start them from the word embeddings)")
    ap.add_argument("--base-sha256", help="expected SHA-256 of --base; training refuses to start on a mismatch")
    ap.add_argument("--skip-hash", dest="hash_base", action="store_false", help="do not hash the base file (it is hashed once by default)")
    ap.add_argument("--retain-val", nargs="*", default=[], metavar="NAME=PATH",
                    help="pretraining validation .bin files; their loss is measured before and after (forgetting check)")
    a = ap.parse_args(argv)
    sft_train(SFTConfig(base=a.base, vocab=a.vocab, train=a.train, val=a.val, out_dir=a.out_dir, mode=a.mode, lora_r=a.lora_r,
                        lora_alpha=a.lora_alpha, lora_dropout=a.lora_dropout, lr=a.lr, min_lr_ratio=a.min_lr_ratio,
                        warmup_ratio=a.warmup_ratio, weight_decay=a.weight_decay, beta1=a.beta1, beta2=a.beta2, grad_clip=a.grad_clip,
                        epochs=a.epochs, max_steps=a.max_steps, batch_size=a.batch_size, grad_accum=a.grad_accum, max_len=a.max_len,
                        eval_interval=a.eval_interval, save_interval=a.save_interval, log_interval=a.log_interval, patience=a.patience,
                        seed=a.seed, device=a.device, dtype=a.dtype, end_token=a.end_token, init_markers=a.init_markers, base_sha256=a.base_sha256,
                        hash_base=a.hash_base, retain_val=a.retain_val, retain_batches=a.retain_batches, retain_seq_len=a.retain_seq_len))
    return 0


if __name__ == "__main__":
    sys.exit(main())
