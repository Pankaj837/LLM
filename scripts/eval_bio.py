"""Evaluate a trained checkpoint on the prepared biology data.

    python scripts/eval_bio.py --run runs/bio_tiny/ckpt_last.pt --data data/bio --out runs/bio_tiny/eval.json

Reports (all on held-out DOCUMENTS the tokenizer and model never saw):
  * validation loss / perplexity, overall and per source;
  * bits-per-byte (tokenizer-independent, comparable across vocabularies and datasets):
        bpb = loss_nats_per_token * tokens / bytes / ln 2
  * baselines: uniform (ln V) and a unigram model fitted on the training tokens (add-one smoothing);
  * CONTROL loss on non-biology text (Project Gutenberg), which the model was not trained on;
  * a few sampled continuations for a human sanity check.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from llm import data  # noqa: E402
from llm.checkpoint import read_checkpoint  # noqa: E402
from llm.pipeline import LLMPipeline  # noqa: E402
from llm.train import evaluate  # noqa: E402

PROMPTS = [
    "The mitochondria are",
    "DNA replication begins when",
    "In this study, we investigated the role of",
    "Natural selection acts on",
]


def unigram_loss(train_ids: np.ndarray, eval_ids: np.ndarray, vocab: int) -> float:
    counts = np.bincount(train_ids, minlength=vocab).astype(np.float64) + 1.0
    logp = np.log(counts / counts.sum())
    return float(-logp[eval_ids].mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="checkpoint file")
    ap.add_argument("--data", required=True, help="directory written by prepare_biology.py")
    ap.add_argument("--out")
    ap.add_argument("--seq-len", type=int, default=None, help="default: the length used in training")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--no-samples", action="store_true")
    args = ap.parse_args()

    manifest = json.load(open(os.path.join(args.data, "manifest.json"), encoding="utf-8"))
    vocab_path = os.path.join(args.data, "vocab.tok")
    _, cfg, meta, _ = read_checkpoint(args.run)
    seq_len = args.seq_len or json.load(open(os.path.join(os.path.dirname(args.run), "train_config.json")))["train"]["seq_len"]
    pipe = LLMPipeline.from_pretrained(args.run, vocab_path, device=args.device)
    dev = torch.device(args.device)

    def loss_on(name: str):
        path = os.path.join(args.data, name)
        if not os.path.exists(path):
            return None
        ds = data.TokenDataset(path, seq_len, cfg.vocab_size)
        res = evaluate(pipe.model, ds, args.batch_size, 10**9, dev, None)
        tk = manifest["tokenization"]
        stats = tk.get(name.replace(".bin", ""))
        if stats is None and name == "val.bin":            # aggregate of the per-source validation sets
            parts = [v for k, v in tk.items() if k.startswith("val_")]
            stats = {"tokens": sum(v["tokens"] for v in parts), "docs": sum(v["docs"] for v in parts),
                     "bytes": sum(v["bytes"] for v in parts)}
        stats = stats or {}
        tokens = max(stats.get("tokens", 0) - stats.get("docs", 0), 1)
        bpb = res["val_loss"] * tokens / max(stats.get("bytes", 1), 1) / math.log(2) if stats else float("nan")
        return {"loss": res["val_loss"], "ppl": res["val_ppl"], "bits_per_byte": bpb, "tokens_evaluated": len(ds.tokens)}

    report = {"checkpoint": args.run, "step": meta.get("step"), "params": sum(p.numel() for p in pipe.model.parameters()),
              "vocab": cfg.vocab_size, "seq_len": seq_len, "uniform_loss": math.log(cfg.vocab_size)}
    report["val"] = loss_on("val.bin")
    report["val_by_source"] = {n[4:-4]: loss_on(n) for n in sorted(os.listdir(args.data)) if n.startswith("val_") and n.endswith(".bin")}
    report["control_non_biology"] = loss_on("control.bin")

    _, train_tok = data.read_tokens(os.path.join(args.data, "train.bin"), mmap=False)
    _, val_tok = data.read_tokens(os.path.join(args.data, "val.bin"), mmap=False)
    uni = unigram_loss(np.asarray(train_tok, dtype=np.int64), np.asarray(val_tok, dtype=np.int64), cfg.vocab_size)
    report["unigram_baseline_loss"] = uni
    v = report["val"]
    report["beats_unigram_by_nats"] = uni - v["loss"]
    if report["control_non_biology"]:
        report["control_minus_val_loss"] = report["control_non_biology"]["loss"] - v["loss"]

    if not args.no_samples:
        torch.manual_seed(0)
        report["samples"] = [{"prompt": p, "greedy": pipe.generate(p, max_new_tokens=50, temperature=0.0),
                              "sampled": pipe.generate(p, max_new_tokens=50, temperature=0.8, top_k=40, top_p=0.9)} for p in PROMPTS]

    out = json.dumps(report, indent=2, ensure_ascii=False)
    print(out)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(out)


if __name__ == "__main__":
    main()
