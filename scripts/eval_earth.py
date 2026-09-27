"""Evaluate a trained checkpoint on the prepared Earth & Environment data.

    python scripts/eval_earth.py --run runs/earth_small/ckpt_last.pt --data data/earth --out runs/earth_small/eval.json

Decoding uses the dataset's own Hugging Face tokenizer (`tokenizer.json`); this repo's `BPETokenizer` is not
used here because the data was pre-tokenized with a different, incompatible vocabulary (see
`scripts/prepare_earth.py` and `docs/EARTH_ENVIRONMENT_SMOKE_TEST.md`).
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
from llm.checkpoint import load_checkpoint  # noqa: E402
from llm.model import TransformerModel  # noqa: E402
from llm.train import evaluate  # noqa: E402

PROMPTS = [
    "On 2001-06-01, the average land temperature in",
    "In 2010,",
    "The earthquake",
    "Soil moisture",
]


def unigram_loss(train_ids: np.ndarray, eval_ids: np.ndarray, vocab: int) -> float:
    counts = np.bincount(train_ids, minlength=vocab).astype(np.float64) + 1.0
    logp = np.log(counts / counts.sum())
    return float(-logp[eval_ids].mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--no-samples", action="store_true")
    args = ap.parse_args()

    train_cfg = json.load(open(os.path.join(os.path.dirname(args.run), "train_config.json"), encoding="utf-8"))
    seq_len = train_cfg["train"]["seq_len"]
    from llm.model_config import ModelConfig
    model_cfg = ModelConfig.from_dict(train_cfg["model"])
    model = TransformerModel(model_cfg).to(args.device)
    info = load_checkpoint(args.run, model, strict=True)

    val_ds = data.TokenDataset(os.path.join(args.data, "val.bin"), seq_len, model_cfg.vocab_size)
    dev = torch.device(args.device)
    val = evaluate(model, val_ds, args.batch_size, 10**9, dev, None)

    _, tr = data.read_tokens(os.path.join(args.data, "train.bin"), mmap=False)
    _, va = data.read_tokens(os.path.join(args.data, "val.bin"), mmap=False)
    uni = unigram_loss(np.asarray(tr, dtype=np.int64), np.asarray(va, dtype=np.int64), model_cfg.vocab_size)

    report = {"checkpoint": args.run, "step": info.get("step"), "params": sum(p.numel() for p in model.parameters()),
              "vocab": model_cfg.vocab_size, "seq_len": seq_len, "uniform_loss": math.log(model_cfg.vocab_size),
              "val_loss": val["val_loss"], "val_ppl": val["val_ppl"], "unigram_baseline_loss": uni,
              "beats_unigram_by_nats": uni - val["val_loss"]}

    if not args.no_samples:
        try:
            from tokenizers import Tokenizer
            hf = Tokenizer.from_file(os.path.join(args.data, "tokenizer.json"))
            model.eval()
            torch.manual_seed(0)
            samples = []
            for p in PROMPTS:
                ids = hf.encode(p).ids
                x = torch.tensor([ids])
                greedy = model.generate(x, max_new_tokens=40, temperature=0.0)[0].tolist()
                sampled = model.generate(x, max_new_tokens=40, temperature=0.8, top_k=40, top_p=0.9)[0].tolist()
                samples.append({"prompt": p, "greedy": hf.decode(greedy), "sampled": hf.decode(sampled)})
            report["samples"] = samples
        except Exception as e:  # pragma: no cover - best effort
            report["samples_error"] = str(e)

    out = json.dumps(report, indent=2, ensure_ascii=False)
    print(out)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(out)


if __name__ == "__main__":
    main()
