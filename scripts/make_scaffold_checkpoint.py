"""Create a deterministic, clearly-labelled UNTRAINED checkpoint for wiring / GPU smoke tests.

The delivered ``pretrain_model.pt`` is a structural scaffold with random weights (per the
integration report) but carries a misleading name and no metadata. Use this script instead:
the result is flagged ``untrained=True`` and the loader warns whenever it is used.

    python scripts/make_scaffold_checkpoint.py --out checkpoints/scaffold_untrained.pt
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from llm.checkpoint import save_checkpoint  # noqa: E402
from llm.model import TransformerModel  # noqa: E402
from llm.model_config import ModelConfig  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="checkpoints/scaffold_untrained.pt")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--vocab-size", type=int, default=8000)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    cfg = ModelConfig(vocab_size=args.vocab_size)
    model = TransformerModel(cfg)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    save_checkpoint(args.out, model, cfg, step=0, untrained=True, extra={"seed": args.seed, "note": "random init scaffold"})
    print(f"wrote {args.out} ({sum(p.numel() for p in model.parameters())/1e6:.2f}M params, untrained=True)")


if __name__ == "__main__":
    main()
