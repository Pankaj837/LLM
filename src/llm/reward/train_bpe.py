"""Stand-in BPE vocabulary for the reward-model corpus (pure Python, no C++ needed).

    python -m llm.reward.train_bpe [--out data/vocab.tok] [--vocab-size 1722]

Thin wrapper over ``llm.bpe_trainer``; NOT the team's C++ vocabulary (see that module's docstring).
"""
from __future__ import annotations

import argparse

from ..bpe_trainer import train_bpe as _train, write_tok
from .preferences import corpus_for_tokenizer


def train_bpe(lines, vocab_size: int = 1722):
    """Merge tokens for ``lines``; min_count=1 because the reward corpus is tiny."""
    return _train(lines, vocab_size, min_count=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/vocab.tok")
    ap.add_argument("--vocab-size", type=int, default=1722)
    args = ap.parse_args()
    vocab = train_bpe(corpus_for_tokenizer(), args.vocab_size)
    n = write_tok(vocab, args.out)
    print(f"wrote {args.out}: {len(vocab)} tokens + specials = {n} ids")


if __name__ == "__main__":
    main()
