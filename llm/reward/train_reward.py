"""Train and evaluate the Conditional Reward Model.

    python -m llm.reward.train_reward                    # FiLM conditioning (default)
    python -m llm.reward.train_reward --conditioning concat
    python -m llm.reward.train_reward --ablate-condition # control: condition input shuffled

The third mode is the important one for the writeup. If accuracy barely drops
when the condition is randomised, the model was never using it, and the
"conditional" part of the name is decorative.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn as nn

from ..tokenizer import BPETokenizer
from .preferences import (CONDITIONS, COND_TO_ID, Response, build_dataset,
                          condition_blind_ceiling, prefers, split,
                          surface_heuristic_accuracy, swap_pair_for)
from .reward_model import ConditionalRewardModel, preference_loss
from .transformer import TransformerConfig

PAD_ID = 0


# ------------------------------------------------------------------ encoding

def encode_pair(tok: BPETokenizer, prompt: str, response: str, max_len: int):
    """Layout: prompt <eos> response <eos>

    The trailing EOS matters: pooling reads the final real token, so every
    example needs a consistent terminator to pool from. Without it the pooled
    vector comes from whatever word happened to end the response.
    """
    ids = tok.encode(prompt) + [tok.eos_id] + tok.encode(response) + [tok.eos_id]
    if len(ids) > max_len:
        # Truncate from the left: the response end is what we pool from, so it
        # is the part we cannot afford to lose.
        ids = ids[-max_len:]
    return ids


def collate(batch_ids, max_len, device):
    n = len(batch_ids)
    T = max(len(x) for x in batch_ids)
    input_ids = torch.full((n, T), PAD_ID, dtype=torch.long)
    attn = torch.zeros((n, T), dtype=torch.long)
    for i, ids in enumerate(batch_ids):
        input_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        attn[i, : len(ids)] = 1
    return input_ids.to(device), attn.to(device)


class Batcher:
    def __init__(self, pairs, tok, max_len, device):
        self.data = [
            (
                encode_pair(tok, p.prompt, p.chosen, max_len),
                encode_pair(tok, p.prompt, p.rejected, max_len),
                COND_TO_ID[p.condition],
                p.conflicting,
            )
            for p in pairs
        ]
        self.max_len, self.device = max_len, device

    def __len__(self):
        return len(self.data)

    def batches(self, batch_size, shuffle=True, rng=None):
        idx = list(range(len(self.data)))
        if shuffle:
            (rng or random).shuffle(idx)
        for s in range(0, len(idx), batch_size):
            chunk = [self.data[i] for i in idx[s : s + batch_size]]
            ch, rj = collate([c[0] for c in chunk], self.max_len, self.device), \
                     collate([c[1] for c in chunk], self.max_len, self.device)
            cond = torch.tensor([c[2] for c in chunk], dtype=torch.long, device=self.device)
            conflicting = torch.tensor([c[3] for c in chunk], dtype=torch.bool)
            yield ch, rj, cond, conflicting


# -------------------------------------------------------------------- eval

@torch.no_grad()
def evaluate(model, batcher, batch_size, shuffle_conditions=False):
    model.eval()
    n = correct = 0
    n_conf = correct_conf = 0
    margins = []
    for (cx, cm), (rx, rm), cond, conflicting in batcher.batches(batch_size, shuffle=False):
        if shuffle_conditions:
            cond = cond[torch.randperm(cond.size(0), device=cond.device)]
        r_c = model(cx, cm, cond)
        r_r = model(rx, rm, cond)
        ok = (r_c > r_r).cpu()
        correct += ok.sum().item()
        n += ok.numel()
        correct_conf += ok[conflicting].sum().item()
        n_conf += conflicting.sum().item()
        margins.append((r_c - r_r).cpu())
    model.train()
    return {
        "acc": correct / max(n, 1),
        "acc_conflicting": correct_conf / max(n_conf, 1),
        "mean_margin": torch.cat(margins).mean().item() if margins else 0.0,
        "n": n,
    }


@torch.no_grad()
def condition_swap_test(model, tok, max_len, device, prompt, detailed, brief):
    """The headline result.

    Take one pair (detailed vs brief) and score it under every condition from a
    single backbone pass. A working CRM ranks the detailed response above the
    brief one under `helpful` and below it under `concise`. A model that has
    collapsed to a single global notion of quality gives the same ordering for
    both, whatever its overall accuracy says.

    The pair must come from a HELD-OUT prompt (see main), otherwise the test only
    shows the model memorised a training example. Conditions that do not
    distinguish the two responses (e.g. `formal`, when both are casual) are
    reported as "n/a" instead of an arbitrary winner.
    """
    ids = [encode_pair(tok, prompt, r, max_len) for r in (detailed, brief)]
    x, m = collate(ids, max_len, device)
    scores = model.score_all_conditions(x, m)      # [2, n_cond]

    rows = []
    for ci, cname in enumerate(CONDITIONS):
        s_det, s_brief = scores[0, ci].item(), scores[1, ci].item()
        applicable = prefers(cname, Response(detailed, 3, 0), Response(brief, 1, 0)) is not None
        rows.append({
            "condition": cname,
            "score_detailed": round(s_det, 3),
            "score_brief": round(s_brief, 3),
            "winner": ("detailed" if s_det > s_brief else "brief") if applicable else "n/a",
        })
    flipped = rows[COND_TO_ID["helpful"]]["winner"] != rows[COND_TO_ID["concise"]]["winner"]
    return rows, flipped


# -------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", default="data/vocab.tok",
                    help="BPE vocab. Use the team's vocab.tok, or generate a stand-in with "
                         "`python -m llm.reward.train_bpe`")
    ap.add_argument("--conditioning", default="film", choices=["film", "concat"])
    ap.add_argument("--ablate-condition", action="store_true",
                    help="shuffle condition ids during training (control run)")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max-len", type=int, default=192)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/crm.pt")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not Path(args.vocab).is_file():
        raise SystemExit(f"vocab not found: {args.vocab} (see --help)")
    tok = BPETokenizer.from_file(args.vocab)
    print(f"tokenizer: vocab_size={tok.n_vocab} eos={tok.eos_id}")

    pairs = build_dataset(seed=args.seed)
    train_pairs, val_pairs = split(pairs, val_frac=0.2, seed=args.seed)
    print(f"pairs: train={len(train_pairs)} val={len(val_pairs)} "
          f"(conflicting: {sum(p.conflicting for p in val_pairs)} in val)")

    cfg = TransformerConfig(
        vocab_size=tok.n_vocab, d_model=128, n_layer=4, n_head=4,
        max_seq_len=args.max_len, dropout=0.1,
    )
    model = ConditionalRewardModel(
        cfg, num_conditions=len(CONDITIONS),
        conditioning=args.conditioning, pad_id=PAD_ID,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {n_params/1e6:.2f}M params, conditioning={args.conditioning}"
          + (" [CONDITION ABLATED]" if args.ablate_condition else ""))

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    tr = Batcher(train_pairs, tok, args.max_len, device)
    va = Batcher(val_pairs, tok, args.max_len, device)
    rng = random.Random(args.seed)

    steps_total = args.epochs * ((len(tr) + args.batch_size - 1) // args.batch_size)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps_total,
                                                pct_start=0.1)

    history = []
    for ep in range(1, args.epochs + 1):
        tot, nb = 0.0, 0
        for (cx, cm), (rx, rm), cond, _ in tr.batches(args.batch_size, rng=rng):
            if args.ablate_condition:
                cond = cond[torch.randperm(cond.size(0), device=cond.device)]
            r_c = model(cx, cm, cond)
            r_r = model(rx, rm, cond)
            loss, bt = preference_loss(r_c, r_r)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += bt.item(); nb += 1

        if ep % 5 == 0 or ep == args.epochs:
            m = evaluate(model, va, args.batch_size)
            history.append({"epoch": ep, "loss": tot / nb, **m})
            print(f"ep {ep:3d} | loss {tot/nb:.4f} | val acc {m['acc']:.3f} "
                  f"| conflicting {m['acc_conflicting']:.3f} | margin {m['mean_margin']:+.2f}")

    final = evaluate(model, va, args.batch_size)
    shuffled = evaluate(model, va, args.batch_size, shuffle_conditions=True)
    swap_prompt = sorted({p.prompt for p in val_pairs})[0]          # held-out topic
    detailed, brief = swap_pair_for(swap_prompt)
    rows, flipped = condition_swap_test(model, tok, args.max_len, device, swap_prompt, detailed, brief)
    baseline = surface_heuristic_accuracy(val_pairs)

    ceiling = condition_blind_ceiling(val_pairs)

    print("\n" + "=" * 62)
    print(f"condition-blind ceiling          : {ceiling:.3f}   <- upper bound for"
          " any reward model that ignores the condition")
    print(f"val accuracy                    : {final['acc']:.3f}")
    print(f"val accuracy (conflicting pairs): {final['acc_conflicting']:.3f}")
    print(f"val accuracy (condition shuffled): {shuffled['acc']:.3f}   <- control")
    print(f"condition-sensitivity gap        : {final['acc'] - shuffled['acc']:+.3f}")
    print(f"surface-heuristic baseline       : {baseline:.3f}   <- 5-line length/contraction rule; "
          "a learned model must beat this to show more than surface cues")
    print("-" * 62)
    print(f"condition-swap test (held-out prompt: {swap_prompt!r}): same pair, scored under each condition")
    for r in rows:
        print(f"  {r['condition']:8s} detailed={r['score_detailed']:+7.3f} "
              f"brief={r['score_brief']:+7.3f} -> {r['winner']}")
    print(f"  ranking flips between helpful and concise: {'YES' if flipped else 'NO'}")
    print("=" * 62)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": vars(args),
                "conditions": CONDITIONS}, args.out)
    Path(args.out).with_suffix(".json").write_text(json.dumps({
        "history": history, "final": final, "shuffled": shuffled,
        "swap_test": rows, "flipped": flipped, "n_params": n_params,
        "condition_blind_ceiling": ceiling, "surface_heuristic_baseline": baseline,
    }, indent=2))
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
