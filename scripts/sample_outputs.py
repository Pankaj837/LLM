#!/usr/bin/env python
"""What the model actually does: real inputs and outputs, held-out accuracy and failure cases, from one checkpoint.

    python scripts/sample_outputs.py --ckpt session/run/ckpt_last.pt --vocab tokenizers/vocab32k_v2/vocab32k.tok \
        --val math=bins/math.val.bin code=bins/code.val.bin ... --out sample_outputs

Three parts, all written to ``<out>.json`` and ``<out>.md`` (nothing is hand-picked: the prompts are fixed and printed as they are):

1. Held-out continuation: for every domain, ``--per-domain`` documents of its validation file are cut after ``--prompt-tokens`` tokens; the model
   continues greedily and the text is shown next to what the document really said. The next-token top-1 accuracy (teacher forcing, same windows
   every time) is reported per domain.
2. Fixed probes: a small list of one-line questions with an automatic check (arithmetic, a function body, a formula, a law). A pass is a regular
   expression found in the first tokens of the answer; most are expected to FAIL at this size: they are the failure cases.
3. Failure flags on every generation: repetition loop (few distinct 3-grams), empty output, invalid UTF-8, copy of the prompt.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from llm.checkpoint import load_checkpoint, read_checkpoint  # noqa: E402
from llm.data import TokenDataset, read_tokens  # noqa: E402
from llm.model import TransformerModel  # noqa: E402
from llm.tokenizer import BPETokenizer  # noqa: E402
from llm.train import resolve_device  # noqa: E402

PROBES = [
    ("arithmetic", "Question: What is 17 * 24?\nAnswer:", r"\b408\b"),
    ("arithmetic", "Question: What is 125 + 379?\nAnswer:", r"\b504\b"),
    ("math", "The derivative of x^2 with respect to x is", r"2\s*x"),
    ("code", "def add(a, b):\n    return", r"a\s*\+\s*b"),
    ("code", "def factorial(n):\n    if n == 0:\n        return 1\n    return", r"n\s*\*\s*factorial\(n\s*-\s*1\)"),
    ("chemistry", "The chemical formula of water is", r"H2O|H_2O|H\$_2\$O"),
    ("physics", "Newton's second law of motion states that", r"force"),
    ("physics", "The speed of light in vacuum is approximately", r"299|3\.00|3\s*[x×*]\s*10"),
    ("biology", "DNA is made of four nucleotides:", r"adenine|thymine|guanine|cytosine"),
    ("earth", "The three main types of rock are", r"igneous|sedimentary|metamorphic"),
]


def repetition_loop(ids) -> bool:
    """True when the continuation keeps repeating itself: fewer than half of its 3-grams are distinct."""
    if len(ids) < 24:
        return False
    grams = [tuple(ids[i:i + 3]) for i in range(len(ids) - 2)]
    return len(set(grams)) / len(grams) < 0.5


def flags(prompt_ids, out_ids, text) -> list:
    f = []
    if not out_ids:
        f.append("empty")
    if repetition_loop(out_ids):
        f.append("repetition loop")
    if "�" in text.rstrip("�"):
        f.append("invalid UTF-8")
    if out_ids and len(out_ids) >= 8 and out_ids[:8] == prompt_ids[-8:]:
        f.append("copies the prompt")
    return f


def continue_greedy(model, tok, prompt_ids, n, eos, device) -> list:
    x = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    y = model.generate(x, max_new_tokens=n, temperature=0.0, eos_token_id=eos)
    out = y[0, len(prompt_ids):].tolist()
    return out[: out.index(eos)] if eos in out else out


def top1_accuracy(model, path, seq_len, batch_size, batches, device) -> dict:
    ds = TokenDataset(path, seq_len)
    hit = tot = 0
    with torch.no_grad():
        for x, y in ds.sequential_batches(batch_size, batches, device):
            hit += int((model(x).argmax(-1) == y).sum())
            tot += y.numel()
    return {"top1": hit / max(tot, 1), "tokens": tot}


def held_out_cases(path, tok, eos, prompt_tokens, new_tokens, k) -> list:
    """k documents spread evenly through a validation file: (prompt ids, true continuation ids)."""
    h, toks = read_tokens(path)
    a = np.asarray(toks[: min(len(toks), 4_000_000)])
    ends = np.flatnonzero(a == h.eos_id)
    docs = [(int(s) + 1, int(e)) for s, e in zip(np.r_[-1, ends[:-1]], ends)]
    docs = [d for d in docs if d[1] - d[0] >= prompt_tokens + 16] or docs
    picks = [docs[i * len(docs) // k] for i in range(k)] if docs else []
    return [(a[s:s + prompt_tokens].tolist(), a[s + prompt_tokens:min(e, s + prompt_tokens + new_tokens)].tolist()) for s, e in picks]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--val", nargs="+", required=True, metavar="NAME=PATH")
    ap.add_argument("--out", default="sample_outputs")
    ap.add_argument("--per-domain", type=int, default=2)
    ap.add_argument("--prompt-tokens", type=int, default=64)
    ap.add_argument("--new-tokens", type=int, default=96)
    ap.add_argument("--probe-tokens", type=int, default=40)
    ap.add_argument("--seq-len", type=int, default=512, help="window for the teacher-forced accuracy")
    ap.add_argument("--batches", type=int, default=20)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args(argv)
    vals = dict(v.split("=", 1) for v in a.val)
    dev = resolve_device(a.device)
    tok = BPETokenizer.from_file(a.vocab)
    _, cfg, meta, _ = read_checkpoint(a.ckpt)
    model = TransformerModel(cfg).to(dev)
    load_checkpoint(a.ckpt, model, expected_vocab_sha256=BPETokenizer.file_sha256(a.vocab))
    model.eval()
    eos = tok.eos_id
    res = {"checkpoint": a.ckpt, "step": meta.get("step"), "val_loss": meta.get("val_loss"), "domains": {}, "probes": []}

    for name, path in vals.items():
        d = {"top1": None, "samples": []}
        try:
            d.update(top1_accuracy(model, path, a.seq_len, 8, a.batches, dev))
        except ValueError:
            pass
        for prompt, truth in held_out_cases(path, tok, eos, a.prompt_tokens, a.new_tokens, a.per_domain):
            out = continue_greedy(model, tok, prompt, a.new_tokens, eos, dev)
            text = tok.decode(out)
            n = min(len(out), len(truth))
            d["samples"].append({"prompt": tok.decode(prompt), "model": text, "truth": tok.decode(truth),
                                 "match_first_8_tokens": int(sum(int(x == y) for x, y in zip(out[:8], truth[:8]))),
                                 "token_agreement": (sum(int(x == y) for x, y in zip(out[:n], truth[:n])) / n) if n else 0.0,
                                 "flags": flags(prompt, out, text)})
        res["domains"][name] = d

    for domain, prompt, pattern in PROBES:
        ids = tok.encode(prompt, allowed_special=set())
        out = continue_greedy(model, tok, ids, a.probe_tokens, eos, dev)
        text = tok.decode(out)
        res["probes"].append({"domain": domain, "prompt": prompt, "model": text, "expected": pattern,
                              "pass": bool(re.search(pattern, text)), "flags": flags(ids, out, text)})

    n_samples = sum(len(d["samples"]) for d in res["domains"].values())
    flagged = sum(1 for d in res["domains"].values() for s in d["samples"] if s["flags"])
    res["summary"] = {"probes_passed": sum(p["pass"] for p in res["probes"]), "probes": len(res["probes"]),
                      "held_out_samples": n_samples, "held_out_flagged": flagged,
                      "mean_top1": float(np.mean([d["top1"] for d in res["domains"].values() if d["top1"] is not None] or [0.0]))}
    with open(a.out + ".json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)

    L = [f"# Model outputs (checkpoint step {res['step']})", "",
         f"- held-out documents continued: {n_samples} ({flagged} flagged); fixed probes passed: {res['summary']['probes_passed']} of {res['summary']['probes']};"
         f" mean next-token top-1 accuracy {res['summary']['mean_top1']:.1%}", "",
         "## Next-token accuracy by domain (held-out, teacher forcing)", "", "| domain | top-1 | tokens |", "|---|---|---|"]
    L += [f"| {n} | {d['top1']:.1%} | {d['tokens']:,} |" for n, d in res["domains"].items() if d["top1"] is not None]
    L += ["", "## Fixed probes (greedy, first tokens of the answer)", "", "| domain | input | model output | pass | flags |", "|---|---|---|---|---|"]
    for p in res["probes"]:
        L.append("| %s | %s | %s | %s | %s |" % (p["domain"], p["prompt"].replace("\n", " ⏎ "), p["model"][:90].replace("\n", " ⏎ ").replace("|", "/"),
                                                  "PASS" if p["pass"] else "FAIL", ", ".join(p["flags"])))
    L += ["", "## Held-out continuations (input = first tokens of a validation document)", ""]
    for n, d in res["domains"].items():
        for i, s in enumerate(d["samples"], 1):
            L += [f"### {n} #{i}  (first 8 tokens matching the truth: {s['match_first_8_tokens']}/8; flags: {', '.join(s['flags']) or 'none'})", "",
                  "**Input**", "```", s["prompt"], "```", "**Model**", "```", s["model"], "```", "**What the document really said**", "```", s["truth"], "```", ""]
    with open(a.out + ".md", "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"wrote {a.out}.json and {a.out}.md: probes {res['summary']['probes_passed']}/{res['summary']['probes']}, "
          f"flagged {flagged}/{n_samples}, mean top-1 {res['summary']['mean_top1']:.1%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
