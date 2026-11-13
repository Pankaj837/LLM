#!/usr/bin/env python
"""Scientific capability evaluation of one checkpoint on the frozen question set in ``evals/`` (see ``evals/README.md``).

    python scripts/eval_science.py --ckpt session_main/run/ckpt_last.pt --vocab tokenizers/vocab32k_v2/vocab32k.tok \
        --evals evals --out results/eval_science_step12345 [--exclude results/leakage.json]

Two kinds of question, both scored without any sampling, so the same checkpoint always gives the same numbers:

* ``mcq``: every choice is appended to ``Question: ...\\nAnswer:`` and the model's log-probability of the choice tokens is
  computed. ``acc`` picks the choice with the highest total log-probability; ``acc_norm`` divides by the number of
  characters of the choice first (fairer when choices differ in length). This works for a base model that cannot yet
  follow instructions. Chance level is 1 / number of choices.
* ``numeric``: a few worked examples are shown, the model continues greedily, and the first number of its answer is
  compared with the reference within the item's relative tolerance. Chance level is 0.

Outputs ``<out>.json`` (every item with the model's choice or answer) and ``<out>.md`` (accuracy by domain, level and
skill with a 95% interval and the chance level, error types, real correct and wrong answers). Exits non-zero if the
question files differ from ``evals/MANIFEST.json`` (the set is frozen) unless ``--allow-modified`` is given.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import re
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llm.checkpoint import load_checkpoint, read_checkpoint  # noqa: E402
from llm.model import TransformerModel  # noqa: E402
from llm.tokenizer import BPETokenizer  # noqa: E402
from llm.train import resolve_device  # noqa: E402
from sample_outputs import repetition_loop  # noqa: E402

NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?(?:[eE][-+]?\d+)?|-?\.\d+")
FEWSHOT_FILES = {"gsm8k": "public/fewshot_gsm8k.json", "science_numeric": "handwritten/fewshot_science_numeric.json"}
LEVELS = ("basic", "intermediate", "advanced")


# ------------------------------------------------------------------ question set
def load_items(evals_dir: str) -> list:
    items = []
    for p in sorted(glob.glob(os.path.join(evals_dir, "**", "*.jsonl"), recursive=True)):
        with open(p, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    it = json.loads(line)
                    it["_file"] = os.path.relpath(p, evals_dir).replace(os.sep, "/")
                    items.append(it)
    ids = [it["id"] for it in items]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate question ids in the evaluation set")
    return items


def manifest_problems(evals_dir: str) -> list:
    """Files whose SHA-256 differs from evals/MANIFEST.json, or that are missing from it."""
    path = os.path.join(evals_dir, "MANIFEST.json")
    if not os.path.exists(path):
        return ["MANIFEST.json missing"]
    known = {e["file"]: e["sha256"] for e in json.load(open(path, encoding="utf-8"))["files"]}
    probs = []
    for p in sorted(glob.glob(os.path.join(evals_dir, "**", "*.jsonl"), recursive=True)
                    + glob.glob(os.path.join(evals_dir, "*", "fewshot_*.json"))):
        rel = os.path.relpath(p, evals_dir).replace(os.sep, "/")   # the manifest always uses forward slashes
        if known.get(rel) != hashlib.sha256(open(p, "rb").read()).hexdigest():
            probs.append(rel)
    return probs


def load_fewshot(evals_dir: str) -> dict:
    out = {}
    for name, rel in FEWSHOT_FILES.items():
        p = os.path.join(evals_dir, rel)
        if os.path.exists(p):
            out[name] = json.load(open(p, encoding="utf-8"))["shots"]
    return out


def fmt_num(x: float) -> str:
    return str(int(x)) if float(x).is_integer() else f"{x:g}"


def numeric_prompt(item: dict, shots: dict) -> str:
    parts = []
    for s in shots.get(item.get("fewshot", ""), []):
        if "solution" in s:
            parts.append(f"Question: {s['question']}\nAnswer: {s['solution']} The answer is {fmt_num(s['answer'])}.\n\n")
        else:
            parts.append(f"Question: {s['question']}\nAnswer: {fmt_num(s['answer'])}\n\n")
    return "".join(parts) + f"Question: {item['question']}\nAnswer:"


def mcq_context(item: dict) -> str:
    return f"Question: {item['question']}\nAnswer:"


# ------------------------------------------------------------------ scoring
def parse_number(text: str, worked: bool):
    """The model's numeric answer: for worked solutions only the number after 'The answer is' (strict: a number picked
    out of the working is often one copied from the question), else the first number; None if absent."""
    text = text.split("\n\nQuestion:")[0].split("\nQuestion:")[0]
    if worked:
        m = re.search(r"answer is\s*\$?\s*(" + NUMBER.pattern + ")", text)
        text = m.group(1) if m else ""
    m = NUMBER.search(text)
    if not m:
        return None
    try:
        return float(m.group(0).replace(",", ""))
    except ValueError:
        return None


def numeric_correct(pred, ref: float, tol: float) -> bool:
    if pred is None:
        return False
    return abs(pred - ref) <= max(tol * abs(ref), 1e-6)


def numeric_error(pred, ref: float, tol: float, ids) -> str:
    if numeric_correct(pred, ref, tol):
        return "correct"
    if pred is None:
        return "no_number"
    if pred != 0 and ref != 0:
        k = round(math.log10(abs(pred / ref)))
        if k != 0 and numeric_correct(pred / 10 ** k, ref, max(tol, 0.01)):
            return "unit_error (off by 10^%d)" % k
    if pred == -ref:
        return "sign_error"
    if repetition_loop(ids):
        return "repetition_loop"
    return "wrong_value"


@torch.no_grad()
def choice_logprobs(model, tok, context: str, choices: list, device, max_len: int) -> list:
    """Sum of log-probabilities of each choice's tokens after the context (one right-padded batch per question)."""
    ctx = tok.encode(context, allowed_special=set())
    conts = [tok.encode(" " + c, allowed_special=set()) or [tok.eos_id] for c in choices]
    keep = max(1, max_len - max(len(c) for c in conts))
    ctx = ctx[-keep:]                                         # left-truncate a very long context
    seqs = [ctx + c for c in conts]
    width = max(len(s) for s in seqs)
    x = torch.full((len(seqs), width), tok.eos_id, dtype=torch.long)
    for i, s in enumerate(seqs):
        x[i, :len(s)] = torch.tensor(s)
    logp = F.log_softmax(model(x.to(device)).float(), dim=-1).cpu()
    out = []
    for i, c in enumerate(conts):
        pos = torch.arange(len(ctx) - 1, len(ctx) - 1 + len(c))   # logits at t predict token t+1
        out.append(float(logp[i, pos, torch.tensor(c)].sum()))
    return out


def score_mcq(model, tok, item, device, max_len) -> dict:
    lps = choice_logprobs(model, tok, mcq_context(item), item["choices"], device, max_len)
    norm = [lp / max(len(c), 1) for lp, c in zip(lps, item["choices"])]
    pick, pick_norm = int(max(range(len(lps)), key=lps.__getitem__)), int(max(range(len(norm)), key=norm.__getitem__))
    types = item.get("distractor_types")
    return {"logprobs": [round(v, 4) for v in lps], "pick": pick, "pick_norm": pick_norm,
            "correct": pick == item["answer"], "correct_norm": pick_norm == item["answer"],
            "error": "correct" if pick_norm == item["answer"] else (types[pick_norm] if types else "wrong_choice"),
            "chance": 1.0 / len(item["choices"])}


@torch.no_grad()
def score_numeric(model, tok, item, shots, device, max_len, new_tokens) -> dict:
    ids = tok.encode(numeric_prompt(item, shots), allowed_special=set())[-(max_len - new_tokens):]
    y = model.generate(torch.tensor([ids], device=device), max_new_tokens=new_tokens, temperature=0.0, eos_token_id=tok.eos_id)
    out = y[0, len(ids):].tolist()
    out = out[: out.index(tok.eos_id)] if tok.eos_id in out else out
    text = tok.decode(out)
    worked = item.get("fewshot") == "gsm8k"
    pred = parse_number(text, worked)
    tol = float(item.get("tolerance", 0.0))
    ok = numeric_correct(pred, float(item["answer"]), tol)
    return {"output": text.split("\n\nQuestion:")[0][:600], "pred": pred, "correct": ok, "correct_norm": ok,
            "error": numeric_error(pred, float(item["answer"]), tol, out), "chance": 0.0}


# ------------------------------------------------------------------ report
def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def summarise(records: list, key) -> dict:
    groups = {}
    for r in records:
        groups.setdefault(key(r), []).append(r)
    out = {}
    for g, rs in sorted(groups.items()):
        k = sum(r["correct_norm"] for r in rs)
        lo, hi = wilson(k, len(rs))
        chance = sum(r["chance"] for r in rs) / len(rs)
        out[g] = {"n": len(rs), "acc_norm": k / len(rs), "acc": sum(r["correct"] for r in rs) / len(rs),
                  "ci95": [lo, hi], "chance": chance, "above_chance": lo > chance}
    return out


def write_report(res: dict, path: str) -> None:
    recs = res["items"]
    s = res["summary"]
    L = [f"# Scientific evaluation: {os.path.basename(res['checkpoint'])} (step {res['step']})", "",
         f"- question set: {res['eval_set']['items']} items, manifest {'matches' if not res['eval_set']['modified'] else 'DIFFERS: ' + ', '.join(res['eval_set']['modified'])}",
         f"- excluded for train-data overlap: {len(res['excluded'])}",
         f"- overall acc_norm {s['overall']['acc_norm']:.1%} (chance {s['overall']['chance']:.1%}, 95% CI {s['overall']['ci95'][0]:.1%}–{s['overall']['ci95'][1]:.1%})",
         f"- time {res['seconds']:.0f} s on {res['device']}", "",
         "Scores are acc_norm for multiple choice (log-likelihood per character) and exact-within-tolerance for numeric answers. "
         "'above chance' means the lower end of the 95% interval is above the chance level.", "",
         "## By domain", "", "| domain | n | accuracy | 95% CI | chance | above chance |", "|---|---|---|---|---|---|"]
    for g, v in s["domain"].items():
        L.append(f"| {g} | {v['n']} | {v['acc_norm']:.1%} | {v['ci95'][0]:.1%}–{v['ci95'][1]:.1%} | {v['chance']:.1%} | {'yes' if v['above_chance'] else 'no'} |")
    L += ["", "## By domain and level", "", "| domain | " + " | ".join(LEVELS) + " |", "|---|" + "---|" * len(LEVELS)]
    for d in s["domain"]:
        cells = []
        for lv in LEVELS:
            v = s["domain_level"].get(f"{d} / {lv}")
            cells.append(f"{v['acc_norm']:.0%} (n={v['n']}, chance {v['chance']:.0%})" if v else "–")
        L.append(f"| {d} | " + " | ".join(cells) + " |")
    L += ["", "## By question type and skill", "", "| group | n | accuracy | chance |", "|---|---|---|---|"]
    for g, v in list(s["type"].items()) + list(s["skill"].items()):
        L.append(f"| {g} | {v['n']} | {v['acc_norm']:.1%} | {v['chance']:.1%} |")
    L += ["", "## Error types (wrong answers only)", "",
          "For hand-written multiple-choice questions every wrong option is labelled (wrong formula, unit error, calculation error, "
          "misconception, unsupported claim, ...), so a wrong pick shows which mistake the model prefers.", "",
          "| error | count |", "|---|---|"]
    for e, n in sorted(res["errors"].items(), key=lambda kv: -kv[1]):
        L.append(f"| {e} | {n} |")
    L += ["", "## Examples (real model behaviour, first items of each domain)", ""]
    for d in s["domain"]:
        rs = [r for r in recs if r["domain"] == d]
        for want, label in ((True, "correct"), (False, "wrong")):
            for r in [r for r in rs if r["correct_norm"] == want][:2]:
                L.append(f"**{d} / {r['level']} / {label}** `{r['id']}`  ")
                L.append(f"Q: {r['question'][:300]}  ")
                if r["type"] == "mcq":
                    L.append(f"Model chose: {r['choices'][r['pick_norm']]}  ")
                    L.append(f"Reference: {r['choices'][r['answer']]} (error type: {r['error']})")
                else:
                    L.append(f"Model wrote: `{r['output'][:200].strip()}` → parsed {r['pred']}  ")
                    L.append(f"Reference: {fmt_num(r['answer'])} {r.get('unit', '')} (error type: {r['error']})")
                L.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


# ------------------------------------------------------------------ main
def evaluate(model, tok, items, shots, device, max_len, new_tokens, gsm8k_tokens, log_every=200) -> list:
    records = []
    t0 = time.time()
    for i, it in enumerate(items, 1):
        if it["type"] == "mcq":
            r = score_mcq(model, tok, it, device, max_len)
        elif it["type"] == "numeric":
            r = score_numeric(model, tok, it, shots, device, max_len,
                              gsm8k_tokens if it.get("fewshot") == "gsm8k" else new_tokens)
        else:
            raise ValueError(f"{it['id']}: unknown type {it['type']}")
        records.append({**{k: v for k, v in it.items() if not k.startswith("_")}, **r})
        if log_every and i % log_every == 0:
            print(f"  {i}/{len(items)} items, {time.time() - t0:.0f} s", flush=True)
    return records


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--vocab", required=True, help=".tok file; its sha256 must match the checkpoint's")
    ap.add_argument("--evals", default="evals")
    ap.add_argument("--out", default="eval_science")
    ap.add_argument("--exclude", help="leakage JSON from scripts/check_leakage.py; its flagged ids are not scored")
    ap.add_argument("--domains", nargs="*", help="only these domains")
    ap.add_argument("--limit", type=int, default=0, help="at most this many items per file (quick runs)")
    ap.add_argument("--new-tokens", type=int, default=24, help="generated tokens for short numeric answers")
    ap.add_argument("--gsm8k-tokens", type=int, default=192, help="generated tokens for GSM8K worked solutions")
    ap.add_argument("--repeat-check", type=int, default=20, help="re-score this many items and require identical results")
    ap.add_argument("--allow-modified", action="store_true", help="run even if the question files differ from the manifest")
    ap.add_argument("--device", default="auto")
    a = ap.parse_args(argv)

    modified = manifest_problems(a.evals)
    if modified and not a.allow_modified:
        print(f"question set differs from evals/MANIFEST.json: {modified} (use --allow-modified for a draft run)")
        return 2
    items = load_items(a.evals)
    if a.domains:
        items = [it for it in items if it["domain"] in a.domains]
    if a.limit:
        per = {}
        items = [it for it in items if per.setdefault(it["_file"], []).append(1) or len(per[it["_file"]]) <= a.limit]
    excluded = []
    if a.exclude:
        flagged = set(json.load(open(a.exclude, encoding="utf-8")).get("flagged_ids", []))
        excluded = [it["id"] for it in items if it["id"] in flagged]
        items = [it for it in items if it["id"] not in flagged]
    shots = load_fewshot(a.evals)

    torch.manual_seed(0)
    dev = resolve_device(a.device)
    tok = BPETokenizer.from_file(a.vocab)
    _, cfg, meta, _ = read_checkpoint(a.ckpt)
    if cfg is None:
        raise ValueError("checkpoint has no embedded config; cannot rebuild the model")
    model = TransformerModel(cfg).to(dev)
    load_checkpoint(a.ckpt, model, expected_vocab_sha256=BPETokenizer.file_sha256(a.vocab))
    model.eval()
    max_len = cfg.max_position_embeddings

    print(f"{len(items)} items on {dev} (checkpoint step {meta.get('step')})", flush=True)
    t0 = time.time()
    records = evaluate(model, tok, items, shots, dev, max_len, a.new_tokens, a.gsm8k_tokens)
    seconds = time.time() - t0
    n_rep = min(a.repeat_check, len(items))
    again = evaluate(model, tok, items[:n_rep], shots, dev, max_len, a.new_tokens, a.gsm8k_tokens, log_every=0)
    repeat_ok = all(x.get("pick_norm") == y.get("pick_norm") and x.get("pred") == y.get("pred") for x, y in zip(records, again))

    errors = {}
    for r in records:
        if not r["correct_norm"]:
            errors[r["error"]] = errors.get(r["error"], 0) + 1
    res = {"checkpoint": os.path.abspath(a.ckpt), "step": meta.get("step"), "val_loss": meta.get("val_loss"),
           "vocab_sha256": BPETokenizer.file_sha256(a.vocab), "device": str(dev), "seconds": seconds,
           "eval_set": {"items": len(records), "modified": modified}, "excluded": excluded,
           "repeat_check": {"items": n_rep, "identical": repeat_ok},
           "summary": {"overall": summarise(records, lambda r: "all")["all"],
                       "domain": summarise(records, lambda r: r["domain"]),
                       "domain_level": summarise(records, lambda r: f"{r['domain']} / {r['level']}"),
                       "type": summarise(records, lambda r: r["type"]),
                       "skill": summarise([r for r in records if r.get("skill")], lambda r: r["skill"])},
           "errors": errors, "items": records}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out + ".json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    write_report(res, a.out + ".md")
    o = res["summary"]["overall"]
    print(f"wrote {a.out}.json and {a.out}.md: {len(records)} items, acc_norm {o['acc_norm']:.1%} "
          f"(chance {o['chance']:.1%}), repeat check {'identical' if repeat_ok else 'DIFFERENT'}")
    return 0 if repeat_ok else 1


if __name__ == "__main__":
    sys.exit(main())
