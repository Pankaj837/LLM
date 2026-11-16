#!/usr/bin/env python
"""Answer the same held-out prompts with one or more checkpoints under identical decoding, and score what comes out.

    python scripts/sft_generate.py --ckpt base=ckpt_1b_full_best.pt sft=sft/run_full/ckpt_best.pt \
        --vocab tokenizers/vocab32k_v2/vocab32k.tok --prompts sft/v1/test.jsonl --out results/sft_compare \
        --max-new-tokens 256 --temperature 0 [--repetition-penalty 1.15] [--no-repeat-ngram 4] [--run-code]

Prompts come from a ``.jsonl`` with ``prompt`` (and optionally ``system``, ``domain``, ``answer``, ``tests``), normally the ``test.jsonl``
of ``sft_prepare.py`` (never used for training). Every checkpoint gets exactly the same chat-format prompts (``llm.sft.prompt_ids``),
the same decoding settings and the same seed. For each output the script records, with nothing edited or re-run:

  * ``finished``    the model emitted the end token itself (False = it ran into ``--max-new-tokens``): termination behaviour
  * ``repetition``  the continuation repeats itself (fewer than half of its 3-grams are distinct, as in ``sample_outputs.py``)
  * ``empty``       no tokens before the end token
  * ``passed``      only when the prompt carries a reference: ``answer`` (final number / text must match) or, with ``--run-code``,
                    ``tests`` (the response's code block must pass them: this EXECUTES model-written code, disposable machine only)

Results go to ``<out>.json`` (everything) and ``<out>.md`` (comparison table, per-domain pass rates, every prompt with every response).
Teacher-forced token accuracy is a different measurement and is not reported here.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)
from llm.sft import DEFAULT_END, generate_ids, load_base_model, load_jsonl, prompt_ids, special_ids  # noqa: E402
from llm.tokenizer import BPETokenizer  # noqa: E402
from llm.train import resolve_device, resolve_dtype  # noqa: E402
from sample_outputs import repetition_loop  # noqa: E402
from sft_verify import answer_matches, extract_code_block, run_python_tests  # noqa: E402


def wilson(k: int, n: int, z: float = 1.96) -> List[float]:
    if n == 0:
        return [0.0, 1.0]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]


def score(text: str, ex: Dict[str, Any], run_code: bool, timeout: float):
    """True / False when the example has a checkable reference, else None."""
    if "answer" in ex:
        return answer_matches(text, ex["answer"])
    if run_code and ex.get("tests"):
        return run_python_tests(extract_code_block(text), ex["tests"], timeout)[0]
    return None


def run_one(name: str, path: str, tok, rows, a, device, amp) -> Dict[str, Any]:
    model, info = load_base_model(path, device, BPETokenizer.file_sha256(a.vocab))
    model.eval()
    eos = special_ids(tok, a.end_token)[a.end_token]
    results = []
    for i, ex in enumerate(rows):
        pids = prompt_ids(tok, ex["prompt"], ex.get("system"), a.end_token)
        out, finished = generate_ids(model, pids, max_new_tokens=a.max_new_tokens, eos_id=eos, device=device, temperature=a.temperature,
                                     top_k=a.top_k, top_p=a.top_p, repetition_penalty=a.repetition_penalty,
                                     no_repeat_ngram=a.no_repeat_ngram, seed=a.seed + i, amp_dtype=amp)
        text = tok.decode(out)
        results.append({"id": ex.get("id"), "domain": ex.get("domain", "unknown"), "prompt": ex["prompt"], "response": text,
                        "tokens": len(out), "finished": finished, "repetition": repetition_loop(out), "empty": len(out) == 0,
                        "passed": score(text, ex, a.run_code, a.code_timeout)})
        print(f"[{name}] {i + 1}/{len(rows)}", flush=True)
    n = len(results)
    scored = [r for r in results if r["passed"] is not None]
    passed = sum(1 for r in scored if r["passed"])
    per_domain: Dict[str, List[int]] = defaultdict(lambda: [0, 0])
    for r in scored:
        per_domain[r["domain"]][1] += 1
        per_domain[r["domain"]][0] += int(r["passed"])
    summary = {"examples": n, "finished_rate": sum(r["finished"] for r in results) / max(n, 1),
               "repetition_rate": sum(r["repetition"] for r in results) / max(n, 1),
               "empty_rate": sum(r["empty"] for r in results) / max(n, 1), "scored": len(scored), "passed": passed,
               "pass_rate": passed / len(scored) if scored else None, "pass_ci95": wilson(passed, len(scored)) if scored else None,
               "per_domain": {d: {"passed": v[0], "scored": v[1]} for d, v in sorted(per_domain.items())}}
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {"name": name, "checkpoint": os.path.basename(path), "step": info["step"], "summary": summary, "results": results}


def pct(x) -> str:
    return "-" if x is None else f"{100 * x:.1f}%"


def write_md(path: str, runs: List[Dict[str, Any]], settings: Dict[str, Any]) -> None:
    L = ["# Generation comparison", "", "Settings (identical for every checkpoint): `" + json.dumps(settings) + "`", "",
         "| checkpoint | examples | finished (emitted end token) | repetition loop | empty | scored | passed | pass rate [95% CI] |",
         "|---|---|---|---|---|---|---|---|"]
    for r in runs:
        s = r["summary"]
        ci = f" [{pct(s['pass_ci95'][0])}, {pct(s['pass_ci95'][1])}]" if s["pass_ci95"] else ""
        L.append(f"| {r['name']} ({r['checkpoint']}, step {r['step']}) | {s['examples']} | {pct(s['finished_rate'])} | {pct(s['repetition_rate'])} | "
                 f"{pct(s['empty_rate'])} | {s['scored']} | {s['passed']} | {pct(s['pass_rate'])}{ci} |")
    domains = sorted({d for r in runs for d in r["summary"]["per_domain"]})
    if domains:
        L += ["", "## Pass rate by domain (passed / scored)", "", "| domain | " + " | ".join(r["name"] for r in runs) + " |",
              "|---|" + "---|" * len(runs)]
        for d in domains:
            L.append(f"| {d} | " + " | ".join(f"{r['summary']['per_domain'].get(d, {}).get('passed', 0)} / {r['summary']['per_domain'].get(d, {}).get('scored', 0)}"
                                             for r in runs) + " |")
    L += ["", "## Every prompt and every response (raw, nothing edited)", ""]
    for i in range(len(runs[0]["results"])):
        L += [f"### {i + 1}. {runs[0]['results'][i]['domain']}", "", "**Prompt**", "```", runs[0]["results"][i]["prompt"], "```"]
        for r in runs:
            x = r["results"][i]
            flags = [k for k in ("repetition", "empty") if x[k]] + ([] if x["finished"] else ["no end token"])
            verdict = "" if x["passed"] is None else (" - PASS" if x["passed"] else " - FAIL")
            L += [f"**{r['name']}**{verdict}" + (f"  (flags: {', '.join(flags)})" if flags else ""), "```", x["response"], "```"]
        L.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", nargs="+", required=True, metavar="NAME=PATH")
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", default="sft_generate")
    ap.add_argument("--limit", type=int, default=0, help="use only the first N prompts (0 = all)")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--repetition-penalty", type=float, default=1.0)
    ap.add_argument("--no-repeat-ngram", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--end-token", default=DEFAULT_END)
    ap.add_argument("--run-code", action="store_true", help="execute model-written code against the prompt's tests (disposable machine only)")
    ap.add_argument("--code-timeout", type=float, default=10.0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="auto", choices=["auto", "fp32", "bf16"])
    a = ap.parse_args(argv)
    device = resolve_device(a.device)
    amp = resolve_dtype(a.dtype, device)
    tok = BPETokenizer.from_file(a.vocab)
    rows = load_jsonl(a.prompts)
    if a.limit:
        rows = rows[:a.limit]
    specs = []
    for item in a.ckpt:
        name, sep, path = item.partition("=")
        if not sep or not name or not path:
            ap.error(f"--ckpt expects NAME=PATH, got {item!r}")
        specs.append((name, path))
    runs = [run_one(n, p, tok, rows, a, device, amp) for n, p in specs]
    settings = {k: getattr(a, k) for k in ("max_new_tokens", "temperature", "top_k", "top_p", "repetition_penalty", "no_repeat_ngram", "seed",
                                           "end_token", "run_code")}
    settings["prompts"] = os.path.basename(a.prompts)
    with open(a.out + ".json", "w", encoding="utf-8") as f:
        json.dump({"settings": settings, "runs": runs}, f, indent=1, ensure_ascii=False)
    write_md(a.out + ".md", runs, settings)
    for r in runs:
        print(r["name"], json.dumps({k: v for k, v in r["summary"].items() if k != "per_domain"}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
