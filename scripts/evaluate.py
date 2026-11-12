#!/usr/bin/env python
"""One command: evaluate any checkpoint and write one summary table.

    python scripts/evaluate.py --ckpt run/ckpt_last.pt --bins bins --out results/eval_pilot

Optional: ``--run run/`` (folder with metrics.jsonl and train_config.json, adds the loss curves), ``--compare-ckpt``
(a second checkpoint of the same run, for the integrity comparison), ``--leakage results/leakage.json`` (reuse an earlier
leakage check instead of running one), ``--device cuda``, ``--quick`` (40 benchmark items per file, for a smoke test).

Steps, each skipped when its inputs are missing:
  1. SHA-256 of every input (checkpoint, tokenizer, question files, bins)  -> <out>/inputs_sha256.txt
  2. leakage: benchmark questions found in the training bins (*.train.bin in --bins); flagged questions are not scored
  3. benchmark: frozen science benchmark (evals/, checked against its manifest), acc_norm with 95% intervals per domain
  4. per-domain next-token metrics on every *.val.bin in --bins: cross-entropy, perplexity, top-1/top-5, calibration
  5. generation: held-out continuations and probes, greedy and with FIXED sampled decoding (analyze_checkpoint.SAMPLING:
     temperature 0.8, top-k 50, top-p 0.95, seeds 1/2/3), repetition, option-order consistency; checkpoint integrity
  6. <out>/summary.md: one table per part, plus the exact command and settings

Nothing is random except the seeded sampling, so running it again on the same checkpoint gives the same numbers.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
VOCAB = os.path.join(ROOT, "tokenizers", "vocab32k_v2", "vocab32k.tok")


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def run(cmd: list) -> None:
    print("$ " + " ".join(shlex.quote(c) for c in cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT)


def pct(x) -> str:
    return "-" if x is None else f"{100 * x:.1f}"


def summary(out: str, args, bench: dict | None, ana: dict | None, leak: dict | None) -> str:
    lines = [f"# Evaluation of `{os.path.basename(args.ckpt)}`", "",
             f"- command: `{' '.join(shlex.quote(a) for a in sys.argv)}`",
             f"- checkpoint sha256: `{sha256(args.ckpt)}`"]
    if ana:
        lines.append(f"- step: {ana.get('step')}, device: {ana.get('device')}, seq_len: {ana.get('seq_len')}")
    if leak:
        c = leak["counts"]
        lines.append(f"- leakage: {c['leaked']} of {leak['questions']} questions found in the training bins (not scored), "
                     f"{c['partial']} partial")
    if bench:
        s = bench["summary"]
        o = s["overall"]
        lines += ["", "## Benchmark (acc_norm, %, 95% interval)", "",
                  f"Overall: **{pct(o['acc_norm'])}** [{pct(o['ci95'][0])}, {pct(o['ci95'][1])}] on {o['n']} items, "
                  f"chance {pct(o['chance'])}", "",
                  "| domain | n | acc_norm | 95% interval | chance | above chance |", "|---|---|---|---|---|---|"]
        for d, r in sorted(s["domain"].items()):
            lines.append(f"| {d} | {r['n']} | {pct(r['acc_norm'])} | {pct(r['ci95'][0])}-{pct(r['ci95'][1])} | "
                         f"{pct(r['chance'])} | {'yes' if r['above_chance'] else 'no'} |")
        if "type" in s:
            lines += ["", "| question type | n | acc_norm | 95% interval | chance |", "|---|---|---|---|---|"]
            for t, r in sorted(s["type"].items()):
                lines.append(f"| {t} | {r['n']} | {pct(r['acc_norm'])} | {pct(r['ci95'][0])}-{pct(r['ci95'][1])} | "
                             f"{pct(r['chance'])} |")
    if ana and "loss" in ana:
        L = ana["loss"]
        lines += ["", "## Training run", "",
                  f"- tokens seen {L['tokens_seen'] / 1e9:.2f}B over {L['last_step']} steps; best validation loss "
                  f"{L['val_loss_best']:.4f} (perplexity {L['val_ppl_best']:.2f}) at step {L['val_best_step']}",
                  f"- train minus validation at the end {L['train_minus_val_at_last']:+.4f}; validation still changing "
                  f"{L['val_change_per_1k_steps_at_end']:+.4f} per 1k steps; gradient-norm spikes {L['grad_norm_spikes']}; "
                  f"all values finite: {L['all_finite']}"]
    if ana and "next_token" in ana:
        lines += ["", "## Validation data (per domain)", "",
                  "| domain | cross-entropy | perplexity | top-1 % | top-5 % |", "|---|---|---|---|---|"]
        for d, r in sorted(ana["next_token"].items()):
            lines.append(f"| {d} | {r['cross_entropy']:.3f} | {r['perplexity']:.2f} | {pct(r['top1'])} | {pct(r['top5'])} |")
    if ana and "generation" in ana:
        g = ana["generation"]
        p, oc = g["probe_summary"], g["order_consistency"]
        lines += ["", "## Generation", "", f"Sampling (fixed): `{json.dumps(g.get('sampling'))}`", "",
                  "| domain | repeated 4-grams greedy % | sampled % | real text % | loops greedy | loops sampled |",
                  "|---|---|---|---|---|---|"]
        for d, r in sorted(g["domains"].items()):
            lines.append(f"| {d} | {pct(r['rep4'])} | {pct(r['rep4_sampled'])} | {pct(r['rep4_reference'])} | "
                         f"{r['loops']}/{r['n']} | {r['loops_sampled']}/{r['n']} |")
        lines += ["", f"Probes: greedy pass {p['greedy_pass']}/{p['probes']}, sampled pass rate {pct(p['sample_pass_rate'])}%, "
                      f"three samples agree on {p['samples_agree']}/{p['probes']}, greedy identical twice "
                      f"{p['greedy_identical_twice']}/{p['probes']}",
                  f"Option order: same answer after reversing options {pct(oc['same_answer'])}% of {oc['items']} "
                  f"(accuracy {pct(oc['acc_original'])}% vs {pct(oc['acc_reversed'])}% reversed)"]
    if ana and "integrity" in ana:
        lines += ["", "## Integrity", ""]
        for name, c in ana["integrity"]["checkpoints"].items():
            lines.append(f"- {name}: step {c['step']}, {c['params']:,} parameters, all finite: {c['all_finite']}, "
                         f"tokenizer matches: {c['vocab_matches']}")
        lines.append(f"- forward pass: `{json.dumps(ana['integrity'].get('forward'))}`")
    text = "\n".join(lines) + "\n"
    with open(os.path.join(out, "summary.md"), "w", encoding="utf-8") as f:
        f.write(text)
    return text


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--bins", help="folder with <domain>.val.bin (and <domain>.train.bin for the leakage check)")
    ap.add_argument("--run", help="training run folder with metrics*.jsonl and train_config*.json (adds the loss part)")
    ap.add_argument("--compare-ckpt")
    ap.add_argument("--leakage", help="existing check_leakage.py JSON; default: run the check on --bins/*.train.bin")
    ap.add_argument("--vocab", default=VOCAB)
    ap.add_argument("--evals", default=os.path.join(ROOT, "evals"))
    ap.add_argument("--out", default="results/eval")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--quick", action="store_true", help="40 benchmark items per file and fewer generation cases (smoke test)")
    a = ap.parse_args(argv)
    for p in ("ckpt", "compare_ckpt", "bins", "run", "leakage", "vocab", "evals"):
        if getattr(a, p):
            setattr(a, p, os.path.abspath(getattr(a, p)))
    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    py, t0 = sys.executable, time.time()

    vals = sorted(glob.glob(os.path.join(a.bins, "*.val.bin"))) if a.bins else []
    trains = sorted(glob.glob(os.path.join(a.bins, "*.train.bin"))) if a.bins else []
    metrics = sorted(glob.glob(os.path.join(a.run, "metrics*.jsonl")))[:1] if a.run else []
    config = sorted(glob.glob(os.path.join(a.run, "train_config*.json")))[:1] if a.run else []

    # 1. inputs
    with open(os.path.join(out, "inputs_sha256.txt"), "w", encoding="utf-8") as f:
        for p in [a.ckpt, a.compare_ckpt, a.vocab, os.path.join(a.evals, "MANIFEST.json"), *metrics, *config, *vals, *trains]:
            if p and os.path.exists(p):
                f.write(f"{sha256(p)}  {p}\n")

    # 2. leakage
    leak_path = a.leakage
    if not leak_path and trains:
        run([py, "scripts/check_leakage.py", "--evals", a.evals, "--vocab", a.vocab, "--bins", *trains,
             "--out", os.path.join(out, "leakage")])
        leak_path = os.path.join(out, "leakage.json")
    leak = json.load(open(leak_path, encoding="utf-8")) if leak_path else None

    # 3. benchmark
    cmd = [py, "scripts/eval_science.py", "--ckpt", a.ckpt, "--vocab", a.vocab, "--evals", a.evals,
           "--out", os.path.join(out, "benchmark"), "--device", a.device]
    if leak_path:
        cmd += ["--exclude", leak_path]
    if a.quick:
        cmd += ["--limit", "40"]
    run(cmd)
    bench = json.load(open(os.path.join(out, "benchmark.json"), encoding="utf-8"))

    # 4-5. validation metrics, generation with fixed sampling, integrity
    cmd = [py, "scripts/analyze_checkpoint.py", "--ckpt", a.ckpt, "--vocab", a.vocab, "--evals", a.evals,
           "--out", os.path.join(out, "analysis"), "--device", a.device]
    if vals:
        cmd += ["--val", *[f"{os.path.basename(v)[:-len('.val.bin')]}={v}" for v in vals]]
    if metrics and config:
        cmd += ["--metrics", metrics[0], "--config", config[0]]
    if a.compare_ckpt:
        cmd += ["--compare-ckpt", a.compare_ckpt]
    if a.quick:
        cmd += ["--windows", "10", "--per-domain", "2", "--mcq", "40"]
    run(cmd)
    ana = json.load(open(os.path.join(out, "analysis.json"), encoding="utf-8"))

    # 6. summary
    print("\n" + summary(out, a, bench, ana, leak))
    print(f"done in {time.time() - t0:,.0f}s: {out}/summary.md (details: benchmark.json, analysis.json, inputs_sha256.txt)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
