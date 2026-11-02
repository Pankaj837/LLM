#!/usr/bin/env python
"""LaTeX tables and figures for the checkpoint analysis report, from the result files (nothing is typed in by hand).

    python scripts/analysis_report_tex.py --analysis results/analysis_100m_step8250.json \
        --curve results/analysis_100m_step8250_curve.csv --eval results/eval_science_100m_step8250.json \
        --config run/train_config.json --out reports/analysis_100m/generated

``--analysis`` and ``--curve`` come from ``scripts/analyze_checkpoint.py``; ``--eval`` from ``scripts/eval_science.py``. The gap table
maps the 20 domains of the 30B token plan to the training bins of the run, the benchmark domains and the validation files.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_report_tex import esc, pct, write  # noqa: E402

# (domain in the 30B token plan, planned tokens in billions, training bins of the run, benchmark domains, validation file)
PLAN = [
    ("Mathematics", 2.4, ["math"], ["math"], "math"),
    ("Physics", 2.1, ["physics"], ["physics"], "physics"),
    ("Chemistry", 1.2, ["chem"], ["chemistry"], "chem"),
    ("Biology", 1.2, ["bio"], ["bio"], "bio"),
    ("Earth \\& Environment", 0.9, ["earth"], ["earth"], "earth"),
    ("Scientific Discovery", 1.2, ["scidisc"], ["scidisc"], "scidisc"),
    ("Scientific Q\\&A", 1.5, ["sciqa"], ["sciqa"], "sciqa"),
    ("Scientific Programming", 1.5, ["sciprog"], ["sciprog"], "sciprog"),
    ("Code", 1.8, ["code"], ["code"], "code"),
    ("CAD/STEP", 1.2, ["cad"], ["cad"], "cad"),
    ("Multi-step Reasoning", 2.1, ["multistep"], ["multistep"], "multistep"),
    ("Astronomy", 0.9, ["astro"], ["astro"], "astro"),
    ("Finance", 0.6, ["finance"], ["finance"], "finance"),
    ("Engineering Agent/Tool Use", 1.2, ["engineering"], [], "engineering"),
    ("Engineering Reasoning", 2.4, [], ["electrical_eng", "engineering"], None),
    ("Generative Engineering", 1.8, ["gen_eng"], [], "gen_eng"),
    ("Experimental Validation", 1.2, ["exp_val"], ["exp_val"], "exp_val"),
    ("Technical Communication", 0.6, ["tech_comm"], ["tech_comm"], "tech_comm"),
    ("Numerical Simulation", 2.1, [], ["simulation"], None),
    ("Physics + AI (PINNs/FNO)", 2.1, [], ["sciml"], None),
]
SMALL = 30   # fewer benchmark questions than this: direction only


def table(header: str, cols: str, rows: list) -> list:
    return [r"\begin{tabular}{" + cols + "}", r"\toprule", header + r" \\", r"\midrule"] + [r + r" \\" for r in rows] + \
           [r"\bottomrule", r"\end{tabular}"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--analysis", required=True)
    ap.add_argument("--curve", required=True)
    ap.add_argument("--eval", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    with open(a.analysis, encoding="utf-8") as f:
        an = json.load(f)
    with open(a.eval, encoding="utf-8") as f:
        ev = json.load(f)
    with open(a.config, encoding="utf-8") as f:
        tc = json.load(f)
    os.makedirs(a.out, exist_ok=True)
    L, nt, g, it = an["loss"], an["next_token"], an["generation"], an["integrity"]
    best, last = it["checkpoints"]["ckpt_best.pt"], it["checkpoints"].get("ckpt_last.pt")

    m = {"LossFirst": f"{L['train_loss_first']:.2f}", "LossTrainEnd": f"{L['train_loss_last_smoothed']:.3f}",
         "ValFirst": f"{L['val_loss_first']:.2f}", "ValBest": f"{L['val_loss_best']:.3f}", "ValBestStep": f"{L['val_best_step']:,}",
         "ValPplBest": f"{L['val_ppl_best']:.2f}", "ValPoints": L["val_points"], "LastStep": f"{L['last_step']:,}",
         "MaxSteps": f"{tc['train']['max_steps']:,}", "DonePct": f"{100 * L['last_step'] / tc['train']['max_steps']:.1f}\\%",
         "TokensSeen": f"{L['tokens_seen'] / 1e9:.2f}B", "Gap": f"{L['train_minus_val_at_last']:+.3f}",
         "ValSlope": f"{L['val_change_per_1k_steps_at_end']:.3f}", "GradMed": f"{L['grad_norm_median']:.2f}",
         "GradMax": f"{L['grad_norm_max']:.2f}", "GradSpikes": L["grad_norm_spikes"], "Tps": f"{L['tokens_per_s_mean']:,.0f}",
         "PeakMem": f"{L['peak_mem_gb']:.1f}",
         "ProbeN": g["probe_summary"]["probes"], "ProbeGreedy": g["probe_summary"]["greedy_pass"],
         "ProbeSample": pct(g["probe_summary"]["sample_pass_rate"], 0), "ProbeAgree": g["probe_summary"]["samples_agree"],
         "ProbeLoops": g["probe_summary"]["greedy_loops"], "ProbeSame": g["probe_summary"]["greedy_identical_twice"],
         "OrderN": g["order_consistency"]["items"], "OrderSame": pct(g["order_consistency"]["same_answer"], 1),
         "OrderAccA": pct(g["order_consistency"]["acc_original"], 1), "OrderAccB": pct(g["order_consistency"]["acc_reversed"], 1),
         "RelDiff": f"{it['pair']['relative_l2_difference'] * 100:.2f}\\%" if "pair" in it else "--",
         "RepeatDiff": f"{it['forward']['repeat_max_abs_diff']:.1g}", "CpuDiff": f"{it['forward']['cpu_vs_device_max_abs_diff']:.1e}",
         "CpuArgmax": pct(it["forward"]["cpu_vs_device_argmax_agreement"], 1), "Params": f"{best['params'] / 1e6:.1f}M",
         "EvalItems": f"{ev['eval_set']['items']:,}", "EvalAcc": pct(ev["summary"]["overall"]["acc_norm"]),
         "EvalChance": pct(ev["summary"]["overall"]["chance"]), "EvalExcluded": len(ev["excluded"])}
    write(os.path.join(a.out, "macros.tex"), [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in m.items()])

    # loss curve: smoothed training loss (every 4th logged point) and validation loss
    with open(a.curve, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    tr = " ".join(f"({r['step']},{float(r['train_loss_smoothed']):.4f})" for i, r in enumerate(rows) if i % 4 == 0 and int(r["step"]) >= 100)
    va = " ".join(f"({v['step']},{v['val_loss']:.4f})" for v in L["val_curve"])
    write(os.path.join(a.out, "fig_loss.tex"), [
        r"\begin{tikzpicture}",
        r"\begin{axis}[width=0.9\linewidth, height=6cm, xlabel={Step}, ylabel={Loss (cross-entropy)}, ymin=2, ymax=5.5,",
        r"  xmin=0, xmax=" + str(tc["train"]["max_steps"]) + r", legend style={font=\footnotesize}, tick label style={font=\footnotesize},",
        r"  scaled x ticks=false, xticklabel style={/pgf/number format/1000 sep={,}}, grid=major, grid style={gray!20}]",
        r"\addplot[blue!70!black, thick] coordinates {" + tr + "};",
        r"\addplot[red!75!black, thick, mark=*, mark size=1.2pt] coordinates {" + va + "};",
        r"\addplot[gray, dashed] coordinates {(" + str(tc["train"]["max_steps"]) + ",2) (" + str(tc["train"]["max_steps"]) + ",5.5)};",
        r"\legend{Training loss (smoothed), Validation loss, Planned end}",
        r"\end{axis}", r"\end{tikzpicture}"])

    rows = [f"{esc(d)} & {r['tokens']:,} & {r['cross_entropy']:.3f} & {r['perplexity']:.2f} & {pct(r['top1'])} & {pct(r['top5'])} & "
            f"{pct(r['calibration_error'])}" for d, r in sorted(nt.items(), key=lambda kv: kv[1]["perplexity"])]
    write(os.path.join(a.out, "tab_next_token.tex"),
          table("Domain & Tokens & Cross-entropy & Perplexity & Top-1 & Top-5 & Calibration error", "lrrrrrr", rows))

    rows = [f"{esc(d)} & {r['n']} & {pct(r['rep4_reference'], 0)} & {pct(r['rep4'], 0)} & {r['loops']} & {pct(r['rep4_sampled'], 0)} & "
            f"{r['loops_sampled']}" for d, r in g["domains"].items()]
    gd = g["domains"].values()
    rows.append(r"\midrule All & " + f"{sum(r['n'] for r in gd)} & & & {sum(r['loops'] for r in gd)} & & {sum(r['loops_sampled'] for r in gd)}")
    write(os.path.join(a.out, "tab_repetition.tex"),
          table(r"Domain & Texts & Real text & Greedy & Greedy loops & Sampled (T=0.8) & Sampled loops", "lrrrrrr", rows))
    m2 = {"LoopsGreedy": sum(r["loops"] for r in gd), "LoopsSampled": sum(r["loops_sampled"] for r in gd), "LoopTexts": sum(r["n"] for r in gd)}
    with open(os.path.join(a.out, "macros.tex"), "a", encoding="utf-8") as f:
        f.write("\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in m2.items()) + "\n")

    rows = [f"{esc(p['domain'])} & \\texttt{{{esc(p['prompt'].replace(chr(10), ' ')[:44])}}} & {'pass' if p['greedy_pass'] else 'fail'} & "
            f"{sum(p['sample_passes'])}/{len(p['sample_passes'])} & {'yes' if p['loop'] else 'no'}" for p in g["probes"]]
    write(os.path.join(a.out, "tab_probes.tex"), table("Domain & Prompt & Greedy & Samples passing & Loop", "llccc", rows))

    ck = [("File size", lambda c: f"{c['bytes'] / 1e9:.2f} GB"), ("SHA-256 (first 16)", lambda c: f"\\texttt{{{c['sha256'][:16]}}}"),
          ("Step", lambda c: f"{c['step']:,}"), ("Parameters", lambda c: f"{c['params']:,}"), ("Tensors", lambda c: str(c["tensors"])),
          ("All weights finite", lambda c: "yes" if c["all_finite"] else "NO"),
          ("Tokenizer hash matches", lambda c: "yes" if c["vocab_matches"] else "NO"),
          ("Optimizer state saved", lambda c: "yes" if c["has_optimizer"] else "no"),
          ("Validation loss stored", lambda c: "--" if c["val_loss_in_checkpoint"] is None else f"{c['val_loss_in_checkpoint']:.4f}"),
          ("Matches metrics.jsonl", lambda c: "--" if c["val_loss_matches"] is None else ("yes" if c["val_loss_matches"] else "NO")),
          ("Saved with PyTorch", lambda c: esc(c["torch_version"]))]
    rows = [f"{k} & {f(best)} & {f(last) if last else '--'}" for k, f in ck]
    write(os.path.join(a.out, "tab_integrity.tex"), table(r"Check & \texttt{ckpt\_best.pt} & \texttt{ckpt\_last.pt}", "lll", rows))

    doms = ev["summary"]["domain"]
    mix = dict((p.split("/")[-1].split(".")[0], float(p.split("=")[1])) for p in tc["train"]["train_mix"])
    tot_w, seen = sum(mix.values()), (ev["step"] or 0) * tc["train"]["seq_len"] * tc["train"]["batch_size"] * tc["train"]["grad_accum"]
    rows, verdicts = [], {}
    for name, plan_b, bins, bench, val in PLAN:
        tok = sum(mix.get(b, 0) for b in bins) / tot_w * seen
        res = []
        for d in bench:
            v = doms.get(d)
            if v:
                res.append(f"{esc(d)} {pct(v['acc_norm'], 0)} (ch.\\ {pct(v['chance'], 0)}, $n$={v['n']})")
        ppl = f"{nt[val]['perplexity']:.1f}" if val and val in nt else "--"
        if not bins:
            verdict = "not in training data"
        elif not bench:
            verdict = "no benchmark questions"
        elif any(doms[d]["above_chance"] for d in bench if d in doms):
            verdict = "above chance"
        else:
            verdict = "at chance"
        if bench and sum(doms[d]["n"] for d in bench if d in doms) < SMALL:
            verdict += " (few questions)"
        verdicts[name] = verdict
        rows.append(f"{name} & {plan_b:.1f}B & {tok / 1e9:.3f}B & {'; '.join(res) or '--'} & {ppl} & {verdict}")
    write(os.path.join(a.out, "tab_gap.tex"), [r"\begin{tabular}{lrrp{4.7cm}rp{2.9cm}}", r"\toprule",
          r"Domain (30B plan) & Plan & 100M run & Benchmark accuracy & Val.\ ppl & Status \\", r"\midrule"]
          + [r + r" \\" for r in rows] + [r"\bottomrule", r"\end{tabular}"])
    print(f"wrote report pieces to {a.out}")
    print(json.dumps(verdicts, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
