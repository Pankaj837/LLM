#!/usr/bin/env python
"""Turn the results of ``scripts/eval_science.py`` (and ``scripts/check_leakage.py``) into LaTeX tables for the report.

    python scripts/eval_report_tex.py --results results/eval_science_100m_step8250.json \
        --leakage results/leakage_namit_bins.json --out reports/eval_100m/generated

Writes ``macros.tex`` (headline numbers as \\newcommand), one ``tab_*.tex`` per table, ``fig_domain.tex`` (accuracy against
chance per domain, pgfplots) and ``examples.tex``. The hand-written ``main.tex`` \\input's them, so every number in the
report comes from the results file and the tables are rebuilt when the evaluation is re-run.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_science import LEVELS, summarise  # noqa: E402

TEX_ESC = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
           "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}


def esc(s) -> str:
    s = "".join(TEX_ESC.get(c, c) for c in str(s))
    return "".join(c if ord(c) < 128 else {"–": "--", "—": "---", "’": "'", "‘": "`", "“": "``", "”": "''", "°": r"\textdegree{}",
                                            "×": r"$\times$", "µ": r"$\mu$", "Ω": r"$\Omega$", "π": r"$\pi$", "≈": r"$\approx$",
                                            "−": "-", "·": r"$\cdot$",
                                            "≤": r"$\le$", "≥": r"$\ge$", "Δ": r"$\Delta$", "ρ": r"$\rho$", "σ": r"$\sigma$",
                                            "ν": r"$\nu$", "η": r"$\eta$", "θ": r"$\theta$", "λ": r"$\lambda$", **SUPERSCRIPTS}.get(c, "?")
                   for c in s).replace("}\\textsuperscript{", "")


SUPERSCRIPTS = {c: "\\textsuperscript{" + d + "}" for c, d in zip("⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺", "0123456789-+")}


def pct(x: float, d: int = 1) -> str:
    return f"{100 * x:.{d}f}\\%"


def source_group(r) -> str:
    s = r["source"]
    if s.startswith("hand-written"):
        return "Hand-written (PRANA-G)"
    if s.startswith("MMLU"):
        return "MMLU"
    return s.split(" (")[0]          # ARC-Easy, ARC-Challenge, GSM8K


def write(path: str, lines) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def acc_table(groups: dict, label: str) -> list:
    L = [r"\begin{tabular}{lrrcrc}", r"\toprule", f"{label} & $n$ & Accuracy & 95\\% CI & Chance & Above chance \\\\", r"\midrule"]
    for g, v in groups.items():
        L.append(f"{esc(g)} & {v['n']} & {pct(v['acc_norm'])} & {pct(v['ci95'][0])}--{pct(v['ci95'][1])} & {pct(v['chance'])} & "
                 f"{'yes' if v['above_chance'] else 'no'} \\\\")
    return L + [r"\bottomrule", r"\end{tabular}"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True)
    ap.add_argument("--leakage", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--compare", help="results JSON of another checkpoint, for a side-by-side table")
    a = ap.parse_args(argv)
    res = json.load(open(a.results, encoding="utf-8"))
    leak = json.load(open(a.leakage, encoding="utf-8"))
    os.makedirs(a.out, exist_ok=True)
    recs, s = res["items"], res["summary"]
    o = s["overall"]
    mcq = [r for r in recs if r["type"] == "mcq"]
    mcq_sum = summarise(mcq, lambda r: "all")["all"]
    num_sum = summarise([r for r in recs if r["type"] == "numeric"], lambda r: "all")["all"]
    raw_vs_norm = sum(r["correct"] for r in mcq) / len(mcq)

    macros = {
        "EvalItems": res["eval_set"]["items"], "EvalExcluded": len(res["excluded"]), "EvalStep": f"{res['step']:,}",
        "OverallAcc": pct(o["acc_norm"]), "OverallLo": pct(o["ci95"][0]), "OverallHi": pct(o["ci95"][1]),
        "OverallChance": pct(o["chance"]), "McqAcc": pct(mcq_sum["acc_norm"]), "McqAccRaw": pct(raw_vs_norm),
        "McqN": mcq_sum["n"], "McqChance": pct(mcq_sum["chance"]), "McqLo": pct(mcq_sum["ci95"][0]), "McqHi": pct(mcq_sum["ci95"][1]),
        "NumAcc": pct(num_sum["acc_norm"]), "NumN": num_sum["n"], "NumCorrect": sum(r["correct"] for r in recs if r["type"] == "numeric"),
        "DomainsAbove": sum(v["above_chance"] for v in s["domain"].values()), "Domains": len(s["domain"]),
        "EvalMinutes": f"{res['seconds'] / 60:.0f}", "EvalDevice": esc(res["device"]),
        "RepeatItems": res["repeat_check"]["items"], "RepeatOK": "identical" if res["repeat_check"]["identical"] else "DIFFERENT",
        "LeakBins": len(leak["bins"]), "LeakLeaked": leak["counts"]["leaked"], "LeakPartial": leak["counts"]["partial"],
        "LeakClean": leak["counts"]["clean"], "LeakShort": leak["counts"]["too_short"],
    }
    write(os.path.join(a.out, "macros.tex"), [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in macros.items()])

    write(os.path.join(a.out, "tab_domain.tex"), acc_table(s["domain"], "Domain"))
    write(os.path.join(a.out, "tab_source.tex"), acc_table(summarise(recs, source_group), "Source"))
    # answer-length bias: how often the longest option is the right one, and how often the model picks the longest
    L = [r"\begin{tabular}{lrrrr}", r"\toprule",
         r"Source & $n$ & Longest option is right & Model accuracy & Model picks longest \\", r"\midrule"]
    groups = {}
    for r in mcq:
        groups.setdefault(source_group(r), []).append(r)
    for g, rs in sorted(groups.items()):
        longest = [max(range(len(r["choices"])), key=lambda i: len(r["choices"][i])) for r in rs]
        L.append(f"{esc(g)} & {len(rs)} & {pct(sum(lg == r['answer'] for lg, r in zip(longest, rs)) / len(rs))} & "
                 f"{pct(sum(r['correct_norm'] for r in rs) / len(rs))} & {pct(sum(lg == r['pick_norm'] for lg, r in zip(longest, rs)) / len(rs))} \\\\")
    write(os.path.join(a.out, "tab_length.tex"), L + [r"\bottomrule", r"\end{tabular}"])
    write(os.path.join(a.out, "tab_type_skill.tex"), acc_table({**{f"type: {k}": v for k, v in s["type"].items()},
                                                               **{f"skill: {k}": v for k, v in s["skill"].items()}}, "Group"))

    L = [r"\begin{tabular}{l" + "c" * len(LEVELS) + "}", r"\toprule", "Domain & " + " & ".join(l.capitalize() for l in LEVELS) + r" \\", r"\midrule"]
    for d in s["domain"]:
        cells = []
        for lv in LEVELS:
            v = s["domain_level"].get(f"{d} / {lv}")
            cells.append(f"{pct(v['acc_norm'], 0)} \\scriptsize{{($n$={v['n']}, ch.\\ {pct(v['chance'], 0)})}}" if v else "--")
        L.append(f"{esc(d)} & " + " & ".join(cells) + r" \\")
    write(os.path.join(a.out, "tab_level.tex"), L + [r"\bottomrule", r"\end{tabular}"])

    errs = sorted(res["errors"].items(), key=lambda kv: -kv[1])
    hw_mcq = [r for r in recs if r["type"] == "mcq" and r.get("distractor_types")]
    hw_wrong = [r for r in hw_mcq if not r["correct_norm"]]
    L = [r"\begin{tabular}{lr}", r"\toprule", r"Error type & Count \\", r"\midrule"]
    L += [f"{esc(e)} & {n} \\\\" for e, n in errs]
    write(os.path.join(a.out, "tab_errors.tex"), L + [r"\bottomrule", r"\end{tabular}"])
    # error picks among the labelled hand-written MCQs, against how often each wrong-option type is offered
    offered, picked = {}, {}
    for r in hw_mcq:
        for i, t in enumerate(r["distractor_types"]):
            if t != "correct":
                offered[t] = offered.get(t, 0) + 1
        if not r["correct_norm"]:
            picked[r["error"]] = picked.get(r["error"], 0) + 1
    L = [r"\begin{tabular}{lrrr}", r"\toprule", r"Wrong-option type & Offered & Picked & Picked / offered \\", r"\midrule"]
    for t in sorted(offered, key=lambda t: -picked.get(t, 0)):
        L.append(f"{esc(t)} & {offered[t]} & {picked.get(t, 0)} & {pct(picked.get(t, 0) / offered[t], 0)} \\\\")
    write(os.path.join(a.out, "tab_distractors.tex"), L + [r"\bottomrule", r"\end{tabular}"])
    gsm_all = [r for r in recs if r.get("fewshot") == "gsm8k"]
    macros2 = {"HwMcqN": len(hw_mcq), "HwMcqWrong": len(hw_wrong),
               "GsmN": len(gsm_all), "GsmCorrect": sum(r["correct"] for r in gsm_all),
               "GsmAnswerLine": sum("answer is" in r["output"] for r in gsm_all),
               "GsmTagged": sum("Benchmark source" in r["output"] for r in gsm_all),
               "GsmLoops": sum(r["error"] == "repetition_loop" for r in gsm_all)}
    if a.compare:
        other = json.load(open(a.compare, encoding="utf-8"))
        so = other["summary"]
        L = [r"\begin{tabular}{lrrr}", r"\toprule",
             f"Domain & Step {other['step']:,} & Step {res['step']:,} & Difference \\\\", r"\midrule"]
        for d, v in list(s["domain"].items()) + [("overall", o)]:
            w = so["overall"] if d == "overall" else so["domain"][d]
            L.append(f"{esc(d)} & {pct(w['acc_norm'])} & {pct(v['acc_norm'])} & {100 * (v['acc_norm'] - w['acc_norm']):+.1f} \\\\")
        write(os.path.join(a.out, "tab_compare.tex"), L + [r"\bottomrule", r"\end{tabular}"])
        other_pick = {r["id"]: (r.get("pick_norm"), r.get("pred")) for r in other["items"]}
        same = sum(other_pick.get(r["id"]) == (r.get("pick_norm"), r.get("pred")) for r in recs)
        macros2.update({"CompareStep": f"{other['step']:,}", "CompareAgree": pct(same / len(recs)),
                        "CompareAcc": pct(so["overall"]["acc_norm"])})

    by_dom = {}
    for key, v in leak["by_domain_source"].items():
        d = key.split(" | ")[0]
        t = by_dom.setdefault(d, {"n": 0, "leaked": 0, "partial": 0, "clean": 0, "too_short": 0})
        for k in t:
            t[k] += v[k]
    L = [r"\begin{tabular}{lrrrrr}", r"\toprule", r"Domain & $n$ & Leaked & Partial & Clean & Too short \\", r"\midrule"]
    for d, t in sorted(by_dom.items()):
        L.append(f"{esc(d)} & {t['n']} & {t['leaked']} & {t['partial']} & {t['clean']} & {t['too_short']} \\\\")
    write(os.path.join(a.out, "tab_leak.tex"), L + [r"\bottomrule", r"\end{tabular}"])
    write(os.path.join(a.out, "leaked_ids.tex"), [r"\begin{itemize}\raggedright"] + [
        f"\\item \\texttt{{{esc(q)}}} ({leak['items'][q]['matched']}/{leak['items'][q]['ngrams']} windows matched, "
        f"in \\texttt{{{esc(leak['items'][q]['first_hit'][0])}}})" for q in leak["flagged_ids"]] + [r"\end{itemize}"])

    doms = list(s["domain"])
    coords = lambda f: " ".join(f"({100 * f(s['domain'][d]):.1f},{esc(d)})" for d in doms)
    errs_pm = " ".join(f"({100 * s['domain'][d]['acc_norm']:.1f},{esc(d)}) +- ({100 * (s['domain'][d]['ci95'][1] - s['domain'][d]['acc_norm']):.1f},0) -= "
                       f"({100 * (s['domain'][d]['acc_norm'] - s['domain'][d]['ci95'][0]):.1f},0)" for d in doms)
    write(os.path.join(a.out, "fig_domain.tex"), [
        r"\begin{tikzpicture}",
        r"\begin{axis}[xbar, y=0.62cm, bar width=0.36cm, xmin=0, xmax=100, xlabel={Accuracy (\%)},",
        f"  symbolic y coords={{{','.join(esc(d) for d in doms)}}}, ytick=data, y dir=reverse, enlarge y limits=0.06,",
        r"  legend style={at={(0.5,1.02)}, anchor=south, legend columns=2, font=\footnotesize}, tick label style={font=\footnotesize},",
        r"  xmajorgrids, grid style={gray!25}]",
        r"\addplot+[fill=blue!55, draw=blue!70!black, error bars/.cd, x dir=both, x explicit] coordinates {" + errs_pm + "};",
        r"\addplot+[only marks, mark=|, mark size=5pt, very thick, red!80!black] coordinates {" + coords(lambda v: v["chance"]) + "};",
        r"\legend{Model (95\% CI), Chance}",
        r"\end{axis}", r"\end{tikzpicture}"])

    ex = []
    gsm = [r for r in recs if r.get("fewshot") == "gsm8k"]
    sci = [r for r in recs if r["type"] == "numeric" and r.get("fewshot") != "gsm8k"]
    for r in [x for x in gsm if x["correct"]][:1] + [x for x in gsm if not x["correct"]][:2] + sci[:3]:
        out = re.sub(r"\s+", " ", r["output"]).strip()[:260]
        ex += [r"\item[] \textbf{" + esc(r["id"]) + "} (" + ("correct" if r["correct"] else esc(r["error"])) + r")\\",
               r"\textit{Q:} " + esc(r["question"][:260]) + r"\\",
               r"\textit{Expected:} " + esc(r["answer"]) + r" \quad \textit{Model:} \texttt{" + esc(out) + "}"]
    for r in [x for x in hw_mcq if not x["correct_norm"]][:3] + [x for x in hw_mcq if x["correct_norm"]][:1]:
        ex += [r"\item[] \textbf{" + esc(r["id"]) + "} (" + ("correct" if r["correct_norm"] else "picked a " + esc(r["error"]) + " option") + r")\\",
               r"\textit{Q:} " + esc(r["question"][:260]) + r"\\",
               r"\textit{Right:} " + esc(r["choices"][r["answer"]]) + r" \quad \textit{Model picked:} " + esc(r["choices"][r["pick_norm"]])]
    write(os.path.join(a.out, "examples.tex"), [r"\begin{itemize}\setlength\itemsep{0.6em}"] + ex + [r"\end{itemize}"])
    with open(os.path.join(a.out, "macros.tex"), "a", encoding="utf-8") as f:
        f.write("\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in macros2.items()) + "\n")
    print(f"wrote LaTeX pieces to {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
