#!/usr/bin/env python
"""Compare two ``eval_science.py`` result files: did a re-run reproduce a reference, or how do two checkpoints differ?

    # reproduction check (same checkpoint, new machine): exit 0 only if the run matches the reference
    python scripts/compare_eval.py --ref results/eval_science_100m_step8250.json --new my_rerun.json --reproduce

    # two different checkpoints on the same frozen question set: table of gains and regressions
    python scripts/compare_eval.py --ref results/eval_science_100m_step8250.json --new results/eval_science_pilot.json

Both files must hold the same question ids and the same tokenizer hash (otherwise the comparison is meaningless and the
script stops). Per item it compares ``correct_norm`` (the headline score); for multiple choice it also reports how many
picks and how large the log-probability differences are, which is what a different device or library version changes.
``--reproduce`` passes when the overall score differs by at most ``--tol-overall`` points and at most ``--tol-items`` of the
items change correctness (defaults 0.5 points and 2%): GPU versus CPU rounding can flip a handful of near-tie choices, a wrong
checkpoint, question set or scorer changes far more.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict


def _load(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _acc(items) -> float:
    return sum(bool(i["correct_norm"]) for i in items) / max(len(items), 1)


def _wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def compare(ref: dict, new: dict) -> dict:
    """Pure function: the comparison as a dict (see ``main`` for the printed form). Raises ValueError if the files cannot be compared."""
    if ref.get("vocab_sha256") != new.get("vocab_sha256"):
        raise ValueError("different tokenizer hashes: the scores are not comparable")
    ri = {i["id"]: i for i in ref["items"]}
    ni = {i["id"]: i for i in new["items"]}
    if set(ri) != set(ni):
        only_r, only_n = sorted(set(ri) - set(ni)), sorted(set(ni) - set(ri))
        raise ValueError(f"different question sets: {len(only_r)} ids only in ref (e.g. {only_r[:3]}), {len(only_n)} only in new (e.g. {only_n[:3]})")
    ids = sorted(ri)
    flipped = [i for i in ids if bool(ri[i]["correct_norm"]) != bool(ni[i]["correct_norm"])]
    gained = [i for i in flipped if ni[i]["correct_norm"]]
    lost = [i for i in flipped if ri[i]["correct_norm"]]
    pick_diff = [i for i in ids if ri[i]["type"] == "mcq" and ri[i].get("pick_norm") != ni[i].get("pick_norm")]
    lp = [max(abs(a - b) for a, b in zip(ri[i]["logprobs"], ni[i]["logprobs"])) for i in ids
          if ri[i]["type"] == "mcq" and ri[i].get("logprobs") and ni[i].get("logprobs")]
    by_dom = defaultdict(lambda: ([], []))
    for i in ids:
        by_dom[ri[i]["domain"]][0].append(ri[i])
        by_dom[ri[i]["domain"]][1].append(ni[i])
    domains = []
    for d in sorted(by_dom):
        a, b = by_dom[d]
        ka, kb = sum(bool(x["correct_norm"]) for x in a), sum(bool(x["correct_norm"]) for x in b)
        lo, hi = _wilson(kb, len(b))
        domains.append({"domain": d, "n": len(a), "ref": ka / len(a), "new": kb / len(b), "diff": (kb - ka) / len(a),
                        "new_ci95": [lo, hi], "items_changed": sum(bool(x["correct_norm"]) != bool(y["correct_norm"]) for x, y in zip(a, b))})
    return {"n": len(ids), "ref_overall": _acc([ri[i] for i in ids]), "new_overall": _acc([ni[i] for i in ids]),
            "items_changed": len(flipped), "gained": len(gained), "lost": len(lost), "mcq_pick_changed": len(pick_diff),
            "max_logprob_diff": max(lp) if lp else 0.0, "median_logprob_diff": sorted(lp)[len(lp) // 2] if lp else 0.0, "domains": domains}


def verdict(c: dict, tol_overall: float, tol_items: float) -> tuple:
    problems = []
    if abs(c["new_overall"] - c["ref_overall"]) * 100 > tol_overall + 1e-9:
        problems.append(f"overall differs by {abs(c['new_overall'] - c['ref_overall']) * 100:.2f} points (limit {tol_overall})")
    if c["items_changed"] / c["n"] > tol_items + 1e-12:
        problems.append(f"{c['items_changed']} of {c['n']} items changed correctness ({c['items_changed'] / c['n']:.1%}, limit {tol_items:.1%})")
    return (not problems, problems)


def render(c: dict, ref_name: str, new_name: str) -> str:
    out = [f"ref: {ref_name}", f"new: {new_name}", f"items: {c['n']}",
           f"overall acc_norm: ref {c['ref_overall']:.2%}, new {c['new_overall']:.2%} (diff {100 * (c['new_overall'] - c['ref_overall']):+.2f} points)",
           f"items whose correctness changed: {c['items_changed']} ({c['items_changed'] / c['n']:.2%}); gained {c['gained']}, lost {c['lost']}",
           f"multiple-choice picks changed: {c['mcq_pick_changed']}; log-probability difference: median {c['median_logprob_diff']:.4f}, max {c['max_logprob_diff']:.4f}",
           "", "| domain | n | ref | new | diff (points) | new 95% CI | items changed |", "|---|---|---|---|---|---|---|"]
    for d in c["domains"]:
        out.append(f"| {d['domain']} | {d['n']} | {d['ref']:.1%} | {d['new']:.1%} | {100 * d['diff']:+.1f} | "
                   f"{d['new_ci95'][0]:.1%}-{d['new_ci95'][1]:.1%} | {d['items_changed']} |")
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--reproduce", action="store_true", help="pass/fail check that --new reproduces --ref")
    ap.add_argument("--tol-overall", type=float, default=0.5, help="allowed overall difference in points (default 0.5)")
    ap.add_argument("--tol-items", type=float, default=0.02, help="allowed share of items that change correctness (default 0.02)")
    ap.add_argument("--out", help="also write the table to this .md file")
    args = ap.parse_args(argv)
    try:
        c = compare(_load(args.ref), _load(args.new))
    except ValueError as exc:
        print(f"CANNOT COMPARE: {exc}")
        return 2
    text = render(c, args.ref, args.new)
    ok, problems = verdict(c, args.tol_overall, args.tol_items)
    if args.reproduce:
        text += "\n\n" + ("REPRODUCED" if ok else "NOT REPRODUCED: " + "; ".join(problems))
    print(text)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text + "\n")
    return 0 if (ok or not args.reproduce) else 1


if __name__ == "__main__":
    sys.exit(main())
