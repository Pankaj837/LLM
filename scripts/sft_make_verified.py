#!/usr/bin/env python
"""Generate SFT examples whose answers are correct BY CONSTRUCTION (computed, not written by a model), in ``sft_prepare.py`` input format.

    python scripts/sft_make_verified.py --out raw/verified_synthetic.jsonl --per-kind 2000 --seed 0

Kinds (``domain``): ``arith`` multi-digit multiplication / addition / subtraction with a digit-spaced worked scratchpad (math), ``poly``
derivative of a polynomial by the power rule (math), ``physics`` one-formula numeric problems (Ohm, kinematics, density, kinetic energy,
ideal gas) with the formula, the substitution and the unit (physics), ``code`` small Python functions with ``assert`` tests (code).
Every row carries ``answer`` (and ``tests`` for code), so ``sft_prepare.py`` re-checks the final answer (and ``--verify-code`` runs the tests).

What this is NOT: a substitute for a real, diverse dataset. It is narrow and templated (the same lesson as the CAD domain: a model reaches a
near-zero loss on templates without learning anything general). Use it as a small part of the pilot (a few thousand rows) aimed at one
measured weakness (multi-digit arithmetic written step by step, units), keep the evaluation questions out of it (``sft_prepare.py --evals``) and
judge it only by held-out problems of the same kind AND by the untouched benchmarks. Prompts are unique (random operands); the seed fixes the file.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from typing import Any, Dict, List


def spaced(n: int) -> str:
    return " ".join(str(abs(n))) if n >= 0 else "- " + " ".join(str(abs(n)))


def arith(rng: random.Random) -> Dict[str, Any]:
    op = rng.choice(["*", "*", "+", "-"])
    if op == "*":
        a, b = rng.randint(12, 99), rng.randint(12, 99)
        parts = [(int(d) * 10 ** (len(str(b)) - 1 - i), int(d)) for i, d in enumerate(str(b))]      # tens and units of b
        partial = [a * v for v, _ in parts if v]
        ans = a * b
        steps = " + ".join(f"{a} * {v}" for v, _ in parts if v) + " = " + " + ".join(str(p) for p in partial) + f" = {ans}"
        q = rng.choice([f"Question: What is {a} * {b}?", f"Compute {a} times {b}.", f"What is the product of {a} and {b}?"])
        r = f"Split {b} into tens and units. {steps}.\nDigits: {spaced(a)} x {spaced(b)} = {spaced(ans)}.\nThe answer is {ans}."
    else:
        a, b = rng.randint(100, 9999), rng.randint(100, 9999)
        ans = a + b if op == "+" else a - b
        word = "plus" if op == "+" else "minus"
        q = rng.choice([f"Question: What is {a} {op} {b}?", f"Compute {a} {word} {b}."])
        r = f"{a} {op} {b}.\nDigits: {spaced(a)} {op} {spaced(b)} = {spaced(ans)}.\nThe answer is {ans}."
    return {"prompt": q, "response": r, "answer": ans, "domain": "math", "source": "synthetic:arith"}


def poly(rng: random.Random) -> Dict[str, Any]:
    exps = sorted(rng.sample(range(1, 7), rng.randint(2, 3)), reverse=True)
    coef = [rng.choice([c for c in range(-9, 10) if c]) for _ in exps]

    def term(c: int, e: int, first: bool) -> str:
        body = "x" if e == 1 else f"x^{e}" if e > 1 else ""
        mag = "" if abs(c) == 1 and body else str(abs(c))
        sign = ("-" if c < 0 else "") if first else (" - " if c < 0 else " + ")
        return f"{sign}{mag}{body}"

    f = "".join(term(c, e, i == 0) for i, (c, e) in enumerate(zip(coef, exps)))
    d = [(c * e, e - 1) for c, e in zip(coef, exps)]
    g = "".join(term(c, e, i == 0) for i, (c, e) in enumerate(d))
    steps = "; ".join(f"d/dx({term(c, e, True)}) = {term(c * e, e - 1, True)}" for c, e in zip(coef, exps))
    return {"prompt": f"Differentiate f(x) = {f} with respect to x.",
            "response": f"Use the power rule d/dx(c x^n) = c n x^(n-1) on each term: {steps}.\nThe answer is f'(x) = {g}", "answer": g,
            "domain": "math", "source": "synthetic:poly"}


def _fmt(x: float) -> str:
    return f"{float(f'{x:.4g}'):g}"


def physics(rng: random.Random) -> Dict[str, Any]:
    k = rng.choice(["ohm", "speed", "density", "kinetic", "gas"])
    if k == "ohm":
        i, r = round(rng.uniform(0.2, 9.0), 1), rng.randint(2, 470)
        v = i * r
        return {"prompt": f"A current of {i} A flows through a {r} ohm resistor. What is the voltage across it?",
                "response": f"Ohm's law: V = I R.\nV = {i} A * {r} ohm = {_fmt(v)} V.\nThe answer is {_fmt(v)} V.", "answer": float(_fmt(v)),
                "domain": "physics", "source": "synthetic:physics"}
    if k == "speed":
        s, t = rng.randint(20, 900), rng.randint(2, 90)
        v = s / t
        return {"prompt": f"A car covers {s} m in {t} s at constant speed. What is its speed in m/s?",
                "response": f"Speed = distance / time.\nv = {s} m / {t} s = {_fmt(v)} m/s.\nThe answer is {_fmt(v)} m/s.", "answer": float(_fmt(v)),
                "domain": "physics", "source": "synthetic:physics"}
    if k == "density":
        m, vol = rng.randint(50, 900), rng.randint(5, 120)
        d = m / vol
        return {"prompt": f"A sample has a mass of {m} g and a volume of {vol} cm^3. What is its density in g/cm^3?",
                "response": f"Density = mass / volume.\nrho = {m} g / {vol} cm^3 = {_fmt(d)} g/cm^3.\nThe answer is {_fmt(d)} g/cm^3.", "answer": float(_fmt(d)),
                "domain": "physics", "source": "synthetic:physics"}
    if k == "kinetic":
        m, v = rng.randint(2, 90), rng.randint(2, 40)
        e = 0.5 * m * v * v
        return {"prompt": f"What is the kinetic energy of a {m} kg object moving at {v} m/s?",
                "response": f"E = (1/2) m v^2.\nE = 0.5 * {m} kg * ({v} m/s)^2 = 0.5 * {m} * {v * v} = {_fmt(e)} J.\nThe answer is {_fmt(e)} J.",
                "answer": float(_fmt(e)), "domain": "physics", "source": "synthetic:physics"}
    n, t, vol = round(rng.uniform(0.5, 5.0), 1), rng.randint(250, 450), rng.randint(2, 40)
    p = n * 8.314 * t / (vol / 1000.0)
    return {"prompt": f"What is the pressure in Pa of {n} mol of an ideal gas at {t} K in a volume of {vol} L? Use R = 8.314 J/(mol K).",
            "response": f"Ideal gas law: P = n R T / V, with V = {vol} L = {vol / 1000:g} m^3.\nP = {n} * 8.314 * {t} / {vol / 1000:g} = {_fmt(p)} Pa.\nThe answer is {_fmt(p)} Pa.",
            "answer": float(_fmt(p)), "domain": "physics", "source": "synthetic:physics"}


def code(rng: random.Random) -> Dict[str, Any]:
    k = rng.choice(["sumsq", "mul", "clamp", "evens", "mean"])
    n = rng.randint(2, 9)
    if k == "sumsq":
        name = f"sum_squares_{n}"
        c = f"def {name}():\n    return sum(i * i for i in range(1, {n} + 1))"
        t = f"assert {name}() == {sum(i * i for i in range(1, n + 1))}"
        q = f"Write a Python function {name}() that returns the sum of the squares of the integers from 1 to {n}."
    elif k == "mul":
        name = f"times_{n}"
        c = f"def {name}(x):\n    return x * {n}"
        t = f"assert {name}(7) == {7 * n}\nassert {name}(0) == 0"
        q = f"Write a Python function {name}(x) that returns x multiplied by {n}."
    elif k == "clamp":
        lo, hi = rng.randint(0, 5), rng.randint(10, 20)
        name = f"clamp_{lo}_{hi}"
        c = f"def {name}(x):\n    return max({lo}, min({hi}, x))"
        t = f"assert {name}({lo - 1}) == {lo}\nassert {name}({hi + 1}) == {hi}\nassert {name}({lo + 1}) == {lo + 1}"
        q = f"Write a Python function {name}(x) that limits x to the range from {lo} to {hi} inclusive."
    elif k == "evens":
        name = f"count_evens_below_{n * 5}"
        c = f"def {name}():\n    return len([i for i in range({n * 5}) if i % 2 == 0])"
        t = f"assert {name}() == {len([i for i in range(n * 5) if i % 2 == 0])}"
        q = f"Write a Python function {name}() that returns how many even integers are in range({n * 5})."
    else:
        name = f"mean_plus_{n}"
        c = f"def {name}(xs):\n    return sum(xs) / len(xs) + {n}"
        t = f"assert {name}([2, 4, 6]) == {4 + n}"
        q = f"Write a Python function {name}(xs) that returns the arithmetic mean of the list xs plus {n}."
    return {"prompt": q, "response": f"```python\n{c}\n```", "tests": t, "domain": "code", "source": "synthetic:code"}


KINDS = {"arith": arith, "poly": poly, "physics": physics, "code": code}


def make(per_kind: int, seed: int, kinds: List[str]) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    rows, seen = [], set()
    for kind in kinds:
        made, tries = 0, 0
        while made < per_kind and tries < per_kind * 50:
            tries += 1
            ex = KINDS[kind](rng)
            if ex["prompt"] in seen:
                continue
            seen.add(ex["prompt"])
            rows.append(ex)
            made += 1
    rng.shuffle(rows)
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-kind", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--kinds", nargs="+", default=list(KINDS), choices=list(KINDS))
    a = ap.parse_args(argv)
    rows = make(a.per_kind, a.seed, a.kinds)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} examples to {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
