#!/usr/bin/env python
"""Build the whole training set from one JSON config: tokenize every domain (capped), merge per domain, check, print the mix.

    python scripts/build_dataset.py configs/run_100m.json

Config (paths relative to the current directory; ``weights`` are shares of training tokens, normalised later):

    {"vocab": "tokenizers/vocab32k_v2/vocab32k.tok", "out": "bins", "workers": 6, "val_frac": 0.005, "plan_tokens": 2000000000,
     "domains": [
       {"name": "math", "kind": "text",    "paths": ["math/maths.clean.txt.gz"], "max_tokens": 450000000, "weight": 0.20},
       {"name": "code", "kind": "parquet", "paths": ["ravi/code.parquet"],       "sample_rg": 0.09,        "weight": 0.12},
       {"name": "bio",  "kind": "parquet", "paths": ["gokul/biology/*.parquet"], "max_files": 3,           "weight": 0.10},
       {"name": "multistep", "kind": "text", "paths": ["prab/multistep_corpus.txt"], "blank": true, "max_tokens": 60000000, "weight": 0.03}]}

``kind`` is ``text`` (``.txt``/``.gz``; ``blank`` = documents separated by blank lines; ``marker`` = documents between
``<|document|>`` and ``<|endofdocument|>`` lines; ``max_tokens`` caps the read, ``spread`` = take it as that many evenly
spaced slices of a plain-text file instead of a prefix) or
``parquet`` (``sample_rg`` = fraction of evenly spaced row groups; ``max_files`` = only the first N matching files). For every
domain it writes ``<out>/<name>.train.bin`` and ``<name>.val.bin``, then runs the integrity check on all of them and prints the exact
``--train-mix`` / ``--val`` arguments for ``llm.train`` and ``eval_domains.py`` with the epochs each domain will see for
``plan_tokens``. A failure in any step stops the build (nothing is half-merged).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import check_bins as cb  # noqa: E402
import merge_bins as mb  # noqa: E402
import tokenize_parquet as tp  # noqa: E402
import tokenize_text as tt  # noqa: E402
from llm.data import read_header  # noqa: E402


def _expand(paths):
    out = []
    for p in paths:
        out += sorted(glob.glob(p))
    return out


def build_domain(d: dict, cfg: dict, out: str) -> dict:
    name = d["name"]
    work = os.path.join(out, "work", name)
    files = _expand(d["paths"])
    if d.get("max_files"):
        files = files[: int(d["max_files"])]
    if not files:
        raise SystemExit(f"{name}: no input files match {d['paths']}")
    common = ["--vocab", cfg["vocab"], "--out", work, "--workers", str(cfg.get("workers", 4)),
              "--val-frac", str(cfg.get("val_frac", 0.005)), "--no-merge"]
    if d["kind"] == "text":
        args = [f"{name}={f}" for f in files] + common
        if d.get("blank"):
            args += ["--blank-domains", name]
        if d.get("marker"):
            args += ["--marker-domains", name]
        if d.get("max_tokens"):
            args += ["--max-tokens", str(int(d["max_tokens"]))]
        if d.get("spread"):
            args += ["--spread", str(int(d["spread"]))]
        if d.get("dedup"):
            args += ["--dedup"]
        rc = tt.main(args)
    elif d["kind"] == "parquet":
        args = files + common
        if d.get("sample_rg"):
            args += ["--sample-rg", str(d["sample_rg"])]
        if d.get("dedup"):
            args += ["--dedup"]
        rc = tp.main(args)
    else:
        raise SystemExit(f"{name}: kind must be 'text' or 'parquet'")
    if rc != 0:
        raise SystemExit(f"{name}: tokenization failed")
    res = {}
    for split in ("train", "val"):
        parts = sorted(p for p in glob.glob(os.path.join(work, f"*.{split}.bin")))
        if not parts:
            raise SystemExit(f"{name}: no {split} bins were produced")
        dest = os.path.join(out, f"{name}.{split}.bin")
        if mb.main([*parts, "--out", dest, "--split", split]) != 0:
            raise SystemExit(f"{name}: merging {split} failed")
        res[split] = read_header(dest).total_tokens
        if split == "val" and res[split] == 0:          # a tiny domain can have no validation document at this fraction
            os.remove(dest)
            print(f"{name}: no validation documents at this val_frac -> no validation file (raise val_frac to get one)")
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--keep-work", action="store_true", help="keep the per-job intermediate bins (default: delete after merging)")
    a = ap.parse_args(argv)
    cfg = json.load(open(a.config, encoding="utf-8"))
    out = cfg["out"]
    os.makedirs(out, exist_ok=True)
    totals = {}
    for d in cfg["domains"]:
        print(f"\n=== {d['name']} ===", flush=True)
        totals[d["name"]] = build_domain(d, cfg, out)
        if not a.keep_work:
            shutil.rmtree(os.path.join(out, "work", d["name"]), ignore_errors=True)
    bins = sorted(glob.glob(os.path.join(out, "*.bin")))
    print("\n=== integrity check ===")
    if cb.main([*bins, "--vocab", cfg["vocab"]]) != 0:
        raise SystemExit("integrity check FAILED: do not train on these files")
    plan = int(cfg.get("plan_tokens", 0))
    wsum = sum(float(d["weight"]) for d in cfg["domains"])
    print("\ndomain               train tokens   weight   epochs at plan_tokens")
    for d in cfg["domains"]:
        t = totals[d["name"]]["train"]
        share = float(d["weight"]) / wsum
        ep = share * plan / t if plan and t else float("nan")
        print(f"{d['name']:<20}{t:>14,}{share:>9.3f}{ep:>12.2f}" + ("   <-- repeated more than 4x" if ep > 4 else ""))
    mix = " ".join(f"{os.path.join(out, d['name'] + '.train.bin')}={d['weight']}" for d in cfg["domains"])
    vals = " ".join(f"{d['name']}={os.path.join(out, d['name'] + '.val.bin')}" for d in cfg["domains"]
                    if totals[d["name"]]["val"] > 0)
    print(f"\ntrain:  --train-mix {mix}\nevaluate: scripts/eval_domains.py --val {vals}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
