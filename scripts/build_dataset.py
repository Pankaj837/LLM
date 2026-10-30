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
``<|document|>`` and ``<|endofdocument|>`` lines; ``eot`` = documents separated by lines reading ``<|endoftext|>``; ``max_tokens`` caps the read, ``spread`` = take it as that many evenly
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
import re
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import check_bins as cb  # noqa: E402
import merge_bins as mb  # noqa: E402
import tokenize_parquet as tp  # noqa: E402
import tokenize_text as tt  # noqa: E402
from llm.data import read_header  # noqa: E402


FROZEN_MD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tokenizers", "FROZEN.md")


def frozen_sha() -> str:
    """The SHA-256 of the frozen tokenizer, read from tokenizers/FROZEN.md."""
    m = re.search(r"SHA-256\s*\|\s*`([0-9a-f]{64})`", open(FROZEN_MD, encoding="utf-8").read())
    if not m:
        raise SystemExit(f"cannot find the SHA-256 in {FROZEN_MD}")
    return m.group(1)


def require_frozen_vocab(vocab: str) -> None:
    """Refuse any other vocabulary: the two 32K candidates have the same size, EOS id and file name, so nothing else would notice."""
    from llm.tokenizer import BPETokenizer

    got = BPETokenizer.file_sha256(vocab)
    if got != frozen_sha():
        raise SystemExit(f"{vocab} is NOT the frozen tokenizer (sha256 {got[:16]}..., frozen {frozen_sha()[:16]}...). "
                         "Use tokenizers/vocab32k_v2/vocab32k.tok (git checkout scale-run-rc6), or pass --allow-unfrozen for a rehearsal.")


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
        if d.get("eot"):
            args += ["--eot-domains", name]
        if d.get("start"):
            args += ["--start-domains", name, "--start-prefix", str(d["start"])]
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


def write_build_manifest(cfg: dict, totals: dict, out: str, config_path: str) -> None:
    """Provenance next to the bins: which tokenizer, which code, which inputs, how many tokens. The bins alone cannot tell."""
    import hashlib
    from datetime import datetime, timezone

    from llm.tokenizer import BPETokenizer
    from llm.train import run_info

    inputs = {d["name"]: [{"path": p, "bytes": os.path.getsize(p)} for p in _expand(d["paths"])[: int(d.get("max_files") or 10**9)]]
              for d in cfg["domains"]}
    manifest = {"built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "run_info": run_info(),
                "tokenizer_sha256": BPETokenizer.file_sha256(cfg["vocab"]), "config": cfg,
                "config_sha256": hashlib.sha256(open(config_path, "rb").read()).hexdigest(), "tokens": totals, "inputs": inputs}
    with open(os.path.join(out, "build_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--keep-work", action="store_true", help="keep the per-job intermediate bins (default: delete after merging)")
    ap.add_argument("--allow-unfrozen", action="store_true", help="rehearsal only: accept a vocabulary other than the frozen one")
    a = ap.parse_args(argv)
    cfg = json.load(open(a.config, encoding="utf-8"))
    if not a.allow_unfrozen:
        require_frozen_vocab(cfg["vocab"])           # before any tokenizing: hours of work on the wrong vocabulary are worthless
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
    # weights are shares of the WHOLE mix: when this config holds only one owner's domains, ``weights_sum`` (the sum of all
    # domains' weights in the full config) keeps the epoch report honest
    wsum = float(cfg.get("weights_sum") or sum(float(d["weight"]) for d in cfg["domains"]))
    print("\ndomain               train tokens   weight   epochs at plan_tokens")
    for d in cfg["domains"]:
        t = totals[d["name"]]["train"]
        share = float(d["weight"]) / wsum
        ep = share * plan / t if plan and t else float("nan")
        print(f"{d['name']:<20}{t:>14,}{share:>9.3f}{ep:>12.2f}" + ("   <-- repeated more than 4x" if ep > 4 else ""))
    write_build_manifest(cfg, totals, out, a.config)
    mix = " ".join(f"{os.path.join(out, d['name'] + '.train.bin')}={d['weight']}" for d in cfg["domains"])
    vals = " ".join(f"{d['name']}={os.path.join(out, d['name'] + '.val.bin')}" for d in cfg["domains"]
                    if totals[d["name"]]["val"] > 0)
    print(f"\ntrain:  --train-mix {mix}\nevaluate: scripts/eval_domains.py --val {vals}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
