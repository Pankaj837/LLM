#!/usr/bin/env python
"""Build the frozen scientific evaluation set in ``evals/``: public benchmark questions plus the hand-written ones.

    python scripts/build_eval_set.py              # downloads the public files (pinned revisions), writes evals/public/*.jsonl
    python scripts/build_eval_set.py --manifest   # only re-hashes evals/**/*.jsonl into evals/MANIFEST.json

Public sources (test splits only, each at a pinned dataset revision so the selection never changes):
  * ARC (allenai/ai2_arc, CC BY-SA 4.0): grade-school science questions, Easy = basic, Challenge = intermediate
  * MMLU (cais/mmlu, MIT): maths, physics, chemistry, computer science and electrical engineering subjects
  * GSM8K (openai/gsm8k, MIT): multi-step word problems with a numeric answer
  * PubMedQA (qiaojin/PubMedQA, MIT): yes/no questions about a research abstract (scientific discovery / reading)

MMLU also gives biology, astronomy, economics and accounting ("finance"). ARC questions whose text mentions an earth-science
term (``EARTH_TERMS``) form the ``earth`` domain; the other ARC questions stay in ``sciqa``.

Which questions are taken does not depend on a random generator: the ``cap`` items with the smallest SHA-1 of their source
id are kept, so the same ids come out on every machine. The hand-written files in ``evals/handwritten/`` are not touched;
the manifest records the SHA-256 of every file, so any later change to the frozen set is visible.

Needs ``pyarrow`` and ``huggingface_hub`` (only this script; scoring does not).
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
EVALS = os.path.join(ROOT, "evals")

ARC = ("allenai/ai2_arc", "210d026faf9955653af8916fad021475a3f00453", "CC BY-SA 4.0")
MMLU = ("cais/mmlu", "c30699e8356da336a370243923dbaf21066bb9fe", "MIT")
GSM8K = ("openai/gsm8k", "740312add88f781978c0658806c59bc2815b9866", "MIT")
PUBMEDQA = ("qiaojin/PubMedQA", "9001f2853fb87cab8d220904e0de81ac6973b318", "MIT")

# (file in evals/public, dataset, parquet path, domain, level, cap)
MMLU_SUBJECTS = [
    ("math", "elementary_mathematics", "basic", 150),
    ("math", "high_school_mathematics", "intermediate", 150),
    ("math", "college_mathematics", "advanced", 100),
    ("physics", "conceptual_physics", "basic", 150),
    ("physics", "high_school_physics", "intermediate", 150),
    ("physics", "college_physics", "advanced", 102),
    ("chemistry", "high_school_chemistry", "intermediate", 150),
    ("chemistry", "college_chemistry", "advanced", 100),
    ("code", "high_school_computer_science", "intermediate", 100),
    ("code", "college_computer_science", "advanced", 100),
    ("electrical_eng", "electrical_engineering", "intermediate", 145),
    ("bio", "high_school_biology", "intermediate", 150),
    ("bio", "college_biology", "advanced", 140),
    ("astro", "astronomy", "intermediate", 150),
    ("finance", "high_school_macroeconomics", "intermediate", 150),
    ("finance", "econometrics", "advanced", 110),
    ("finance", "professional_accounting", "advanced", 100),
]
ARC_SPLITS = [("ARC-Easy", "basic", 300), ("ARC-Challenge", "intermediate", 300)]
EARTH_CAP = 150          # per ARC split
EARTH_TERMS = re.compile(
    r"\b(rocks?|minerals?|earthquakes?|volcan\w*|weather\w*|climates?|erosion|erod\w*|glacier\w*|sediment\w*|igneous|"
    r"metamorphic|fossils?|tectonic|atmospher\w*|tides?|clouds?|precipitation|hurricanes?|tornado\w*|crust|mantle|magma|"
    r"lava|landforms?|canyons?|groundwater|water cycle|humidity|ozone|greenhouse|global warming|droughts?|air mass\w*|"
    r"cold front|warm front)\b", re.I)
PUBMEDQA_PER_LABEL = 100  # balanced yes / no, so always answering "yes" scores 50%
GSM8K_CAP = 200
GSM8K_SHOTS = 4          # worked examples from the *train* split, shown before every GSM8K question


def stable_pick(rows: list, key, cap: int) -> list:
    """The cap rows with the smallest SHA-1 of key(row), returned in that order (independent of any RNG)."""
    return sorted(rows, key=lambda r: hashlib.sha1(str(key(r)).encode()).hexdigest())[:cap]


def read_parquet(repo: str, rev: str, path: str) -> list:
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    return pq.read_table(hf_hub_download(repo, path, repo_type="dataset", revision=rev)).to_pylist()


def gsm8k_number(answer: str) -> float:
    """GSM8K answers end with '#### 1,234'."""
    return float(answer.split("####")[-1].strip().replace(",", ""))


def gsm8k_worked(answer: str) -> str:
    """The worked solution without the calculator annotations <<2*3=6>>."""
    return re.sub(r"<<[^>]*>>", "", answer.split("####")[0]).strip()


def build_public() -> dict:
    out = {}
    repo, rev, lic = MMLU
    for domain, subject, level, cap in MMLU_SUBJECTS:
        rows = read_parquet(repo, rev, f"{subject}/test-00000-of-00001.parquet")
        for i, r in enumerate(rows):
            r["_id"] = f"{subject}/test/{i}"
        # a few source rows repeat a choice (two options with the same text); they are left out
        rows = [r for r in rows if len({str(c).strip() for c in r["choices"]}) == len(r["choices"])]
        for r in stable_pick(rows, lambda r: r["_id"], cap):
            out.setdefault(f"mmlu_{domain}", []).append({
                "id": f"mmlu-{r['_id']}", "domain": domain, "level": level, "type": "mcq",
                "question": r["question"].strip(), "choices": [str(c).strip() for c in r["choices"]],
                "answer": int(r["answer"]), "source": f"MMLU {subject} (test)", "source_id": r["_id"],
                "licence": lic})
    repo, rev, lic = ARC
    for split, level, cap in ARC_SPLITS:
        rows = read_parquet(repo, rev, f"{split}/test-00000-of-00001.parquet")
        earth = [r for r in rows if EARTH_TERMS.search(r["question"])]
        rest = [r for r in rows if not EARTH_TERMS.search(r["question"])]
        for name, domain, picked in (("arc", "sciqa", stable_pick(rest, lambda r: r["id"], cap)),
                                     ("arc_earth", "earth", stable_pick(earth, lambda r: r["id"], EARTH_CAP))):
            for r in picked:
                labels = list(r["choices"]["label"])
                out.setdefault(name, []).append({
                    "id": f"arc-{r['id']}", "domain": domain, "level": level, "type": "mcq",
                    "question": r["question"].strip(), "choices": [t.strip() for t in r["choices"]["text"]],
                    "answer": labels.index(r["answerKey"]), "source": f"{split} (test)", "source_id": r["id"],
                    "licence": lic})
    repo, rev, lic = PUBMEDQA
    rows = read_parquet(repo, rev, "pqa_labeled/train-00000-of-00001.parquet")
    for label in ("yes", "no"):
        for r in stable_pick([r for r in rows if r["final_decision"] == label], lambda r: r["pubid"], PUBMEDQA_PER_LABEL):
            abstract = " ".join(c.strip() for c in r["context"]["contexts"])
            out.setdefault("pubmedqa", []).append({
                "id": f"pubmedqa-{r['pubid']}", "domain": "scidisc", "level": "advanced", "type": "mcq",
                "question": f"{abstract}\n{r['question'].strip()}", "choices": ["yes", "no"],
                "answer": 0 if label == "yes" else 1, "source": "PubMedQA (expert-labelled)", "source_id": str(r["pubid"]),
                "licence": lic})
    repo, rev, lic = GSM8K
    train = read_parquet(repo, rev, "main/train-00000-of-00001.parquet")
    shots = [{"question": r["question"].strip(), "solution": gsm8k_worked(r["answer"]),
              "answer": gsm8k_number(r["answer"])} for r in train[:GSM8K_SHOTS]]
    test = read_parquet(repo, rev, "main/test-00000-of-00001.parquet")
    for i, r in enumerate(test):
        r["_id"] = f"test/{i}"
    for r in stable_pick(test, lambda r: r["_id"], GSM8K_CAP):
        out.setdefault("gsm8k", []).append({
            "id": f"gsm8k-{r['_id']}", "domain": "multistep", "level": "intermediate", "type": "numeric",
            "question": r["question"].strip(), "answer": gsm8k_number(r["answer"]), "tolerance": 0.0,
            "fewshot": "gsm8k", "source": "GSM8K (test)", "source_id": r["_id"], "licence": lic})
    os.makedirs(os.path.join(EVALS, "public"), exist_ok=True)
    with open(os.path.join(EVALS, "public", "fewshot_gsm8k.json"), "w", encoding="utf-8") as f:
        json.dump({"source": "GSM8K train split, first 4 rows", "licence": lic, "shots": shots}, f, indent=1, ensure_ascii=False)
    for name, items in out.items():
        with open(os.path.join(EVALS, "public", f"{name}.jsonl"), "w", encoding="utf-8") as f:
            for it in items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
    return {k: len(v) for k, v in out.items()}


def write_manifest(evals_dir: str = EVALS) -> dict:
    files = sorted(glob.glob(os.path.join(evals_dir, "**", "*.jsonl"), recursive=True)
                   + glob.glob(os.path.join(evals_dir, "*", "fewshot_*.json")))
    entries, totals = [], {}
    for p in files:
        data = open(p, "rb").read()
        n = sum(1 for line in data.decode("utf-8").splitlines() if line.strip()) if p.endswith(".jsonl") else None
        entries.append({"file": os.path.relpath(p, evals_dir).replace(os.sep, "/"), "sha256": hashlib.sha256(data).hexdigest(), "items": n})
        if p.endswith(".jsonl"):
            for line in data.decode("utf-8").splitlines():
                if line.strip():
                    d = json.loads(line)["domain"]
                    totals[d] = totals.get(d, 0) + 1
    man = {"name": "prana-g-science-eval", "version": "2.1",
           "sources": {"ARC": {"repo": ARC[0], "revision": ARC[1], "licence": ARC[2]},
                       "MMLU": {"repo": MMLU[0], "revision": MMLU[1], "licence": MMLU[2]},
                       "GSM8K": {"repo": GSM8K[0], "revision": GSM8K[1], "licence": GSM8K[2]},
                       "PubMedQA": {"repo": PUBMEDQA[0], "revision": PUBMEDQA[1], "licence": PUBMEDQA[2]},
                       "handwritten": {"licence": "PRANA-G internal, written for this evaluation"}},
           "items_by_domain": dict(sorted(totals.items())), "items": sum(totals.values()), "files": entries}
    with open(os.path.join(evals_dir, "MANIFEST.json"), "w", encoding="utf-8") as f:
        json.dump(man, f, indent=1)
    return man


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", action="store_true", help="only rewrite evals/MANIFEST.json")
    a = ap.parse_args(argv)
    if not a.manifest:
        print("public items:", build_public())
    man = write_manifest()
    print(f"MANIFEST: {man['items']} items, {len(man['files'])} files, by domain {man['items_by_domain']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
