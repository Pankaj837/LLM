#!/usr/bin/env python
"""Turn raw instruction-response files into a checked SFT dataset: train / val / test ``.jsonl`` files plus ``manifest.json``.

    python scripts/sft_prepare.py --input raw/math.jsonl raw/code.jsonl --vocab tokenizers/vocab32k_v2/vocab32k.tok \
        --out-dir sft/v1 --evals evals --val-frac 0.05 --test-frac 0.05

Input: one JSON object per line with ``prompt`` (alias ``instruction``) and ``response`` (alias ``output``); optional ``system``,
``domain``, ``source``, ``id``, ``answer`` (reference answer: a number or a short text) and ``tests`` (Python ``assert`` lines for a code
response). Every example is checked and the reason for every removal is counted in the manifest:

  * ``invalid``          missing / empty / non-string field, bad JSON
  * ``special_text``     the text contains a special-token string (``<|user|>`` ...), which would confuse the chat format
  * ``too_long``         more than ``--max-len`` tokens in the chat format (never truncated)
  * ``duplicate``        same prompt and response after whitespace/case normalisation (first one kept)
  * ``duplicate_prompt`` same prompt with a different response (first kept; one example per prompt keeps every prompt inside one split)
  * ``contradictory``    same prompt, different ``answer`` fields: ALL of them are removed
  * ``answer_mismatch``  the response's final answer differs from its ``answer`` field
  * ``code_test_failed`` (only with ``--verify-code``) the response's code block fails its ``tests``. This EXECUTES the code: run it on a
                         disposable machine only (see ``sft_verify.py``).
  * ``contaminated``     (with ``--evals``) the example contains an evaluation question: the same 13-token-window rule as
                         ``scripts/check_leakage.py`` (at least 20% of the question's windows, and at least 2, or at least 10 windows)

The split is a function of the normalised prompt and ``--seed`` only (no random state): the same prompt always lands in the same split,
and no prompt appears in two splits. The manifest records counts, token counts (tokens that carry loss, in the chat format of
``llm.sft``), per-domain counts, removal reasons, the SHA-256 of every input and output file, and the vocabulary hash.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)
from llm.sft import DEFAULT_END, encode_example, encode_plain  # noqa: E402
from llm.tokenizer import BPETokenizer  # noqa: E402
from sft_verify import answer_matches, extract_code_block, run_python_tests  # noqa: E402

_WS = re.compile(r"\s+")
CANON = ("id", "system", "prompt", "response", "domain", "source", "answer", "tests")


def norm(text: str) -> str:
    return _WS.sub(" ", text).strip().casefold()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_rows(paths: Sequence[str], reasons: Counter, notes: List[str]) -> List[Dict[str, Any]]:
    rows = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for n, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError:
                    reasons["invalid"] += 1
                    notes.append(f"{os.path.basename(p)}:{n}: bad JSON")
                    continue
                prompt = raw.get("prompt", raw.get("instruction"))
                response = raw.get("response", raw.get("output"))
                sysm = raw.get("system")
                if (not isinstance(prompt, str) or not prompt.strip() or not isinstance(response, str) or not response.strip()
                        or (sysm is not None and not isinstance(sysm, str))):
                    reasons["invalid"] += 1
                    notes.append(f"{os.path.basename(p)}:{n}: needs non-empty string prompt and response")
                    continue
                ex = {"id": raw.get("id"), "system": sysm or None, "prompt": prompt.strip(), "response": response.strip(),
                      "domain": raw.get("domain") or "unknown", "source": raw.get("source") or os.path.basename(p)}
                if "answer" in raw:
                    ex["answer"] = raw["answer"]
                if raw.get("tests"):
                    ex["tests"] = raw["tests"]
                rows.append(ex)
    return rows


class EvalIndex:
    """13-token windows of the evaluation questions, for the contamination rule of ``check_leakage.py``."""

    def __init__(self, items: Sequence[Dict[str, Any]], tok: BPETokenizer, n: int = 13, min_n: int = 8,
                 threshold: float = 0.2, min_hits: int = 2, min_matched: int = 10):
        from check_leakage import ngram_hashes
        self._hash = ngram_hashes
        self.threshold, self.min_hits, self.min_matched = threshold, min_hits, min_matched
        self.tables: Dict[int, Dict[int, List[str]]] = {}
        self.size: Dict[str, int] = {}
        for it in items:
            ids = encode_plain(tok, it["question"])
            k = min(n, len(ids))
            if k < min_n:
                continue
            hs = np.unique(ngram_hashes(ids, k))
            self.size[it["id"]] = len(hs)
            for h in hs:
                self.tables.setdefault(k, {}).setdefault(int(h), []).append(it["id"])
        self.keys = {k: np.array(sorted(t), dtype=np.uint64) for k, t in self.tables.items()}

    def leaked_ids(self, ids: Sequence[int]) -> List[str]:
        counts: Counter = Counter()
        for k, table in self.tables.items():
            if len(ids) < k:
                continue
            hs = np.unique(self._hash(ids, k))
            keys = self.keys[k]
            pos = np.minimum(np.searchsorted(keys, hs), len(keys) - 1)
            for h in hs[keys[pos] == hs]:
                for qid in table[int(h)]:
                    counts[qid] += 1
        return sorted(q for q, c in counts.items()
                      if (c / self.size[q] >= self.threshold and c >= self.min_hits) or c >= self.min_matched)


def split_of(prompt: str, seed: int, val_frac: float, test_frac: float) -> str:
    u = int(hashlib.sha256(f"{seed}:{norm(prompt)}".encode("utf-8")).hexdigest()[:8], 16) / 2 ** 32
    return "test" if u < test_frac else "val" if u < test_frac + val_frac else "train"


def prepare(rows: List[Dict[str, Any]], tok: BPETokenizer, *, max_len: int = 2048, val_frac: float = 0.05, test_frac: float = 0.05,
            seed: int = 0, eval_index: Optional[EvalIndex] = None, verify_code: bool = False, code_timeout: float = 10.0,
            end_token: str = DEFAULT_END, reasons: Optional[Counter] = None) -> Dict[str, Any]:
    """Filter and split ``rows``; returns ``{"splits": {name: [rows]}, "reasons": Counter, "contaminated": {id: [qids]}, ...}``."""
    reasons = reasons if reasons is not None else Counter()
    specials = list(tok.special_to_id)
    contaminated: Dict[str, List[str]] = {}

    # contradictory: same prompt, different reference answers -> remove every one of them
    answers: Dict[str, set] = defaultdict(set)
    for ex in rows:
        if "answer" in ex:
            answers[norm(ex["prompt"])].add(json.dumps(ex["answer"], sort_keys=True))
    contradictory = {p for p, a in answers.items() if len(a) > 1}

    seen_pair, seen_prompt = set(), set()
    kept: List[Dict[str, Any]] = []
    for ex in rows:
        key_p = norm(ex["prompt"])
        key_pair = key_p + "\x00" + norm(ex["response"])
        text = ex["prompt"] + "\n" + ex["response"] + "\n" + (ex.get("system") or "")
        if any(s in text for s in specials):
            reasons["special_text"] += 1
            continue
        if key_p in contradictory:
            reasons["contradictory"] += 1
            continue
        if key_pair in seen_pair:
            reasons["duplicate"] += 1
            continue
        if key_p in seen_prompt:
            reasons["duplicate_prompt"] += 1
            continue
        e = encode_example(tok, ex, max_len, end_token)
        if e is None:
            reasons["too_long"] += 1
            continue
        if "answer" in ex and not answer_matches(ex["response"], ex["answer"]):
            reasons["answer_mismatch"] += 1
            continue
        if verify_code and ex.get("tests"):
            ok, _ = run_python_tests(extract_code_block(ex["response"]), ex["tests"], code_timeout)
            if not ok:
                reasons["code_test_failed"] += 1
                continue
        if eval_index is not None:
            hits = eval_index.leaked_ids(e["x"] + e["y"][-1:])
            if hits:
                reasons["contaminated"] += 1
                contaminated[ex.get("id") or key_pair[:12]] = hits
                continue
        seen_pair.add(key_pair)
        seen_prompt.add(key_p)
        if not ex.get("id"):
            ex["id"] = hashlib.sha1(key_pair.encode("utf-8")).hexdigest()[:12]
        kept.append(ex)

    splits: Dict[str, List[Dict[str, Any]]] = {"train": [], "val": [], "test": []}
    for ex in kept:
        splits[split_of(ex["prompt"], seed, val_frac, test_frac)].append(ex)
    return {"splits": splits, "reasons": reasons, "contaminated": contaminated}


def split_stats(rows: List[Dict[str, Any]], tok: BPETokenizer, max_len: int, end_token: str) -> Dict[str, Any]:
    sup = tot = 0
    by_domain: Counter = Counter()
    for ex in rows:
        e = encode_example(tok, ex, max_len, end_token)
        sup += e["n_response"]
        tot += len(e["x"])
        by_domain[ex["domain"]] += 1
    return {"examples": len(rows), "supervised_tokens": sup, "total_tokens": tot, "per_domain": dict(sorted(by_domain.items()))}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", nargs="+", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--evals", nargs="*", default=[], help="evaluation folders (the evals/ layout) to decontaminate against")
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--test-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--verify-code", action="store_true", help="execute each example's code against its tests (disposable machine only)")
    ap.add_argument("--code-timeout", type=float, default=10.0)
    ap.add_argument("--end-token", default=DEFAULT_END)
    a = ap.parse_args(argv)
    if not (0 <= a.val_frac and 0 <= a.test_frac and a.val_frac + a.test_frac < 1):
        ap.error("val-frac and test-frac must be >= 0 and sum to less than 1")

    tok = BPETokenizer.from_file(a.vocab)
    reasons: Counter = Counter()
    notes: List[str] = []
    rows = read_rows(a.input, reasons, notes)
    index = None
    n_eval = 0
    if a.evals:
        from eval_science import load_items
        items = [it for d in a.evals for it in load_items(d)]
        n_eval = len(items)
        index = EvalIndex(items, tok)
    out = prepare(rows, tok, max_len=a.max_len, val_frac=a.val_frac, test_frac=a.test_frac, seed=a.seed, eval_index=index,
                  verify_code=a.verify_code, code_timeout=a.code_timeout, end_token=a.end_token, reasons=reasons)

    os.makedirs(a.out_dir, exist_ok=True)
    files: Dict[str, Any] = {}
    for name, rs in out["splits"].items():
        path = os.path.join(a.out_dir, f"{name}.jsonl")
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            for ex in rs:
                f.write(json.dumps({k: ex[k] for k in CANON if ex.get(k) is not None}, ensure_ascii=False) + "\n")
        files[name] = {"file": os.path.basename(path), "sha256": sha256_file(path), **split_stats(rs, tok, a.max_len, a.end_token)}
    manifest = {"inputs": [{"file": os.path.basename(p), "sha256": sha256_file(p)} for p in a.input], "rows_read": len(rows) + reasons["invalid"],
                "kept": sum(f["examples"] for f in files.values()), "removed": dict(reasons), "first_notes": notes[:20],
                "splits": files, "contaminated": out["contaminated"], "eval_questions_checked": n_eval,
                "vocab_sha256": BPETokenizer.file_sha256(a.vocab), "max_len": a.max_len, "seed": a.seed, "val_frac": a.val_frac,
                "test_frac": a.test_frac, "end_token": a.end_token, "code_verified": bool(a.verify_code),
                "note": "tokens counted in the llm.sft chat format; 'supervised_tokens' are the response tokens plus the end token"}
    with open(os.path.join(a.out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"read {manifest['rows_read']}, kept {manifest['kept']} (train {files['train']['examples']}, val {files['val']['examples']}, "
          f"test {files['test']['examples']}); removed {dict(reasons)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
