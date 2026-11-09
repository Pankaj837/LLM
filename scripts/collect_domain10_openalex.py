#!/usr/bin/env python
"""Domain 10 (Physics + AI) top-up collector: open-licence (CC BY / CC0) full-text papers found through OpenAlex.

Why: Swaraj's v7 corpus holds 13.06M tokens (930 papers from peS2o and the arXiv Common Pile subset); Raju asked for 60 to 80M for the 1B pilot.
Both of his sources are exhausted, so this script looks for papers that are NOT in them: OpenAlex works whose best open-access location is
CC BY or CC0, whose title or abstract mentions the Domain 10 topics, and whose PDF can be downloaded.

    python scripts/collect_domain10_openalex.py list  --out d10 --held <manifest.jsonl> [...]      # candidates -> d10/candidates.jsonl
    python scripts/collect_domain10_openalex.py fetch --out d10 --lane other                        # download + extract (all hosts except arXiv)
    python scripts/collect_domain10_openalex.py fetch --out d10 --lane arxiv                        # arXiv, one request per 3 s (arXiv policy)
    python scripts/collect_domain10_openalex.py build --out d10                                     # d10/domain10_openalex.clean.txt + manifest

Every document keeps its OpenAlex id, DOI, title, licence, source URL and text hash. PDFs are never stored (only extracted text).
Rejected papers are counted per reason in d10/fetch_<lane>.log. The run is resumable: finished ids are skipped.
Run it with ``python -I`` (the downloaded files are data only).
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import io
import json
import os
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request

UA = "PRANA-G-corpus-collector/1.0 (open-licence research papers)"
QUERY = ('"physics-informed neural network" OR "physics informed neural network" OR "neural operator" OR "operator learning" OR DeepONet '
         'OR "Fourier neural operator" OR "physics-informed machine learning" OR "scientific machine learning" OR "physics-guided" '
         'OR "neural PDE" OR "differentiable physics" OR "neural surrogate" OR "physics-informed deep learning" OR "physics-based neural network"')
TOPIC = re.compile(r"physics[- ]informed|neural operator|operator learning|deeponet|fourier neural operator|scientific machine learning|"
                   r"physics[- ]guided|physics[- ]based neural|differentiable physics|neural surrogate|\bPINNs?\b|neural PDE|neural ODE", re.I)
STOP = set("the of and to in is a that for with as are on by this be from an we which at it or can has was were".split())


def norm_title(t: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", unicodedata.normalize("NFKD", t or "").lower()).strip()


def sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def get_json(url: str, tries: int = 8):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 502, 503, 504) or i == tries - 1:
                raise
            time.sleep(20 * (i + 1) if e.code == 429 else 5 * (i + 1))      # rate limit: wait, then retry the same page
        except Exception:  # noqa: BLE001
            if i == tries - 1:
                raise
            time.sleep(5 * (i + 1))


# ── stage 1: candidates ─────────────────────────────────────────────────────

def stage_list(a) -> None:
    held = set()
    for p in a.held:
        for line in open(p, encoding="utf-8"):
            if line.strip():
                held.add(norm_title(json.loads(line).get("title", "")))
    print(f"held titles: {len(held):,}")
    flt = f"title_and_abstract.search:({QUERY}),best_oa_location.license:cc-by|cc0,language:en,type:article|preprint"
    page, n, kept, spent = 1, 0, 0, 0.0
    os.makedirs(a.out, exist_ok=True)
    seen = set()
    with open(os.path.join(a.out, "candidates.jsonl"), "w", encoding="utf-8") as out:
        while page:
            q = urllib.parse.urlencode({"filter": flt, "per-page": 200, "page": page,
                                        "select": "id,doi,title,publication_year,best_oa_location,type"})
            r = get_json("https://api.openalex.org/works?" + q)
            time.sleep(2.0)                                         # polite: the keyless free budget is shared and rate-limited
            spent += r["meta"].get("cost_usd") or 0
            for w in r["results"]:
                n += 1
                loc = w.get("best_oa_location") or {}
                pdf = loc.get("pdf_url")
                if not pdf or not w.get("title") or w["id"] in seen:
                    continue
                seen.add(w["id"])
                if norm_title(w["title"]) in held:
                    continue
                kept += 1
                out.write(json.dumps({"id": w["id"], "doi": w.get("doi"), "title": w["title"], "year": w.get("publication_year"),
                                      "license": loc.get("license"), "pdf_url": pdf, "landing": loc.get("landing_page_url"),
                                      "source": (loc.get("source") or {}).get("display_name"), "type": w.get("type")}) + "\n")
            page = page + 1 if r["results"] and n < r["meta"]["count"] and page < 50 else 0   # basic paging needs no API key (max 10,000 results)
            print(f"  seen {n:,}  candidates {kept:,}  OpenAlex cost so far ${spent:.3f}", flush=True)
    print(f"candidates (not held, with PDF url): {kept:,}")


# ── stage 2: download + extract ─────────────────────────────────────────────

LIG = {"ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl", "­": ""}


def clean_text(raw: str) -> str:
    t = raw.encode("utf-8", "ignore").decode("utf-8")      # PDF text can hold lone surrogates (math alphanumerics): drop them
    t = unicodedata.normalize("NFC", t)
    for k, v in LIG.items():
        t = t.replace(k, v)
    t = re.sub(r"\(cid:\d+\)", "", t)
    t = re.sub(r"[​‌‍⁠﻿�]", "", t)
    t = t.replace("\r", "\n")
    t = re.sub(r"-\n(?=[a-z])", "", t)                       # hyphenated line breaks
    lines = [ln.strip() for ln in t.split("\n")]
    # running headers / footers: short lines that repeat 3 or more times
    from collections import Counter
    cnt = Counter(ln for ln in lines if 0 < len(ln) < 120)
    lines = [ln for ln in lines if not (cnt.get(ln, 0) >= 3 and len(ln) < 120 and not ln.endswith("."))]
    lines = [ln for ln in lines if not re.fullmatch(r"(page )?\d{1,4}( of \d{1,4})?|arxiv:\S+.*|\d{1,4}\s*/\s*\d{1,4}", ln, re.I)]
    # cut the reference list (last "References" heading in the final 45% of the text)
    body = "\n".join(lines)
    ms = [m for m in re.finditer(r"(?im)^\s*(references|bibliography|literature cited)\s*$", body)]
    ms = [m for m in ms if m.start() > 0.55 * len(body)]
    if ms:
        body = body[:ms[0].start()]
        lines = body.split("\n")
    # paragraphs: a line shorter than 60% of the median line length ends a paragraph
    lens = sorted(len(x) for x in lines if len(x) > 30)
    med = lens[len(lens) // 2] if lens else 80
    paras, cur = [], []
    for ln in lines:
        if not ln:
            if cur:
                paras.append(" ".join(cur)); cur = []
            continue
        cur.append(ln)
        if len(ln) < 0.6 * med and ln.endswith((".", ":", "?", "!")):
            paras.append(" ".join(cur)); cur = []
    if cur:
        paras.append(" ".join(cur))
    out = "\n\n".join(re.sub(r"[ \t]+", " ", p).strip() for p in paras if p.strip())
    return out.strip()


def quality(text: str) -> str | None:
    words = text.split()
    if len(words) < 1500:
        return "short (<1500 words)"
    alpha = sum(c.isalpha() for c in text) / max(len(text), 1)
    if alpha < 0.62:
        return "low alphabetic ratio (math/garbage)"
    sw = sum(1 for w in words[:4000] if w.lower() in STOP) / min(len(words), 4000)
    if sw < 0.18:
        return "not English prose"
    if len(TOPIC.findall(text)) < 3:
        return "topic term fewer than 3 times in the text"
    return None


def fetch_one(c: dict, pages: int = 120, max_mb: int = 40):
    from pypdf import PdfReader
    try:
        req = urllib.request.Request(c["pdf_url"], headers={"User-Agent": UA, "Accept": "application/pdf"})
        with urllib.request.urlopen(req, timeout=60) as r:
            data = r.read(max_mb * 1024 * 1024 + 1)
    except Exception as e:  # noqa: BLE001
        return {"id": c["id"], "reject": "download: " + type(e).__name__}
    if len(data) > max_mb * 1024 * 1024:
        return {"id": c["id"], "reject": "pdf larger than %d MB" % max_mb}
    if not data.startswith(b"%PDF"):
        return {"id": c["id"], "reject": "not a PDF (landing page)"}
    try:
        rd = PdfReader(io.BytesIO(data))
        txt = "\n".join((p.extract_text() or "") for p in rd.pages[:pages])
    except Exception as e:  # noqa: BLE001
        return {"id": c["id"], "reject": "pdf parse: " + type(e).__name__}
    try:
        text = clean_text(txt)
    except Exception as e:  # noqa: BLE001
        return {"id": c["id"], "reject": "clean: " + type(e).__name__}
    bad = quality(text)
    if bad:
        return {"id": c["id"], "reject": bad}
    return {"id": c["id"], "ok": True, "doi": c["doi"], "title": c["title"], "year": c["year"], "license": c["license"],
            "pdf_url": c["pdf_url"], "landing": c["landing"], "source": c["source"], "n_pages": len(rd.pages),
            "words": len(text.split()), "text_sha256": sha(text), "text": text}


def stage_fetch(a) -> None:
    cands = [json.loads(x) for x in open(os.path.join(a.out, "candidates.jsonl"), encoding="utf-8")]
    is_arxiv = lambda c: "arxiv.org" in (c["pdf_url"] or "")  # noqa: E731
    cands = [c for c in cands if is_arxiv(c) == (a.lane == "arxiv")]
    res_path = os.path.join(a.out, f"results_{a.lane}.jsonl")
    done = set()
    if os.path.exists(res_path):
        for x in open(res_path, encoding="utf-8"):
            if x.strip():
                done.add(json.loads(x)["id"])
    todo = [c for c in cands if c["id"] not in done]
    if a.limit:
        todo = todo[:a.limit]
    print(f"lane {a.lane}: {len(cands):,} candidates, {len(done):,} done, {len(todo):,} to do", flush=True)
    ok = rej = 0
    reasons: dict = {}
    t0 = time.time()
    with open(res_path, "a", encoding="utf-8") as out:
        if a.lane == "arxiv":                                   # one request per 3 s
            it = ((c, fetch_one(c)) for c in todo)
            def gen():
                for c in todo:
                    t = time.time()
                    yield c, fetch_one(c)
                    time.sleep(max(0.0, 3.0 - (time.time() - t)))
            results = gen()
        else:
            ex = cf.ProcessPoolExecutor(max_workers=a.workers)
            futs = {ex.submit(fetch_one, c): c for c in todo}
            results = ((futs[f], f.result()) for f in cf.as_completed(futs))
        for i, (c, r) in enumerate(results, 1):
            out.write(json.dumps(r, ensure_ascii=False) + "\n")
            out.flush()
            if r.get("ok"):
                ok += 1
            else:
                rej += 1
                reasons[r["reject"]] = reasons.get(r["reject"], 0) + 1
            if i % 50 == 0:
                print(f"  {i:,}/{len(todo):,}  kept {ok:,}  rejected {rej:,}  {time.time() - t0:.0f}s", flush=True)
    print(f"lane {a.lane} finished: kept {ok:,}, rejected {rej:,}")
    for k, v in sorted(reasons.items(), key=lambda kv: -kv[1]):
        print(f"  {v:6,}  {k}")


# ── stage 3: build ──────────────────────────────────────────────────────────

def stage_build(a) -> None:
    seen_h, seen_t, docs = set(), set(), []
    for lane in ("other", "arxiv"):
        p = os.path.join(a.out, f"results_{lane}.jsonl")
        if not os.path.exists(p):
            continue
        for x in open(p, encoding="utf-8"):
            r = json.loads(x)
            if not r.get("ok"):
                continue
            nt = norm_title(r["title"])
            if r["text_sha256"] in seen_h or nt in seen_t:
                continue
            seen_h.add(r["text_sha256"]); seen_t.add(nt)
            docs.append(r)
    txt = os.path.join(a.out, "domain10_openalex.clean.txt")
    with open(txt, "w", encoding="utf-8", newline="\n") as f, open(os.path.join(a.out, "domain10_openalex_manifest.jsonl"), "w", encoding="utf-8") as m:
        for r in docs:
            f.write(r["text"].strip() + "\n<|endoftext|>\n")
            m.write(json.dumps({k: v for k, v in r.items() if k != "text"}, ensure_ascii=False) + "\n")
    print(f"{len(docs):,} documents, {sum(d['words'] for d in docs):,} words -> {txt}")
    print("tokenize with: python scripts/tokenize_text.py domain10_openalex=<txt> --eot-domains domain10_openalex --val-frac 0.02 "
          "--vocab tokenizers/vocab32k_v2/vocab32k.tok --out <bins>")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["list", "fetch", "build"])
    ap.add_argument("--out", default="d10")
    ap.add_argument("--held", nargs="*", default=[], help="manifest .jsonl files whose titles are already held (excluded)")
    ap.add_argument("--lane", choices=["other", "arxiv"], default="other")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    {"list": stage_list, "fetch": stage_fetch, "build": stage_build}[a.stage](a)


if __name__ == "__main__":
    main()
