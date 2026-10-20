#!/usr/bin/env python
"""Turn PubMed baseline XML (``pubmedNNnNNNN.xml.gz``, loose or inside a ZIP) into Parquet shards with a ``text`` column.

Streams each file (never unzips to disk, low memory) and keeps one document per citation that has an abstract:

    <ArticleTitle>

    <Label>: <abstract section text>   (one paragraph per AbstractText)

Columns: ``doc_id`` (``pmid:<PMID>``), ``domain`` (``medical``), ``text``, ``char_count``. Output feeds
``scripts/tokenize_parquet.py`` directly. English-only by default; citations without an abstract are dropped.

    python scripts/pubmed_to_parquet.py "medical data.zip" --out data/medical --workers 8
    python scripts/pubmed_to_parquet.py path/to/xml_gz_folder --out data/medical --limit 3     # quick check

Resumable: a finished shard (``<name>.parquet``) is skipped on re-run.
"""
from __future__ import annotations

import argparse
import glob
import gzip
import os
import sys
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from xml.etree import ElementTree as ET

BATCH = 20_000


def _text(el) -> str:
    return "".join(el.itertext()).strip() if el is not None else ""


def parse_articles(stream, min_chars: int = 100, english_only: bool = True):
    """Yield ``(pmid, text)`` for every PubmedArticle in an XML stream that has a usable abstract."""
    for _, el in ET.iterparse(stream, events=("end",)):
        if el.tag != "PubmedArticle":
            continue
        cit = el.find("MedlineCitation")
        art = cit.find("Article") if cit is not None else None
        if art is not None:
            langs = [(x.text or "").strip().lower() for x in art.findall("Language")]
            if not english_only or not langs or "eng" in langs:
                parts = []
                for ab in art.findall("Abstract/AbstractText"):
                    body = _text(ab)
                    if body:
                        label = (ab.get("Label") or "").strip()
                        parts.append(f"{label.capitalize()}: {body}" if label else body)
                abstract = "\n".join(parts)
                if len(abstract) >= min_chars:
                    title = _text(art.find("ArticleTitle"))
                    pmid = _text(cit.find("PMID"))
                    yield pmid, (f"{title}\n\n{abstract}" if title else abstract)
        el.clear()


def process_member(job) -> dict:
    """job = (source, member_name_or_None, out_dir, min_chars, english_only, limit_docs). Returns per-file stats."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    source, member, out_dir, min_chars, english_only, limit_docs = job
    name = Path(member or source).name
    stem = name[:-len(".xml.gz")] if name.endswith(".xml.gz") else Path(name).stem
    out = Path(out_dir) / f"{stem}.parquet"
    if out.exists():
        meta = pq.ParquetFile(out).metadata
        return {"file": name, "docs": meta.num_rows, "chars": -1, "skipped": True}
    out.parent.mkdir(parents=True, exist_ok=True)
    zf = zipfile.ZipFile(source) if member else None
    raw = zf.open(member) if zf else open(source, "rb")
    schema = pa.schema([("doc_id", pa.string()), ("domain", pa.string()), ("text", pa.string()), ("char_count", pa.int64())])
    tmp = out.with_suffix(".parquet.tmp")
    docs = chars = 0
    ids: list = []
    texts: list = []
    try:
        with gzip.open(raw, "rb") as stream, pq.ParquetWriter(tmp, schema, compression="zstd") as w:
            def flush():
                if ids:
                    w.write_table(pa.table({"doc_id": ids, "domain": ["medical"] * len(ids), "text": texts,
                                            "char_count": [len(t) for t in texts]}, schema=schema))
                    ids.clear()
                    texts.clear()
            for pmid, text in parse_articles(stream, min_chars, english_only):
                ids.append(f"pmid:{pmid}")
                texts.append(text)
                docs += 1
                chars += len(text)
                if len(ids) >= BATCH:
                    flush()
                if limit_docs and docs >= limit_docs:
                    break
            flush()
    finally:
        raw.close()
        if zf:
            zf.close()
    os.replace(tmp, out)
    return {"file": name, "docs": docs, "chars": chars, "skipped": False}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", help="a .zip containing *.xml.gz, or a folder / glob of *.xml.gz")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--min-chars", type=int, default=100, help="minimum abstract length to keep a citation")
    ap.add_argument("--all-languages", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="only the first N files (quick check)")
    ap.add_argument("--limit-docs", type=int, default=0, help="stop each file after N documents (quick check)")
    a = ap.parse_args(argv)

    if a.source.lower().endswith(".zip"):
        with zipfile.ZipFile(a.source) as zf:
            members = sorted(n for n in zf.namelist() if n.endswith(".xml.gz"))
        jobs = [(a.source, m, a.out, a.min_chars, not a.all_languages, a.limit_docs) for m in members]
    else:
        paths = sorted(glob.glob(os.path.join(a.source, "**", "*.xml.gz"), recursive=True)) if os.path.isdir(a.source) \
            else sorted(glob.glob(a.source))
        jobs = [(p, None, a.out, a.min_chars, not a.all_languages, a.limit_docs) for p in paths]
    if a.limit:
        jobs = jobs[:a.limit]
    if not jobs:
        ap.error("no .xml.gz files found")
    t0 = time.time()
    total_docs = total_chars = 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for i, r in enumerate(ex.map(process_member, jobs), 1):
            total_docs += r["docs"]
            total_chars += max(r["chars"], 0)
            if i % 10 == 0 or i == len(jobs):
                print(f"[{i}/{len(jobs)}] docs={total_docs:,} chars={total_chars:,} {time.time() - t0:.0f}s", flush=True)
    print(f"done: {total_docs:,} documents, {total_chars:,} characters (~{total_chars // 4:,} tokens at 4 chars/token)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
