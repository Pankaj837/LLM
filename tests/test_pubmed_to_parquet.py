"""scripts/pubmed_to_parquet.py: PubMed XML (loose or in a ZIP) -> Parquet with a text column."""
import gzip
import sys
import zipfile
from pathlib import Path

import pytest

pq = pytest.importorskip("pyarrow.parquet")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import pubmed_to_parquet as pp  # noqa: E402

LONG = "Mitochondria produce ATP through oxidative phosphorylation in nearly every eukaryotic cell type studied."


def _article(pmid, title, abstract_xml, lang="eng"):
    return (f"<PubmedArticle><MedlineCitation><PMID>{pmid}</PMID><Article><Language>{lang}</Language>"
            f"<ArticleTitle>{title}</ArticleTitle>{abstract_xml}</Article></MedlineCitation></PubmedArticle>")


XML = ("<?xml version='1.0'?><PubmedArticleSet>"
       + _article(1, "Energy in cells", f"<Abstract><AbstractText Label='BACKGROUND'>{LONG}</AbstractText>"
                                         f"<AbstractText Label='RESULTS'>Results with <i>italic</i> text are kept whole and readable here.</AbstractText></Abstract>")
       + _article(2, "No abstract here", "")
       + _article(3, "Short abstract", "<Abstract><AbstractText>Too short.</AbstractText></Abstract>")
       + _article(4, "Texte français", f"<Abstract><AbstractText>{LONG}</AbstractText></Abstract>", lang="fre")
       + "</PubmedArticleSet>")


def test_parse_keeps_only_english_articles_with_a_real_abstract():
    import io
    got = list(pp.parse_articles(io.BytesIO(XML.encode())))
    assert [p for p, _ in got] == ["1"]
    text = got[0][1]
    assert text.startswith("Energy in cells\n\nBackground: " + LONG)
    assert "Results: Results with italic text" in text
    assert [p for p, _ in pp.parse_articles(io.BytesIO(XML.encode()), english_only=False)] == ["1", "4"]


def test_zip_to_parquet_end_to_end_and_resumable(tmp_path):
    z = tmp_path / "med.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("data/pubmed_baseline/pubmed26n0001.xml.gz", gzip.compress(XML.encode()))
        zf.writestr("data/pubmed_baseline/pubmed26n0002.xml.gz", gzip.compress(XML.encode()))
        zf.writestr("data/readme.txt", "ignored")
    out = tmp_path / "out"
    assert pp.main([str(z), "--out", str(out), "--workers", "1"]) == 0
    t = pq.read_table(out / "pubmed26n0001.parquet").to_pylist()
    assert len(t) == 1 and t[0]["doc_id"] == "pmid:1" and t[0]["domain"] == "medical"
    assert t[0]["char_count"] == len(t[0]["text"])
    assert sorted(p.name for p in out.glob("*.parquet")) == ["pubmed26n0001.parquet", "pubmed26n0002.parquet"]
    first = (out / "pubmed26n0001.parquet").stat().st_mtime_ns
    r = pp.process_member((str(z), "data/pubmed_baseline/pubmed26n0001.xml.gz", str(out), 100, True, 0))
    assert r["skipped"] and (out / "pubmed26n0001.parquet").stat().st_mtime_ns == first


def test_loose_files_and_tokenize_parquet_can_read_the_output(tmp_path):
    d = tmp_path / "x"
    d.mkdir()
    (d / "pubmed26n0009.xml.gz").write_bytes(gzip.compress(XML.encode()))
    out = tmp_path / "o"
    assert pp.main([str(d), "--out", str(out), "--workers", "1"]) == 0
    import tokenize_parquet as tp
    st = tp.process_file((str(out / "pubmed26n0009.parquet"), str(tmp_path / "c"), None, 0.0, False, 0, True, True))
    assert st["docs"] == 1 and st["per_domain"]["medical"]["docs"] == 1
