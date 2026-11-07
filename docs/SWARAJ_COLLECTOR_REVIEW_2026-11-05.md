# Swaraj's Domain 9 / 10 collector script: static review, 5 Nov 2026

**Evidence labels.** *Read* = I read all 668 lines of the script (SHA-256 `f763422a...eda02c`) and parsed it with Python's `ast` (no execution). *Run* = computed by me from the v7 manifests I downloaded earlier (`docs/SWARAJ_V7_HANDOVER_CHECK_2026-11-04.md`). I did **not** run the script or its `--self-test`: it is code from outside the team's repository and it downloads from the network. It was written for Google Colab and is meant to be run by the person who takes over Domains 9 and 10.

## What it does (read)
- Source: all shards of `common-pile/peS2o_filtered` (full-text records only: `source` must contain `s2orc`) and `common-pile/arxiv_papers_filtered`, found with `HfApi` at the moment of the run and recorded in `full_collection_state.json`.
- Screens **title and abstract only**. Domain 9 needs at least 2 distinct canonical concepts (CFD, finite element, Navier-Stokes, PDE, finite volume, computational mechanics, fluid-structure interaction, finite difference, Galerkin, spectral element, meshless); the abbreviations CFD / FEM / FEA count only with context words. Domain 10 needs PINN / DeepONet / Fourier neural operator / neural operator / operator learning / physics-informed ML / SciML plus a scientific context word. A paper that matches both goes to Domain 10 only.
- Licence from the record's `license` / `oa_license`: CC BY and CC0 go into the main corpus; CC BY-SA and public domain are written to separate "held" files; non-commercial, no-derivatives, missing, unrecognised or conflicting licences are dropped.
- Keeps the whole document, cleans it (HTML unescape, NFC, whitespace), requires at least 200 words and a letters-and-spaces ratio of 0.55, removes exact and whitespace/case-normalised duplicates across shards, and writes documents separated by `\n\n<|endoftext|>\n\n`. A shard is committed only after it is fully merged; a run can resume.

## Safety scan (read)
No `exec`, `eval`, `subprocess`, `os.system`, `pickle`, `socket`, `requests` or `urllib`. The only network calls are `HfApi.dataset_info`, `list_repo_files` and `hf_hub_download` from `huggingface_hub`. Writes go to `--out-dir` and to two hard-coded Colab folders `/content/prana_full_corpus_staging` and `/content/prana_full_corpus_workers` (on a Windows machine that would create `C:\content`).

## Does it reproduce v7? Findings (read / run)
1. **Which documents.** The v7 manifests record one pinned revision per source: peS2o `297747513bfb0ff1fbf61ddad3b03319d0f04597`, arXiv `033cf7f53f9b348deec868c1a5a48484f3ee9e52`, and shards peS2o 0000 to 0090 (91 shards) and arXiv 0000 to 0007 (8 shards). The script itself takes the **current** revision at run time, so re-running it unchanged may pick up different documents. To reproduce v7, pin these two revisions.
2. **Order and duplicates.** Shards are merged in the order they finish, so document order and which copy of a duplicate survives can differ between runs. The set of documents should be nearly the same, not byte-identical.
3. **v7 has cleanups this script does not contain.** The script strips e-mail placeholders only for Domain 10 (`strip_domain10_author_frontmatter`); Domain 9 gets `clean_text` only. The v7 files have 0 `<EMAIL_ADDRESS>` and 0 lines starting with `# ` in both domains (checked 4 Nov), so those steps were done in a later, undocumented pass. A rerun of this script alone would bring back about 700 Domain 9 documents with placeholders (1 Nov handover: 702 documents, 1,425 occurrences).
4. **"BOM, zero-width and control characters handled" is only partly true.** `clean_text` removes characters below code point 32 (except newline and tab) but keeps zero-width characters and the BOM; v7 has zero-width characters in 5 (D9) and 6 (D10) documents and a BOM in 1 (D10). The file is read with `errors="replace"`, which turns invalid bytes into U+FFFD; v7 has U+FFFD in 161 Domain 9 (3.7%) and 20 Domain 10 documents (2.2%). I cannot say how many come from the source and how many from this setting.
5. **Licence order.** The test for "public domain" comes before the test for CC0, so a record labelled "CC0 1.0 Public Domain Dedication" would be held instead of accepted. It can only lose documents, not admit a wrong licence.
6. **Self-test.** It states 18 assertions; there are 19. Not run (see above).

## What the v7 numbers say about the volume gap (run)
| | Domain 9 | Domain 10 |
|---|---|---|
| Documents / megabytes / tokens held | 4,368 / 162.3 MB / 39.90M | 930 / 50.1 MB / 13.06M |
| From peS2o / arXiv | 3,673 / 695 | 337 / 593 |
| Documents with exactly 2 concepts (the minimum) | 3,814 (87%); 3 concepts 483; 4 or more 71 | n/a |
| Held CC BY-SA + public domain (not in the counts) | 88 + 44 documents, 8.05 MB, about 2.0M tokens | 22 + 9 documents, 2.07 MB, about 0.5M tokens |
| Raju's ask for the 1B run | 60 to 80M tokens, so +20 to +40M | 60 to 80M tokens, so +47 to +67M |

- **The two named sources are already exhausted by this rule**: the script scanned every shard of both repositories (99 shards), so running it again cannot add volume. Releasing the held licences adds only about 2.0M and 0.5M tokens (5 to 10% of the Domain 9 gap, about 1% of the Domain 10 gap).
- Where more volume could come from (options for the new owner, not checked by me): (a) relax the Domain 9 rule from 2 concepts to 1 plus a context word (the 87% at the minimum shows the cut is steep; the effect cannot be measured without the raw shards, which are not on this machine); (b) abstract-only matching misses papers whose methods are in the body; scoring a window of the full text would widen it; (c) other openly licensed sources (for example the PubMed Central open-access CC BY subset, OSTI, NASA technical reports, solver documentation such as OpenFOAM, FEniCS, deal.II, MOOSE, SU2) each need their own licence check; (d) the licence decision on CC BY-SA and public domain.
- At about 9,100 tokens per Domain 9 document and 14,000 per Domain 10 document, +20 to +40M tokens is roughly 2,200 to 4,400 more Domain 9 documents and +47 to +67M is 3,400 to 4,800 more Domain 10 documents, against 5,298 collected from 99 shards.

## Recommendation
Treat the script as documentation of the selection rule, not as the way to grow the corpus. The new owner should: pin the two revisions above; copy the three missing cleanups (placeholders and `# ` lines for Domain 9; U+FFFD, zero-width and BOM for both) into the script before any rerun; and agree with Raju which of (a) to (d) to use before collecting. First batch of any new source: a 20 MB sample for review (the team rule).


## Domain 10 volume: what exists (8 Nov 2026, Raju: "Pankaj will cover it")
**Run (OpenAlex metadata counts, 8 Nov; works with a CC BY licence and a PDF; title or abstract match; overlap with the 930 documents we already hold not removed):** "physics-informed neural network" 2,237; neural operator / operator learning / DeepONet / FNO 1,000; scientific ML / physics-informed ML / physics-guided / neural PDE 1,500; one broad query (the terms above plus differentiable physics, neural surrogate, surrogate model with partial differential) 6,227. CC BY-SA adds about 2%.
**Estimate (mine):** at our mean of 14,046 tokens per Domain 10 document, 6,200 papers are about 87M tokens at most, about 40 to 70M of it new after removing what Swaraj already collected. The pilot gap (+57M) is borderline reachable; **the full-run gap (+512M, 36,000 more papers) does not exist as open CC BY text**. Domain 10's 2.1B allocation would be 25 to 35 passes. Raju has to cut it or accept the repeats, and give the freed tokens to domains with real volume.
