# Swaraj's Domain 9 / 10 handover (v7): check of 4 Nov 2026

**Evidence labels.** *Run* = I downloaded both zips (with Pankaj's permission) and checked them myself. *Read* = taken from Swaraj's README or audit JSON.

## What was delivered (run)
- `PRANA_G_D9_D10_REVISED_FINAL_HANDOVER_v7.zip` (206.8 MB) contains `PRANA_G_D9_D10_REVISED_CORPUS_v7.zip` (85.6 MB), `PRANA_G_Solver_Docs_Supplement_v1.zip` (121.2 MB), a one-line README and `SHA256SUMS.txt`. Both inner zips match their SHA-256 in `SHA256SUMS.txt` (corpus `bea30f18...2ef736`, supplement `c6668e53...699ce`). The standalone corpus zip Swaraj sent separately is byte-identical to the inner one.
- Swaraj is leaving the internship (family medical situation, 4 Nov) and says he has spoken to Raju. Token counts were not calculated by him ("Pankaj to recount").

## Results (run)
| Check | Domain 9 | Domain 10 |
|---|---|---|
| Documents in manifest and text file | 4,368 | 930 |
| Hashes: `text_sha256` of every document recomputed (the hash is of the document with surrounding whitespace stripped, separator `\n<|endoftext|>\n`) | 4,368 / 4,368 match | 930 / 930 match |
| `<EMAIL_ADDRESS>` or raw email left | 0 | 0 |
| Lines starting `# ` left | 0 | 0 |
| Exact duplicate documents | 0 | 0 |
| Tokens with the frozen `vocab32k_v2` (hash `10b3c5ae...b51cdc`) via `scripts/tokenize_text.py` | **39,897,904** | **13,063,500** |
| Documents with no domain term | 8 (my rule; README says 8 retained) | n/a |
| Documents with one domain term | 106 (README: 106) | n/a |
| Documents with a PINN / neural operator / FNO / DeepONet term | n/a | 917 of 930 |
| Licences | CC BY 4,362; CC0 6 | CC BY 930 |
| Dropped off-topic IDs (35) still present | 0 | n/a |

- Total 52.96M tokens (1 Nov handover: 40.15M and 13.07M, so the cleanup cost only about 0.25M tokens).
- All 17 earlier zero-match documents and the 132 one-match ones were reviewed: 9 + 26 clearly off-topic documents removed, the rest kept (decisions in `domain9_weak_document_decisions.csv`). The 8 retained zero-match documents are numerical or modelling papers (fluid-structure interaction, ADER-WENO, hydrothermal flow, shale-gas simulation) with a few borderline ones (neurotransmitter diffusion model, MRI pressure field).
- `check_bins` PASS on the domain 9 bin. The domain 10 validation bin has 0 documents because the hash-based split puts none of 930 documents into validation. A validation share has to be set by hand before training.
- Held licences (not in the counts above): CC BY-SA 88 + 22 documents, public domain 44 + 9 documents. Raju's licence decision is still open.
- Solver-docs supplement: OpenFOAM ENCCS (CC BY 4.0, 12 documents) and FEniCSx tutorial (CC BY 4.0, 40 pages), about 0.5 MB of text, i.e. about 0.13M tokens. DeepXDE (LGPL-2.1) is held and not merged. It is far too small to change the totals.

## What this means
- Cleanup requested on 1 Nov is complete and verified.
- Volume is the remaining gap. Raju's allocation is 2.1B tokens each for domains 9 and 10; we hold 39.9M and 13.1M. At 4 passes that covers 0.16B and 0.05B. Raju's pilot (3B tokens) would repeat domain 9 about 5x and domain 10 about 16x.
- Open: validation split for domain 10; Ravi's claim to hold more data for domains 9 and 10 is unproven; who takes over from Swaraj (Raju).

## Follow-up the same night (Swaraj's reply, run by me)
Swaraj says he has no verified v7 token counts and asks the successor to recount: done above (39,897,904 and 13,063,500). Remaining gap against 2.1B each is about 2.06B and 2.09B; at the 4-pass limit the unique tokens needed are 0.525B each, i.e. about 0.485B and 0.512B more.

His cleaning rules also claim BOM, zero-width and control characters were handled. Measured in v7:
| Item | Domain 9 docs | Domain 10 docs |
|---|---|---|
| U+FFFD replacement character (encoding damage) | 161 (3.7%) | 20 (2.2%) |
| Zero-width characters | 5 | 6 |
| BOM | 0 | 1 |
| Control characters | 0 | 0 |
| Non-breaking space | 9 | 0 |
| Line `#` followed by whitespace other than a space | 4 | 1 |
Every document except the first starts with one blank line in the text file (the manifest hash is of the stripped text, so hashes still match). These are small, but "handled" is not exactly true and the successor should fix them.

His "raw source" Drive folder `PRANA_G_D9_D10_FULL_v2` (not downloaded) lists: `domain9/10_full_corpus.clean.txt` (earlier version), `domain9_full_corpus.clean.txt.pre_repair.bak`, manifests, `final_quality_audit.json`, `full_collection_state.json`, `full_collection_summary.json`, held manifests, the solver supplement zip, `README_FOR_PANKAJ.txt`. It holds earlier processed outputs, not raw shards; the upstream sources named in the manifest are `common-pile/peS2o_filtered` (3,673 D9 docs) and `common-pile/arxiv_papers_filtered` (695). The selection code is not in the folder.
