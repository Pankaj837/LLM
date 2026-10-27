# Shared progress report — scale-up to 1.5B

Single source of truth for status, blockers, owners and deadlines (Raju, 21 Oct: "track failures, blockers, owners and
deadlines in one shared progress report"). Updated by Pankaj. **"Evidence" is what was actually seen** (a message, a file,
a test); anything without evidence is marked *awaiting*. Last update: 27 Oct 2026.

## Summary (27 Oct 2026)
**Ready:** the 32K tokenizer (frozen, `vocab32k_v2`); the whole pipeline (tokenize, merge, integrity check, weighted mix, per-domain evaluation); the training code; a one-command GPU session (`scripts/gpu_session.py`: preflight, benchmark, 1.5B smoke test, timed 100M run, evaluation, report), rehearsed end to end on CPU; code tag `scale-run-rc6`, 334 tests pass.

**Data, token files:** verified: Prabanjan, 612M tokens (math, Multi-step, Astronomy, Finance). Reported, not yet verified: Ravi, 767M tokens (code 322.8M, Scientific Q&A 188.6M, chemistry 127.6M, CAD 78.4M, Scientific Programming 49.8M; his Drive folder is private). Rejected: Gokul's Biology and Earth files (built with the wrong 32K vocabulary; being rebuilt). Not yet built: Physics, Scientific Discovery.

**Not available:** a GPU session has not happened. The plan stays on the 3 confirmed H100-hours; AIKosh (22 hours, asked for by Raju on 26–27 Oct, GPU type and start date unknown) is not part of the plan (Pankaj, 27 Oct). No data exists for capabilities 4, 9, 10, 11, 12, 14, 15 or general language.

**Needs a decision (Raju):** owners for the capabilities without data (4, 9, 10, 11, 12, 14, 15, general language); whether to include Medical; the source of the 3 GPU hours (Gokul).

**Ownership (Raju, 27 Oct):** Swaraj takes ownership of the LLM project in coordination with Pankaj.

Who was allotted what, when, and whether it was delivered (from 10 Oct): see `docs/ASSIGNMENT_LEDGER.md`.

## Gates (in order; nothing starts before the gate above it)
| # | Gate | Owner | Status | Evidence / next step |
|---|---|---|---|---|
| 0 | 32K tokenizer approved and frozen | Raju (approve), Pankaj (freeze) | **DONE 25 Oct**: `vocab32k_v2` frozen (Raju: digits version only later if numerical reasoning needs it); `tokenizers/FROZEN.md`, guarded by a test | two candidates were built and verified (round-trip, C++ parity 3003/3003); the digits one (`tokenizers/vocab32k_digits_v2/digits_NOT_FROZEN.tok`) is NOT used. 25 Oct: Gokul's bio/earth token files turned out to be built with the digits vocabulary (same size and EOS, so the header check passed) — rejected, to be rebuilt; `check_bins.py` now catches this |
| 1 | Compute budget decided | Raju | **decided 24 Oct**: smaller model first, 3 GPU-hours. 26–27 Oct: Raju asked everyone to register on AIKosh for 22 hours (Ravi, Gokul registered; Prabanjan's access, approved for Saturday, handed to Raju / Pankaj). GPU type and start date unknown, so the plan does not depend on it (Pankaj, 27 Oct) | 1.51B on 30B tokens (≈ 257 H100-hours) is out of reach; the 3-hour session is `docs/TRAINING_PLAN_1p5B.md` §5b |
| 2 | All cleaned data in one shared Drive folder | Raju (create/confirm folder), all owners (upload) | **awaiting** | owners post their own folders; Ravi's two folders are private (sign-in required), Om asked him for access (27 Oct) |
| 3 | Recount per domain from the manifests | Pankaj | blocked by gates 0 and 2 | `tokenize_*.py --count-only` is ready |
| 4 | Tokenize, merge per domain, `check_bins.py` PASS | each owner for own domains | **in progress**: Prabanjan verified (612M); Ravi reports 5 domains PASS, 767M tokens (not verified yet); Gokul bio/earth rejected, rebuild; Physics and Scientific Discovery not built. Each owner runs `build_dataset.py` on own domains (it now refuses a non-frozen vocabulary) | tools ready and tested: `tokenize_parquet.py`, `tokenize_text.py`, `merge_bins.py`, `check_bins.py` |
| 5 | Smoke test on the GPU (≈ 40 min) | Pankaj (`gpu_session.py`), Gokul, Om | **ready, awaiting GPU**: the whole session (benchmark, smoke, timed run, evaluation) is one tested command | smoke data and the exact command are shared; Om's first benchmark promised "as soon as I train it on the GPU" |
| 6 | First run (100M, ≈ 1.3 H100-hours) → evaluate per domain | Gokul, Om, Pankaj | blocked by gates 4 and 5 | recipe in `docs/TRAINING_PLAN_1p5B.md` §4 |
| 7 | 1.51B run | Raju (budget) | blocked by gate 1 | |

## Owners and open items
| Owner | Item | Status | Since |
|---|---|---|---|
| Gokul | exact dedup of Biology / Earth / Scientific Discovery | **done** (Colab, 22 Oct): 150,229 duplicates removed; 1,642,491 documents kept | 22 Oct |
| Gokul | share the three cleaned folders (Drive link) | awaiting (the shared `processed` folder still holds the 17–18 Oct files) | 22 Oct |
| Gokul | smoke test | **ran on his laptop RTX 4050 (6 GB), `small` preset, seq 512, batch 1: loss 9.71 → 7.57, no NaN** — first proof that the CUDA/bf16 path trains; the H100 benchmark is still to do | 24 Oct |
| Gokul | token files for Biology (53.4M train tokens, 4 files) and Earth & Environment (77.7M, 6 files), `bio.*.bin` / `earth.*.bin` | **REJECTED (25 Oct)**: header, size and EOS are fine (`check_bins` PASS at the time) and the counts equal his log, but they were tokenized with `vocab32k_digits_v2`, not the frozen `vocab32k_v2` (same size, same EOS id, same file name): decoded with the frozen vocabulary 300 of 300 documents are gibberish, with the digits one they read as the right text. Rebuild with the frozen file. Scientific Discovery run was still going after 30 min: the full set is 2.8 GB (~0.8B tokens, ~50 min at his earth speed) against ~170M planned, so cap it (`"sample_rg": 0.2`) | 25 Oct |
| Gokul | Medical extraction (`scripts/pubmed_to_parquet.py`), CPU only | awaiting Raju's decision to include Medical | 21 Oct |
| Prabanjan | math dedup | **done**: 4,924,517 documents kept (331 duplicates, 8,444 broken-encoding, 1 short removed) | 21 Oct |
| Prabanjan | upload the cleaned math file to Drive | **done and verified (25 Oct)**: folder `Prabanjan_LLM` holds `maths_final_corpus.clean.txt` (34.05 GB, plain text, one document per line), the Multi-step / Astronomy / Finance files (byte-identical in size to the zip already checked) and `physics_1B_top_notch.txt` (2.33 GB) | 21 Oct |
| Prabanjan | Multi-step / Astronomy / Finance corpora | **delivered** (373 MB zip), checked by Pankaj | 21 Oct |
| Prabanjan | token files for math, Multi-step, Astronomy, Finance (`bins.zip`, 607 MB) | **built with the real pipeline and verified (25 Oct)**: math 486.5M, Multi-step 88.6M, Astronomy 20.3M, Finance 17.2M tokens; integrity check PASS; decoded with the frozen vocabulary they read as the right text, with the other candidate as gibberish (so the vocabulary is the frozen one). Math was read as a prefix (first 1.83 GB): rerun with `spread` for a representative sample. Physics not yet tokenized | 25 Oct |
| Prabanjan | Physics file | **read (25 Oct)**: mixed sources (65% Cosmopedia textbook units, 8% raw arXiv LaTeX, OpenWebMath pages), 13% of documents with little physics, document markers `<|document|>`; usable, ~0.64B tokens; supported by `tokenize_text.py --marker-domains` | — |
| Ravi | token files for 5 domains (27 Oct message: built and verified on tag `scale-run-rc3` with the frozen tokenizer; integrity PASS, 10 files, ~770M tokens: code 322,833,044; sciqa 188,559,089; sciprog 49,848,074; chem 127,623,382; cad 78,449,381) | **reported, not verified**: the numbers add up (767M), but both Drive folders he posted ask for sign-in; built on the older tag, which cannot detect a wrong vocabulary. Asked him to open the folder and re-run `check_bins.py` on `scale-run-rc6`. Scientific Programming repeats ~3.9x at the planned mix | 27 Oct |
| Ravi | final 32K vocabulary | **closed**: cannot exist before data; built by Pankaj instead | 21 Oct |
| Ravi | training-text rule for the v4.0 files | **done**: his rule adopted in `v4_text` (verified against the files) | 23 Oct |
| Ravi | Scientific Programming dedup by sha256 (491,834 distinct names in 1,831,796 rows) | **"deduped all 4 parquet, uploaded as <domain>_cleaned.parquet" (24 Oct)**; the link he posted is the Pranag AI root folder, which shows no such files, and his earlier team folder now asks for sign-in: **cannot verify** — asked for the folder link and row counts | 21 Oct |
| Avinash | merged file with dedup (promised "by tomorrow", 21–22 Oct) | **reported, file not uploaded (27 Oct)**: his message gives `data/processed/universal_merged.parquet`, 64,364,210 rows "subject to the latest verification output", SHA-256 "to be confirmed", and a Drive folder with five small reports (dedup report, domain counts, manifest, a duplicate sample, and a text report he named `pankaj_verification_report.txt`, which Pankaj did not produce). The merged file itself is not in the shared `processed` folder (only `reports`). **Reports read (27 Oct)**: `merged_dataset_manifest.json` gives 64,364,210 rows, 44.3 GB, SHA-256 `00632085…06100F7` (computed 24 Oct on his D: drive; the parquet itself not seen). `dedup_report.json`: method "exact duplicate-key count; **no rows deleted**": 50,910,626 unique `entity_id`, 8,910,000 duplicated ids, **13,453,584 removable rows** — the file is merged, not yet deduplicated. His own text report records `FAIL - Dedup report records no row-count reduction`, `set.md documents 13 mappings; the official list of 15 capabilities was not established`, and is named `pankaj_verification_report.txt` (not Pankaj's). `domain_row_counts.csv`: 33 domain labels that are variants of the same domain (`Human_Social`, `Human & Social`, `human_social`, `Human & Social Sciences`, `Human & Social Systems`; `Chemical` / `chemical`; `Economic & Industrial` / `Economic_Industrial`) and many domains with exactly 2x rows per id (e.g. computer_science_ai 403,492 rows / 201,746 ids, mathematics 360,000 / 180,000, quant_finance 3,592,860 / 1,796,430). Not part of the LLM training mix (domain owners' files are); needed for the PINN / knowledge-graph side. He asks for help with intrinsic properties, 768-dim embeddings and the surrogate cache (laptop RAM); Raju told him to describe the work to Namit | 22 Oct |
| Avinash | domain mapping to the 15 capabilities | asked 24 Oct what the list is: answered with `docs/CAPABILITIES.md`; mapping awaiting | 24 Oct |
| Avinash | steps 4–6 (intrinsic properties, 768-dim embeddings, surrogate cache) | **postponed**: not needed for LLM training (Raju, 24 Oct: required later for the PINN side) | 24 Oct |
| Avinash | cross-dataset near-duplicate removal | awaiting | 20 Oct |
| Om | 1.51B benchmark | awaiting GPU time; FSDP/FP8/FA3 not in the repo | 21 Oct |
| Swaraj | gap analysis of the LLM against the requirements | **3 of the asked 5 posted (24 Oct)**; they point to a fine-tuning stage that is not yet planned (plan §6b). Asked for 2 more and for the `scientific_ir` examples | 21 Oct |
| Swaraj / Avinash | 768-dim embeddings and surrogate cache for the FAISS retrieval pipeline | Swaraj: needed as soon as integration testing of the reasoning agent starts. **Proposal: start with a ~1M-row sample** (about 1.5 GB of vectors, minutes on any GPU) instead of all 64M rows (~100 GB, hours on an H100); needs the embedding model's name | 24 Oct |
| Afnan → ? | capabilities 4, 9, 10 and Engineering | **Afnan left the internship (24 Oct)**; Raju: give his part to someone else. Proposal to Raju: Swaraj takes 9 and 10 (PINN side), Prabanjan takes 4. No data exists | 24 Oct |
| Avinash | six engineering-side capabilities (12, 14, 15) | no data delivered | 20 Oct |
| Everyone | general-language data (none exists, nobody owns it) | open | 20 Oct |
| Raju | "some domains are missing like physics and others" (25 Oct) | physics **is delivered** (2.33 GB, ~0.64B tokens) and only waits to be tokenized; the domains with no data at all are capabilities 4, 9, 10, 11, 12, 14, 15 (`docs/CAPABILITIES.md`) and general language | 25 Oct |
| Pankaj | tools, plan, tokenizer, audit | **done**: see `CHANGELOG.md`; 24 Oct audit below | 24 Oct |

## Failures and corrections (kept on purpose)
| Date | What went wrong | Fix |
|---|---|---|
| 20 Oct | Plan said 30B tokens = 8 GPU-days | wrong; ≈ 11 GPU-days (257 H100-hours); corrected in the plan |
| 21 Oct | First `v4_text` rule dropped most CAD text, Q&A options/explanations, `doc_content` | rule rebuilt from a field census of the real files; Ravi's audit confirmed the Q&A and documentation gaps (23 Oct) |
| 21 Oct | "CAD is almost all synthetic" (from one row group of a source-sorted file) | wrong; all 940,576 rows: 40.5% procedural, 59.5% zerotocad |
| 21 Oct | Tokenizer candidate v1 had never seen CAD/STEP syntax | retrained as v2 before anything was tokenized |
| 24 Oct audit | No tokenizer for `.txt/.gz` corpora (math = the largest dataset); big Parquet files used one core; same-named files overwrote each other; Windows `\r\n` and lone surrogates would have corrupted or crashed runs; no domain mixing; non-atomic checkpoint writes; numbers tokenized inconsistently | fixed and tested (`tokenize_text.py`, row-group jobs, unique names, CRLF/surrogate repair, `--train-mix`, atomic saves, `--split-digits`) |
| 25 Oct | `check_bins.py` PASSED Gokul's bio / earth token files that were built with the wrong 32K vocabulary (the header holds only size and EOS id, identical in both candidates; both folders held a file named `vocab32k.tok`; docs still called the digits one "recommended") | `check_bins --vocab` now decodes and re-encodes 20 chunks per file (ids must come back identical: 0 false alarms on Prabanjan's 8 verified files, all 4 of Gokul's rejected); `build_dataset.py` refuses any vocabulary but the frozen one; the digits file is renamed `digits_NOT_FROZEN.tok`; the "recommended" wording is removed; `gpu_session.py preflight` also checks the manifests' hash |
| 25 Oct | `llm.train --val-bin` takes one file, the builder wrote only per-domain validation files (a plain concatenation would validate on the first domain only) | `merge_bins.py --cap-per-input` builds a balanced `val_mix.bin`; `gpu_session.py` does it automatically |
| 24 Oct audit, pass 1 (code) | A rerun with a different vocabulary into the same folder would have silently reused the old shards; a crashed job left a valid-looking bin; the dedup tool overwrote same-named inputs and wrote tiny row groups | fingerprint stored with every finished job; outputs written to temporary names and renamed on success; unique output names; 64 MB row groups; golden regression test proves training results are unchanged |
| 24 Oct audit, pass 2 (claims, commands) | CAD tokens reported as 2.0B but ~0.75B under the agreed text rule; the plan's total was optimistic; docs did not say what was read, sampled, reported or estimated | numbers re-derived from the files; evidence legend added; every documented command run on real data (count, dedup, tokenize, merge, check, mixed training, per-domain evaluation) |
| 24 Oct audit, pass 3 (process) | The vocabulary had one special token only; chat / tool-use / structured-output stages (Raju's requirements) would have needed an embedding resize of a frozen vocabulary; memory notes were stale | 64 special tokens reserved before the freeze; candidates rebuilt as `vocab32k_v2` / `vocab32k_digits_v2`; a fresh clone of the pushed repo passes the whole test suite |

## Risks
GPU time is the binding constraint (3 hours confirmed; 22 hours through AIKosh requested by Raju, not yet granted, GPU type unknown); the multi-GPU path and the 1.51B model have never run on CUDA;
all token counts are estimates until the manifests exist; the tokenizer sample has no physics, engineering or general-language
text; data licences are the owners' claims, unchecked.
