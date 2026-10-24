# Shared progress report — scale-up to 1.5B

Single source of truth for status, blockers, owners and deadlines (Raju, 21 Oct: "track failures, blockers, owners and
deadlines in one shared progress report"). Updated by Pankaj. **"Evidence" is what was actually seen** (a message, a file,
a test); anything without evidence is marked *awaiting*. Last update: 24 Oct 2026.

## Gates (in order; nothing starts before the gate above it)
| # | Gate | Owner | Status | Evidence / next step |
|---|---|---|---|---|
| 0 | 32K tokenizer approved and frozen | Raju (approve), Pankaj (freeze) | **awaiting approval** | two candidates built and verified (round-trip, C++ parity 3003/3003): `tokenizers/vocab32k_v1/` and `tokenizers/vocab32k_digits_v1/` (numbers digit by digit, ~5–6% more tokens; recommended). Raju/Pankaj choose one, then it is frozen |
| 1 | Compute budget decided | Raju | **awaiting** | Gokul: ₹330/H100-hour, credit ≈ ₹1,000 ≈ 3 hours; 1.51B on 30B tokens ≈ 257 H100-hours ≈ ₹85,000 (est.) |
| 2 | All cleaned data in one shared Drive folder | Raju (create/confirm folder), all owners (upload) | **awaiting** | requested in the group; no folder link seen yet |
| 3 | Recount per domain from the manifests | Pankaj | blocked by gates 0 and 2 | `tokenize_*.py --count-only` is ready |
| 4 | Tokenize, merge per domain, `check_bins.py` PASS | Ravi (data), Gokul (machine) | blocked by gates 0 and 2 | tools ready and tested: `tokenize_parquet.py`, `tokenize_text.py`, `merge_bins.py`, `check_bins.py` |
| 5 | Smoke test on the GPU (≈ 40 min) | Gokul, Om | **awaiting GPU time** | smoke data and the exact command are shared; Om's first benchmark promised "as soon as I train it on the GPU" |
| 6 | First run (100M, ≈ 1.3 H100-hours) → evaluate per domain | Gokul, Om, Pankaj | blocked by gates 4 and 5 | recipe in `docs/TRAINING_PLAN_1p5B.md` §4 |
| 7 | 1.51B run | Raju (budget) | blocked by gate 1 | |

## Owners and open items
| Owner | Item | Status | Since |
|---|---|---|---|
| Gokul | exact dedup of Biology / Earth / Scientific Discovery | **done** (Colab, 22 Oct): 150,229 duplicates removed; 1,642,491 documents kept | 22 Oct |
| Gokul | share the three cleaned folders (Drive link) | awaiting | 22 Oct |
| Gokul | Medical extraction (`scripts/pubmed_to_parquet.py`), CPU only | awaiting Raju's decision to include Medical | 21 Oct |
| Prabanjan | math dedup | **done**: 4,924,517 documents kept (331 duplicates, 8,444 broken-encoding, 1 short removed) | 21 Oct |
| Prabanjan | upload `maths_final_corpus.clean.txt` (as `.gz`) to Drive | awaiting | 21 Oct |
| Prabanjan | Multi-step / Astronomy / Finance corpora | **delivered** (373 MB zip), checked by Pankaj | 21 Oct |
| Prabanjan | Physics file | delivered earlier; not read in full | — |
| Ravi | final 32K vocabulary | **closed**: cannot exist before data; built by Pankaj instead | 21 Oct |
| Ravi | training-text rule for the v4.0 files | **done**: his rule adopted in `v4_text` (verified against the files) | 23 Oct |
| Ravi | Scientific Programming dedup by sha256 (491,834 distinct names in 1,831,796 rows) | **awaiting** — Ravi said Avinash's final merge will do it | 21 Oct |
| Avinash | merged file with dedup (promised "by tomorrow", 21–22 Oct) | **awaiting**; he asked Pankaj to run index/quality/embeddings/cache — deferred, not on the training path | 22 Oct |
| Avinash | cross-dataset near-duplicate removal; domain mapping to the 15 capabilities | awaiting | 20 Oct |
| Om | 1.51B benchmark | awaiting GPU time; FSDP/FP8/FA3 not in the repo | 21 Oct |
| Swaraj | gap analysis of the LLM against the requirements, before training | awaiting (Raju: focus on the 3B model) | 21 Oct |
| Afnan, Avinash | six engineering capabilities: owners and dates | **no reply seen**; no data exists | 20 Oct |
| Everyone | general-language data (none exists, nobody owns it) | open | 20 Oct |
| Pankaj | tools, plan, tokenizer, audit | **done**: see `CHANGELOG.md`; 24 Oct audit below | 24 Oct |

## Failures and corrections (kept on purpose)
| Date | What went wrong | Fix |
|---|---|---|
| 20 Oct | Plan said 30B tokens = 8 GPU-days | wrong; ≈ 11 GPU-days (257 H100-hours); corrected in the plan |
| 21 Oct | First `v4_text` rule dropped most CAD text, Q&A options/explanations, `doc_content` | rule rebuilt from a field census of the real files; Ravi's audit confirmed the Q&A and documentation gaps (23 Oct) |
| 21 Oct | "CAD is almost all synthetic" (from one row group of a source-sorted file) | wrong; all 940,576 rows: 40.5% procedural, 59.5% zerotocad |
| 21 Oct | Tokenizer candidate v1 had never seen CAD/STEP syntax | retrained as v2 before anything was tokenized |
| 24 Oct audit | No tokenizer for `.txt/.gz` corpora (math = the largest dataset); big Parquet files used one core; same-named files overwrote each other; Windows `\r\n` and lone surrogates would have corrupted or crashed runs; no domain mixing; non-atomic checkpoint writes; numbers tokenized inconsistently | fixed and tested (`tokenize_text.py`, row-group jobs, unique names, CRLF/surrogate repair, `--train-mix`, atomic saves, `--split-digits`) |

## Risks
GPU time (≈ 3 hours available) is the binding constraint; the multi-GPU path and the 1.51B model have never run on CUDA;
all token counts are estimates until the manifests exist; the tokenizer sample has no physics, engineering or general-language
text; data licences are the owners' claims, unchecked.
