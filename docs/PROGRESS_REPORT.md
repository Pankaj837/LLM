# Shared progress report — scale-up to 1.5B

Single source of truth for status, blockers, owners and deadlines (Raju, 21 Oct: "track failures, blockers, owners and
deadlines in one shared progress report"). Updated by Pankaj. **"Evidence" is what was actually seen** (a message, a file,
a test); anything without evidence is marked *awaiting*. Last update: 24 Oct 2026.

## Gates (in order; nothing starts before the gate above it)
| # | Gate | Owner | Status | Evidence / next step |
|---|---|---|---|---|
| 0 | 32K tokenizer approved and frozen | Raju (approve), Pankaj (freeze) | **DONE 25 Oct**: `vocab32k_v2` frozen (Raju: digits version only later if numerical reasoning needs it); `tokenizers/FROZEN.md`, guarded by a test | two candidates built and verified (round-trip, C++ parity 3003/3003): `tokenizers/vocab32k_v2/` and `tokenizers/vocab32k_digits_v2/` (numbers digit by digit, ~5–6% more tokens; recommended). Raju/Pankaj choose one, then it is frozen |
| 1 | Compute budget decided | Raju | **decided 24 Oct**: smaller model first, 3 GPU-hours total, no more available (Gokul: team heads will not allow) | 1.51B on 30B tokens (≈ 257 H100-hours) is out of reach; the 3-hour session is `docs/TRAINING_PLAN_1p5B.md` §5b |
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
| Gokul | share the three cleaned folders (Drive link) | awaiting (the shared `processed` folder still holds the 17–18 Oct files) | 22 Oct |
| Gokul | smoke test | **ran on his laptop RTX 4050 (6 GB), `small` preset, seq 512, batch 1: loss 9.71 → 7.57, no NaN** — first proof that the CUDA/bf16 path trains; the H100 benchmark is still to do | 24 Oct |
| Gokul | Medical extraction (`scripts/pubmed_to_parquet.py`), CPU only | awaiting Raju's decision to include Medical | 21 Oct |
| Prabanjan | math dedup | **done**: 4,924,517 documents kept (331 duplicates, 8,444 broken-encoding, 1 short removed) | 21 Oct |
| Prabanjan | upload `maths_final_corpus.clean.txt` (as `.gz`) to Drive | **"I have uploaded" (24 Oct)**; link/file not yet checked | 21 Oct |
| Prabanjan | Multi-step / Astronomy / Finance corpora | **delivered** (373 MB zip), checked by Pankaj | 21 Oct |
| Prabanjan | Physics file | delivered earlier; not read in full | — |
| Ravi | final 32K vocabulary | **closed**: cannot exist before data; built by Pankaj instead | 21 Oct |
| Ravi | training-text rule for the v4.0 files | **done**: his rule adopted in `v4_text` (verified against the files) | 23 Oct |
| Ravi | Scientific Programming dedup by sha256 (491,834 distinct names in 1,831,796 rows) | **"deduped all 4 parquet, uploaded as <domain>_cleaned.parquet" (24 Oct)**; the link he posted is the Pranag AI root folder, which shows no such files, and his earlier team folder now asks for sign-in: **cannot verify** — asked for the folder link and row counts | 21 Oct |
| Avinash | merged file with dedup (promised "by tomorrow", 21–22 Oct) | **awaiting**; he asked Pankaj to run index/quality/embeddings/cache — deferred, not on the training path | 22 Oct |
| Avinash | domain mapping to the 15 capabilities | asked 24 Oct what the list is: answered with `docs/CAPABILITIES.md`; mapping awaiting | 24 Oct |
| Avinash | steps 4–6 (intrinsic properties, 768-dim embeddings, surrogate cache) | **postponed**: not needed for LLM training (Raju, 24 Oct: required later for the PINN side) | 24 Oct |
| Avinash | cross-dataset near-duplicate removal | awaiting | 20 Oct |
| Om | 1.51B benchmark | awaiting GPU time; FSDP/FP8/FA3 not in the repo | 21 Oct |
| Swaraj | gap analysis of the LLM against the requirements | **3 of the asked 5 posted (24 Oct)**; they point to a fine-tuning stage that is not yet planned (plan §6b). Asked for 2 more and for the `scientific_ir` examples | 21 Oct |
| Swaraj / Avinash | 768-dim embeddings and surrogate cache for the FAISS retrieval pipeline | Swaraj: needed as soon as integration testing of the reasoning agent starts. **Proposal: start with a ~1M-row sample** (about 1.5 GB of vectors, minutes on any GPU) instead of all 64M rows (~100 GB, hours on an H100); needs the embedding model's name | 24 Oct |
| Afnan → ? | capabilities 4, 9, 10 and Engineering | **Afnan left the internship (24 Oct)**; Raju: give his part to someone else. Proposal to Raju: Swaraj takes 9 and 10 (PINN side), Prabanjan takes 4. No data exists | 24 Oct |
| Avinash | six engineering-side capabilities (12, 14, 15) | no data delivered | 20 Oct |
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
| 24 Oct audit, pass 1 (code) | A rerun with a different vocabulary into the same folder would have silently reused the old shards; a crashed job left a valid-looking bin; the dedup tool overwrote same-named inputs and wrote tiny row groups | fingerprint stored with every finished job; outputs written to temporary names and renamed on success; unique output names; 64 MB row groups; golden regression test proves training results are unchanged |
| 24 Oct audit, pass 2 (claims, commands) | CAD tokens reported as 2.0B but ~0.75B under the agreed text rule; the plan's total was optimistic; docs did not say what was read, sampled, reported or estimated | numbers re-derived from the files; evidence legend added; every documented command run on real data (count, dedup, tokenize, merge, check, mixed training, per-domain evaluation) |
| 24 Oct audit, pass 3 (process) | The vocabulary had one special token only; chat / tool-use / structured-output stages (Raju's requirements) would have needed an embedding resize of a frozen vocabulary; memory notes were stale | 64 special tokens reserved before the freeze; candidates rebuilt as `vocab32k_v2` / `vocab32k_digits_v2`; a fresh clone of the pushed repo passes the whole test suite |

## Risks
GPU time (≈ 3 hours available) is the binding constraint; the multi-GPU path and the 1.51B model have never run on CUDA;
all token counts are estimates until the manifests exist; the tokenizer sample has no physics, engineering or general-language
text; data licences are the owners' claims, unchecked.
