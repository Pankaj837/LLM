# Data status and next plan — 20 Oct 2026 (updated 24 Oct)

Prepared by Pankaj. Evidence levels: **read** = taken from the file itself (Parquet footer, a scan of all rows or of one column, gzip trailer);
**sampled** = measured on evenly spaced row groups; **owner** = the owner's own statement, unverified; **est.** = my calculation from the
above. Sizes and row counts are *read*; every token count except Gokul's dedup results is *owner* or *est.* until the manifests exist.

## Where the data stands

| Owner | Domain | Size | Tokens | Status |
|---|---|---|---|---|
| Ravikant | Scientific Programming | 3.82 GB, 1,831,796 rows | 4.56B owner; ~1.2–2.1B est. after repeats | v4.0 schema; 491,834 distinct file names, ~73% of rows repeat a name |
| Ravikant | CAD/STEP | 0.93 GB, 940,576 rows | 2.00B owner; **~0.75B est.** with the agreed text rule (rows average ~2,560 characters: description + CadQuery code, plus STEP snippet and Q&A on the 36% that have them, ÷ 3.3 characters/token) | 40.5% procedural (cad_synthetic), 59.5% zerotocad. His counter is not stated; it may include metadata the text rule leaves out |
| Ravikant | Scientific Q&A | 7.23 GB, 5,929,942 rows | 2.50B owner; ~2.5B est. holds (5.93M rows × ~1,400 characters ÷ 3.25 characters/token) | each Q&A stored 3× (name, description, properties), which inflates the file but not the text rule's output; 49% math Q&A |
| Ravikant | Code | 3.29 GB, 2,692,769 rows | 3.28B owner (report of 21 Oct; row count matches the file; counter not stated) | 83% of tokens are TypeScript; Python only 6 files; "100% permissive" and "zero leakage" are his claims, not checked |
| Gokul | Biology / Earth & Env. / Sci. Discovery | 5.8 / 4.2 / 2.8 GB | ~8.5B est. (1,642,491 documents after dedup) | 18 columns with `text`. Exact dedup run 22 Oct on the full folders (Colab): Biology 5 duplicates of 800,000; Earth 57,273 of 432,855 (13.2%); Scientific Discovery 92,951 of 559,865 (16.6%); 150,229 removed, matching his own earlier count of 150,226 |
| Prabanjan | Physics | 2.33 GB | ~0.6B est. | 165,528 docs |
| Prabanjan | Mathematics | 11.38 GB .gz, ~34 GB unpacked | ~9.5B est. (4,924,517 documents after his dedup, 21 Oct) | his run removed 331 duplicates, 8,444 broken-encoding and 1 short document; the cleaned file is not on Drive yet |
| Prabanjan | Multi-step (rebuilt 21 Oct) | 1.18 GB, 18.0M Q&A | ~0.45B est. | DeepMind arithmetic only, procedurally generated; 804,443 exact duplicates (4.5%) |
| Prabanjan | Astronomy (21 Oct) | 391 MB, 754,260 records | ~0.11B est. | 500,000 are SDSS catalog rows; the rest are 247,809 abstracts and 6,451 exoplanets |
| Prabanjan | Quant / finance (21 Oct) | 166 MB, 1.27M lines | ~0.06B est. | numbers only: Binance candles, Ken French factors, SEC facts; no prose |
| Avinash | Universal index | 39,705,376 rows, 13.84 GB | 21.92B owner (cl100k over name/synonyms/description/properties/tags) | those columns = 58.3 GB, 86% `properties` JSON; ~48% structured DB records. Swaraj confirmed it is structurally valid (23 columns). On 22 Oct Avinash was re-merging Ravi's and Gokul's data into an updated file (Medical excluded: laptop disk); not yet delivered |
| Avinash | Medical | 41.6 GB zip, 922 PubMed `.xml.gz` files | ~6–8B est. (≈23.5M citations, title + abstract, from 3 sampled files) | raw; `scripts/pubmed_to_parquet.py` extracts it without unzipping; inclusion is Raju's decision |
| Afnan, Avinash | 6 engineering capabilities | — | — | nothing delivered |

## Findings
1. Token counts are not comparable (cl100k on serialized fields, chars/4, unstated, file sizes). The v4.0 files have no `text` column.
2. Overlap: the rebuilt Multi-step (21 Oct) is DeepMind arithmetic only, so the earlier GSM8K/NuminaMath overlap with Scientific Q&A is gone. Code vs Scientific Programming: 1 shared file name, 7,188 shared repositories (3.7%).
3. Duplicates: Gokul's 150,229 exact duplicates were removed on 22 Oct (see table); Scientific Programming still repeats files (Ravi's dedup pending).
4. Synthetic / non-prose: CAD 40.5% procedural (the rest is zerotocad); Multi-step is generated arithmetic; OpenFOAM part-00000 (250,000 rows) is synthetic parametric CFD templates; Avinash's index is largely database records.
5. Math: Raju's folder holds ~1.6 GB (4 shards); the 39.17 GB cleaned text was on Shrot's drive (he has left; Raju's re-shared folder holds the same 4 of 114 shards). Prabanjan's .gz (~34 GB unpacked) is the math source.
6. Three schemas (Gokul 18 cols, v4.0 21 cols, Avinash 23 cols). The training-text rule for the v4.0 files is settled (next section); Gokul's files already have `text`.
7. Tokenizer: Ravikant has no vocabulary (it cannot be trained before there is data). A 32K candidate was trained on a balanced 218 MB sample and verified (C++ parity 3003/3003, lossless round-trip) — `tokenizers/vocab32k_v2/`, SHA-256 `10b3c5ae…6b51cdc`. Nothing is tokenized before the team lead approves and freezes it.

## Training-text rule for the v4.0 files (`v4_text` in `scripts/tokenize_parquet.py`; agreed with Ravikant, 23 Oct)
Checked against the real files (field census over 16–24 evenly spaced row groups per file; CAD sources over all rows):
1. Code and documentation: `code_content`, `doc_content`, `text`, `content`. (`doc_content` exists in only 362 of 1,831,796 Scientific Programming rows.)
2. Q&A: Question, Options (when present), Answer, Explanation (not the `Source origin:` boilerplate). In 3 sampled row groups a third of explanations were empty, a third boilerplate and a third `Expected Answer: N`; over 16 row groups they average 195 characters.
3. CAD/STEP: DeepCAD pairs (`objective_prompt` + `sketch_extrude_completion`; only 5 rows exist), otherwise description + `cadquery_code` + `step_file_snippet` + `qa_pairs`. CAD sources, all 940,576 rows: zerotocad 59.5%, cad_synthetic 40.5% (procedural), deepcad_sft 0.0005%. `operations_hierarchy` (an AST dump of the CadQuery code, ~2,000 characters per zerotocad row) is deliberately left out: it is derived from the code.
4. Anything else: `description`.

## Plan to the first training run
1. Freeze the 32K tokenizer: candidate `tokenizers/vocab32k_v2/vocab32k.tok` is built and verified; approve, then freeze. Ravikant tokenizes with it.
2. One honest count: `scripts/tokenize_parquet.py --count-only` (Parquet) and `scripts/tokenize_text.py --count-only` (text) give rows, characters and ~tokens per domain in minutes.
3. Clean: `scripts/dedup_parquet.py` / `scripts/dedup_text.py` (exact, across files); cross-dataset near-duplicate removal is still open (Avinash).
4. Tokenize with `scripts/tokenize_parquet.py` / `scripts/tokenize_text.py` (measured: ~1.3M tokens/s on 8 cores, so 25B tokens ≈ 5 h; ~50 GB of output), merge per domain (`scripts/merge_bins.py`), check (`scripts/check_bins.py`).
5. Smoke test, then the run: budget, recipe, mix and checklist are in `docs/TRAINING_PLAN_1p5B.md`.

Model size vs data: ~20 tokens/parameter → 1.51B needs ~30B tokens, 2.0B ~40B, 3B ~60B. How long the run takes is decided by the compute budget, not by the data (plan §2).

## Decisions needed from Raju
The current list (tokenizer approval, compute budget and model size, data mix, Medical, engineering and general-language data,
true 3B vs 2B ceiling) is `docs/TRAINING_PLAN_1p5B.md` §9. The 20 Oct draft mix (Engineering 20%, general language 8%, ...)
cannot be filled because that data does not exist; the proposed mix for the data we do have is in plan §3.

## Not yet read in full
Gokul's shards (only sampled, for the tokenizer), the `chem_clean_*.txt` files (sampled), Prabanjan's Physics file, nine of ten OpenFOAM parts.
