# Data status and next plan — 20 Oct 2026

Prepared by Pankaj. Sizes and row counts below were read from the files (Parquet footers, sampled rows, gzip trailer);
token counts marked "owner" are the owners' own and unverified; "est." are my estimates.

## Where the data stands

| Owner | Domain | Size | Tokens | Status |
|---|---|---|---|---|
| Ravikant | Scientific Programming | 3.82 GB, 1,831,796 rows | 4.56B owner; ~1.2–1.9B est. after repeats | v4.0 schema; 491,834 distinct file names, ~73% of rows repeat a name |
| Ravikant | CAD/STEP | 0.93 GB | 2.00B owner | 41% procedurally generated |
| Ravikant | Scientific Q&A | 7.23 GB, 5,929,942 rows | 2.50B owner | each Q&A stored 3× (name, description, properties); 49% math Q&A |
| Ravikant | Code | 3.29 GB, 2,692,769 rows | 3.28B owner (report of 21 Oct; row count matches the file; counter not stated) | 83% of tokens are TypeScript; Python only 6 files; "100% permissive" and "zero leakage" are his claims, not checked |
| Gokul | Biology / Earth & Env. / Sci. Discovery | 5.8 / 4.2 / 2.8 GB | ~8.8B owner after exact dedup | 18 columns with `text`; 150,226 exact duplicates still in files |
| Prabanjan | Physics | 2.33 GB | ~0.6B est. | 165,528 docs |
| Prabanjan | Mathematics | 11.38 GB .gz, ~34 GB unpacked | ~9.5B est. (4,924,517 documents after his dedup, 21 Oct) | his run removed 331 duplicates, 8,444 broken-encoding and 1 short document; the cleaned file is not on Drive yet |
| Prabanjan | Multi-step (rebuilt 21 Oct) | 1.18 GB, 18.0M Q&A | ~0.45B est. | DeepMind arithmetic only, procedurally generated; 804,443 exact duplicates (4.5%) |
| Prabanjan | Astronomy (21 Oct) | 391 MB, 754,260 records | ~0.11B est. | 500,000 are SDSS catalog rows; the rest are 247,809 abstracts and 6,451 exoplanets |
| Prabanjan | Quant / finance (21 Oct) | 166 MB, 1.27M lines | ~0.06B est. | numbers only: Binance candles, Ken French factors, SEC facts; no prose |
| Avinash | Universal index | 39,705,376 rows, 13.84 GB | 21.92B owner (cl100k over name/synonyms/description/properties/tags) | those columns = 58.3 GB, 86% `properties` JSON; ~48% structured DB records |
| Avinash | Medical | 41.6 GB zip, 929 PubMed XML.gz | unknown | raw, unprocessed |
| Afnan, Avinash | 6 engineering capabilities | — | — | nothing delivered |

## Findings
1. Token counts are not comparable (cl100k on serialized fields, chars/4, unstated, file sizes). The v4.0 files have no `text` column.
2. Overlap: the rebuilt Multi-step (21 Oct) is DeepMind arithmetic only, so the earlier GSM8K/NuminaMath overlap with Scientific Q&A is gone. Code vs Scientific Programming: 1 shared file name, 7,188 shared repositories (3.7%).
3. Duplicates: Gokul 150,226 exact; Scientific Programming repeats files (see table).
4. Synthetic / non-prose: CAD 41% procedural; OpenFOAM part-00000 (250,000 rows) is synthetic parametric CFD templates; Avinash's index is largely database records.
5. Math: Raju's folder holds ~1.6 GB (4 shards); the 39.17 GB cleaned text was on Shrot's drive (he has left; Raju's re-shared folder holds the same 4 of 114 shards). Prabanjan's .gz (~34 GB unpacked) is the math source.
6. Three schemas (Gokul 18 cols, v4.0 21 cols, Avinash 23 cols) and no agreed training-text rule.
7. Tokenizer: Ravikant has no vocabulary (it cannot be trained before there is data). A 32K candidate was trained on a balanced 218 MB sample and verified (C++ parity 3003/3003, lossless round-trip) — `tokenizers/vocab32k_v1/`, SHA-256 `50a0910f…df32da9`. Nothing is tokenized before the team lead approves and freezes it.

## Training-text rule for the v4.0 files (`v4_text` in `scripts/tokenize_parquet.py`; agreed with Ravikant, 23 Oct)
Checked against the real files (field census over 16–24 evenly spaced row groups per file; CAD sources over all rows):
1. Code and documentation: `code_content`, `doc_content`, `text`, `content`. (`doc_content` exists in only 362 of 1,831,796 Scientific Programming rows.)
2. Q&A: Question, Options (when present), Answer, Explanation (not the `Source origin:` boilerplate). In 3 sampled row groups a third of explanations were empty, a third boilerplate and a third `Expected Answer: N`; over 16 row groups they average 195 characters.
3. CAD/STEP: DeepCAD pairs (`objective_prompt` + `sketch_extrude_completion`; only 5 rows exist), otherwise description + `cadquery_code` + `step_file_snippet` + `qa_pairs`. CAD sources, all 940,576 rows: zerotocad 59.5%, cad_synthetic 40.5% (procedural), deepcad_sft 0.0005%. `operations_hierarchy` (an AST dump of the CadQuery code, ~2,000 characters per zerotocad row) is deliberately left out: it is derived from the code.
4. Anything else: `description`.

## Plan to the first training run
1. Freeze the 32K tokenizer: candidate `tokenizers/vocab32k_v1/vocab32k.tok` is built and verified; approve, then freeze. Ravikant tokenizes with it.
2. One honest count: each owner runs `python scripts/tokenize_parquet.py <files> --count-only --out runs/count` (reads v4.0 files too) and posts the totals.
3. Clean and merge: exact dedup, cross-dataset near-dup, remove overlaps, one schema; start engineering and general-language sources in parallel.
4. Tokenize on the training machine (~0.2M tokens/s/core; 25B tokens ≈ 2 h on 16 cores, ~50 GB output).
5. Smoke run, then the full 1.5–3B run (Om's config, Gokul's GPUs).

Model size vs data: ~20 tokens/parameter → 1.51B needs ~30B tokens, 2.0B ~40B, 3B ~60B. Earliest full run: 2–3 days.

## Decisions needed from Raju
1. Start with the 1.51B model on ~30B cleaned tokens and grow with the data?
2. Domain mix draft: Engineering 20%, Math 15%, Code+Sci. Programming 15%, Physics+Astronomy 10%, Chemistry 8%, Biology+Earth 8%, Sci. Q&A+Discovery 8%, general language 8%, Quant 5%, CAD/STEP 3%.
3. Owners for the empty engineering capabilities and general-language data.
4. Machine for tokenizing and training (≥200 GB free disk, 16+ cores) — Gokul's H100?

## Not yet read in full
Gokul's shards (only sampled, for the tokenizer), the `chem_clean_*.txt` files (sampled), Prabanjan's Physics file, nine of ten OpenFOAM parts.
