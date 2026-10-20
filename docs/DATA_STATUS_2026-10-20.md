# Data status and next plan — 20 Oct 2026

Prepared by Pankaj. Sizes and row counts below were read from the files (Parquet footers, sampled rows, gzip trailer);
token counts marked "owner" are the owners' own and unverified; "est." are my estimates.

## Where the data stands

| Owner | Domain | Size | Tokens | Status |
|---|---|---|---|---|
| Ravikant | Scientific Programming | 3.82 GB, 1,831,796 rows | 4.56B owner; ~1.2–1.9B est. after repeats | v4.0 schema; 491,834 distinct file names, ~73% of rows repeat a name |
| Ravikant | CAD/STEP | 0.93 GB | 2.00B owner | 41% procedurally generated |
| Ravikant | Scientific Q&A | 7.23 GB, 5,929,942 rows | 2.50B owner | each Q&A stored 3× (name, description, properties); 49% math Q&A |
| Ravikant | Code | 3.29 GB, 2,692,769 rows | ~2.4B est. (file's own token_count, 2 of 52 row groups) | delivered 19 Oct, no report |
| Gokul | Biology / Earth & Env. / Sci. Discovery | 5.8 / 4.2 / 2.8 GB | ~8.8B owner after exact dedup | 18 columns with `text`; 150,226 exact duplicates still in files |
| Prabanjan | Physics | 2.33 GB | ~0.6B est. | 165,528 docs |
| Prabanjan | Mathematics | 11.38 GB .gz, ~34 GB unpacked | ~9–10B est. | mojibake (U+FFFD) and junk lines; needs cleaning |
| Prabanjan | Multi-step / Quant / Astronomy | 717 / 485 / 90 MB | ~0.25 / 0.12 / 0.05B est. | Multi-step overlaps Sci. Q&A; Quant ~23% off-topic; Astronomy = catalog rows |
| Avinash | Universal index | 39,705,376 rows, 13.84 GB | 21.92B owner (cl100k over name/synonyms/description/properties/tags) | those columns = 58.3 GB, 86% `properties` JSON; ~48% structured DB records |
| Avinash | Medical | 41.6 GB zip, 929 PubMed XML.gz | unknown | raw, unprocessed |
| Afnan, Avinash | 6 engineering capabilities | — | — | nothing delivered |

## Findings
1. Token counts are not comparable (cl100k on serialized fields, chars/4, unstated, file sizes). The v4.0 files have no `text` column.
2. Overlap: Multi-step = GSM8K + NuminaMath-CoT, already inside Scientific Q&A. Code vs Scientific Programming: 1 shared file name, 7,188 shared repositories (3.7%).
3. Duplicates: Gokul 150,226 exact; Scientific Programming repeats files (see table).
4. Synthetic / non-prose: CAD 41% procedural; OpenFOAM part-00000 (250,000 rows) is synthetic parametric CFD templates; Avinash's index is largely database records.
5. Math: Raju's folder holds ~1.6 GB (4 shards); the 39.17 GB cleaned text is on Shrot's D drive; Prabanjan's .gz may be that text (unverified).
6. Three schemas (Gokul 18 cols, v4.0 21 cols, Avinash 23 cols) and no agreed training-text rule.
7. Tokenizer gate: Ravikant's final 32K vocab not yet received. Nothing is tokenized before it is verified and its SHA-256 frozen.

## Plan to the first training run
1. Freeze the 32K tokenizer (Ravikant sends; Pankaj verifies format, byte tokens, round-trip, C++/Python parity). If trained on a narrow sample, retrain on a balanced 1–2 GB sample.
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
Gokul's shards, the `chem_clean_*.txt` files, Prabanjan's Physics file, nine of ten OpenFOAM parts.
