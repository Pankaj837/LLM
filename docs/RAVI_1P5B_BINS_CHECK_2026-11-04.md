# Ravi's six 1.5B bins: check of 4 Nov 2026

**Evidence labels.** *Read* = Ravi's pasted `check_bins` output and `token_summary.pdf`. *Run* = I downloaded the three validation files from the Drive folder `1.5B Bins` and ran `check_bins` and my own analysis on them. Only the validation files were downloaded; the 2.8 to 3.5 GB training files were not, so statements about training files are inferences from the validation files, built by the same pipeline.

## Token summary (read; sums checked by me)
| bin | train tokens | val tokens | docs (train) | mean doc | in Drive folder on 4 Nov |
|---|---|---|---|---|---|
| engineering (agent/tool use) | 2,986,716,046 | 18,384,897 | 69,586 | 42,921 | not seen |
| chem | 2,447,159,991 | 12,356,427 | 9,339,542 | 262 | not seen |
| sciqa | 1,901,493,585 | 9,581,589 | 4,500,344 | 423 | yes (3.54 GB) |
| sciprog | 1,672,410,945 | 8,937,467 | 489,280 | 3,418 | yes (3.12 GB) |
| code | 1,517,422,577 | 7,826,749 | 997,104 | 1,522 | yes (2.83 GB) |
| cad | 840,606,093 | 4,063,889 | 954,150 | 881 | not seen |
| total | 11,365,809,237 | 61,151,018 | | | 11,426,960,255 in all |
The frozen tokenizer hash in his output is `10b3c5ae...b51cdc` (correct). Documents x mean document length reproduces each train total. Allocation in Raju's 30B table for the same six: chem 1.2B, sciqa 1.5B, sciprog 1.5B, code 1.8B, cad 1.2B, agent 1.2B (8.4B in all; Raju accepted the chemistry surplus in the group).

## Run: the three validation files (`code.val.bin`, `sciprog.val.bin`, `sciqa.val.bin`)
`check_bins` PASS; tokens, documents and mean document length equal Ravi's pasted numbers exactly (7,826,749 / 4,974 / 1,574; 8,937,467 / 2,462 / 3,630; 9,581,589 / 22,738 / 421). Files carry an 18-token header. No exact duplicate documents inside any of the three.

**sciqa (22,738 validation documents): the source labels are still there.** Documents containing: "Expected Answer" 5,360 (23.6%); "Augmentation strategy" 1,901 (8.4%); "Benchmark source" 1,225 (5.4%); "GSM_" 1,154; "data/CoT" 857; "AnsAug" 697; "Rephrased" 665; "data/PoT" 368; "FOBAR" 281; "math50k" 240; "gsm_rft" 93. Content is MetaMath / MathInstruct-style math, MedMCQA-style medical questions, and others. Raju asked for unwanted source labels to be removed; they have not been. Benchmark leakage (Namit found 39 benchmark questions in the previous sciqa) cannot be tested without his question set; the same source families are present.

**sciprog: mostly general code, not scientific code.** On a broad list of science and numerics terms (numpy, scipy, cuda, mpi, simulation, solver, tensor, fft ...), 23.6% of validation documents contain at least one and 5.7% contain three or more, against 7.4% and 1.5% for the `code` file. Samples without any term include a crypto-exchange library (ccxt), a Rust FAT file-system, a Japanese-text utility library, an SQL migration, and a page about an election candidate (not code). Top-10 document starts (licence banners) cover 6.5% of documents.

**code: language-balanced looks plausible.** Heuristic markers per document: Python 17.8%, TypeScript/JavaScript 23.0%, Go 9.7%, Java 6.6%, C/C++ 5.7%, C# 3.7%, Rust 2.5% (earlier source was 83% TypeScript). 129 of 4,974 documents (2.6%) start with the same Apache licence header. No source tags.

## Open
- `chem`, `cad` and `engineering` bins are not in the Drive folder; chem mean document is 262 tokens; whether chem was built from the verified deduplicated file is not stated (the deduplication check was still pending at 10:40 am on 4 Nov per the group chat).
- Train-versus-validation overlap and train-file checks not done (training files not downloaded).

## Chemistry proof from Ravi (read; `chemistry_dedup_and_token_proof.md`, received 4 Nov evening)
Checked arithmetic only; the files it cites (`dedup_report.json`, `build_manifest.json`, the parquet files) were not provided on 4 Nov.

**Update 5 Nov: `dedup_report.json` received** (copy and hash in `docs/evidence/`, sha256 `21eb9769d129d05655a5462afa44dbd8052a7110275c5836e13e1399d9052337`). Run: totals add up (18,164,444 in = 16,432,416 kept + 1,732,028 duplicate), and for every file kept rows = train documents + validation documents of the bin: code 1,002,078 = 997,104 + 4,974; sciqa 4,523,082 = 4,500,344 + 22,738; sciprog 491,742 = 489,280 + 2,462 (all exact); chemistry 9,386,757, cad 958,797 and agent 69,960 agree within the validation counts I could derive (about 47K, 4.6K and 374). Chemistry lost 15.58%, cad 166, code 3; sciqa, sciprog and agent show 0 duplicates. Limits: the file gives counts only, with no hashes, no definition of "duplicate" (exact text or exact row) and no near-duplicate check; `build_manifest.json` is still missing.
- 11,118,616 rows in, 1,731,859 duplicates dropped, 9,386,757 kept: correct (15.58%).
- Train documents 9,339,542 + validation documents 47,215 = 9,386,757 = kept rows exactly. This supports "one document per row of the cleaned file" and fits the bins coming from the deduplicated parquet.
- Token numbers equal `token_summary.pdf` (train 2,447,159,991; val 12,356,427; mean 262 in both).
- Not proven: the split into 4,863,675 database records (about 90 tokens) and 4,523,082 Q&A records (about 447 tokens) is stated without a count from the data, and the weighted mean is built from those assumed lengths, so it explains 262 but does not independently prove it. By his own numbers the database records are about 52% of documents but only about 18% of tokens.
- Deduplication was exact match on `text` only; near-duplicates (same molecule in different wording, repeated Q&A) were not checked.
- Ravi says Raju asked Avinash to create one Drive folder for all domain bins and he will upload chem, cad and engineering there once he has the link (owner-reported).
