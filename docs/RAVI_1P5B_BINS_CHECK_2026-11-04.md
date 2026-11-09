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


## Update 6 Nov 2026: new files from Ravi, and a problem in the chemistry bin
**Evidence labels.** *Run* = my own check. *Read* = Ravi's files (`docs/evidence/ravi_build_manifest_2026-11-06.json`, `docs/evidence/ravi_chem_build_config_2026-11-06.json`). I downloaded the five changed validation files (cad, chem, engineering, sciprog, sciqa, about 100 MB) from the Drive folder `1.5B Bins`; for the chemistry training file I read six 12 MB slices (range requests), not the whole 4.6 GB file. The other training files were not downloaded.

**Folder (run).** All six train and validation pairs are now there. File sizes equal 2 x tokens + 36 bytes for every bin (exact for the five validation files), so nothing is truncated.

**`build_manifest.json` (read).** One manifest for all six bins, same tokenizer hash (correct). It changes two numbers: sciqa train 1,901,493,585 to **1,880,407,792** (-21.1M) and sciprog train 1,672,410,945 to **1,606,252,084** (-66.2M); the other four are unchanged. The six bins now total **11,278,564,583** train tokens (was 11,365,809,237). It still has no per-bin sha256 and no document counts; its `config` lists only the sciprog build (filtered + supplement), and `chem_build.json` is only the chemistry build configuration.

**`check_bins` on the five downloaded validation files (run): PASS.** Tokens / documents: cad 4,063,889 / 4,647, chem 12,356,427 / 47,215, engineering 18,384,897 / 374, sciprog 7,637,929 / 2,151, sciqa 9,475,307 / 22,738. Together with `dedup_report.json`, kept rows equal train + validation documents **exactly** for cad (954,150 + 4,647 = 958,797), chem (9,339,542 + 47,215 = 9,386,757) and agent (69,586 + 374 = 69,960).

**sciqa: the source labels are gone (run).** "Expected Answer", "Augmentation strategy", "Benchmark source" and the other tags: 0 of 22,738 validation documents (before: 23.6%, 8.4%, 5.4%). Content is still MetaMath-style math word problems plus MedMCQA-style questions. 13,419 of the 22,738 validation documents are identical to documents of the old sciqa validation file.

**sciprog: better, not clean (run).** 50.3% of validation documents contain at least one science or numerics term from my list (before 23.6%); 11.1% contain three or more distinct terms. Only 688 of 2,151 documents also appear in the old validation file, so it was rebuilt with a new filter plus a supplement. Samples now include BLAS/SLATEC Fortran, SeqAn and Julia tests, but also Boost headers and selenium tests.

**chemistry: the bin contains the old sciqa data (run on validation, sampled on training). This is a blocker.**
- All **22,738** documents of the old sciqa validation file (labels included) are in `chem.val.bin`, identical after tokenizing. They are 48% of its 47,215 documents and **77.5% of its tokens** (9,581,589 of 12,356,427). The rest is real chemistry: 16,233 documents starting "Formula:", 6,386 "Compound", 1,858 "Reaction", 2,774,838 tokens.
- Six slices of 12 MB of `chem.train.bin` (185,550 whole documents): 27.7% of documents and **66.9% of tokens** start with "Question:", and 17,559 documents carry the source labels.
- Arithmetic: chemistry train 2,447,159,991 minus the old sciqa train 1,901,493,585 leaves about **546M tokens of real chemistry**; the document count fits the same split (9,339,542 minus 4,500,344 old sciqa documents = 4,839,198). The training-file figures are an inference; the validation figures are exact.
- Consequences: the "chemistry surplus" Raju accepted on 4 Nov does not exist (real chemistry is about 0.55B against a 1.2B allocation, 2.2 passes: still within 4); the source labels are back in the training data through the chemistry bin; and `dedup_report.json` could not see it because it deduplicates each file on its own.
- Unique tokens held, recomputed: 16,851M minus 87M (rebuilt bins) minus about 1,900M = **about 14.9B**.
- Fix (Ravi): rebuild chem from the chemistry rows only (no `scientific_qa` rows), re-send `chem.val.bin` and `chem.train.bin` with the `check_bins` output and sha256. Gate 0 of the pilot plan stays closed until then.

**engineering (agent trajectories) and cad (run, validation only).** engineering: 374 documents of mean 49,157 tokens, 78% start with an agent system prompt ("You are opencode...", "...a Claude agent..."); the source and licence of these trajectories are not recorded. cad: 52% of documents start "Parametric B-Rep model of a ..." (templated, as known).


## Update 8 Nov 2026: rebuilt chemistry bin verified (run)
Ravi re-uploaded `chem.train.bin` and `chem.val.bin` and posted his `check_bins` output. I downloaded both whole files (1.09 GB and 5.5 MB) and ran the checks myself.
- `check_bins` with the frozen tokenizer: **PASS**. Train 545,666,406 tokens / 4,839,198 documents, validation 2,774,838 / 24,477; file sizes equal 2 x tokens + 36 bytes. Identical to Ravi's numbers and to my 6 Nov arithmetic (2,447,159,991 - 1,901,493,585 = 545,666,406).
- sha256: `chem.train.bin` `f4b5e4469aac14a5a6e5cc9999338779a01242ce827f86c883eafc56962c095b`, `chem.val.bin` `eebd217d996af74ff7323b0b9f7bcb5b548404e8af0cf776363abdd156942479`.
- Documents starting "Question:" (the old sciqa format): **0 of 4,839,198 (train) and 0 of 24,477 (validation)**; in 20,000 random documents of each file no source label; the documents are "Formula:" (about 67%), "Compound:" (26%) and "Reaction:" (7%) records.
- Validation documents also present in train: **0**; duplicates inside train: **0** (all 4,839,198 distinct).
- Remark: mean document length is 113 tokens (property records of molecules, not running text); 546M tokens of it against a 1.2B allocation is 2.2 passes at 30B.
- **Chemistry blocker closed.** Six bins now total 9,377,070,998 train tokens (cad 840.6M, chem 545.7M, code 1,517.4M, engineering 2,986.7M, sciprog 1,606.3M, sciqa 1,880.4M). Unique tokens held (all 20 domains): **about 14.86B** (16,851M - 87M - 1,901M).
- Still open for Gate 0: validation-in-training overlap of sciqa, sciprog, code and cad (their training files were not downloaded); Engineering Reasoning bin; `build_manifest.json` has no per-bin sha256.
- **sha256 list from Ravi (8 Nov, `docs/evidence/ravi_bins_sha256_2026-11-08.txt`).** I re-hashed the files I hold: `chem.train`, `chem.val`, `cad.val`, `engineering.val`, `sciprog.val`, `sciqa.val` and `code.val` all **match** (run). The five other training files (cad, code, engineering, sciprog, sciqa; 16.5 GB together) were not downloaded, so their hashes are owner-reported only; their sizes equal 2 x tokens + 36 bytes (run).

## Gate 0 check on the whole training files (8 Nov 2026, run)
I downloaded the five remaining training files one at a time (16.5 GB in total, each deleted after the check) and ran: sha256 against Ravi's list, and exact (token level) overlap between each validation file and its own training file.
| bin | sha256 vs Ravi | train docs (distinct) | validation docs | validation docs also in train |
|---|---|---|---|---|
| cad | match | 954,150 (954,150) | 4,647 | 0 |
| code | match | 997,104 (997,104) | 4,974 | 0 |
| engineering | match | 69,586 (69,586) | 374 | 0 |
| sciqa | match | 4,500,344 (4,500,344) | 22,738 | 0 |
| sciprog | match | 428,476 (**420,929**) | 2,151 | **73** (3.4%) |
| chem (earlier today) | match | 4,839,198 (4,839,198) | 24,477 | 0 |
All 12 hashes in Ravi's list are now confirmed. **One defect: sciprog** has 7,547 exact duplicate documents inside the training file (1.8%) and 73 validation documents that also occur in training; fix = drop duplicates before the split (Ravi, small). This is exact-document overlap only; Namit's earlier figures (sciprog 25%, code 38%, chem 23%) count shared passages inside different documents, a different and looser test that I did not repeat on the new bins.

## Update 9 Nov 2026: Ravi's de-duplicated sciprog verified on the whole files (run)
Ravi re-uploaded `sciprog.train.bin` and `sciprog.val.bin` to the Drive folder `1.5B Bins` (new file ids; the old ones were deleted). I downloaded both whole files and ran the checks.
- sha256 **match Ravi's message**: train `ba35843d37ceb875176d1b0d281ffbacc043bb9ca7d6f2633a87cc893bc88f42`, validation `58476a497258a65f273c7e0426864327e5ec79f0ab9991e004319fc953ec52a0` (saved in `docs/evidence/ravi_sciprog_sha256_2026-11-09.txt`).
- `check_bins` with the frozen tokenizer: **PASS**. Train 1,587,016,100 tokens / 420,701 documents (3,026.99 MB), validation 7,492,746 / 2,116 (14.29 MB): identical to Ravi's numbers; file sizes equal 2 x tokens + 36.
- Duplicates inside train: **0** (420,701 distinct); inside validation: **0**; validation documents also in train: **0** (before: 7,547 duplicates and 73 overlapping).
- **Gate 0 data check closed for all six of Ravi's bins** (cad, chem, code, engineering, sciqa, sciprog: 0 validation-in-train, 0 duplicates in the two rebuilt ones). Six bins train tokens: cad 840,606,093; chem 545,666,406; code 1,517,422,577; engineering 2,986,716,046; sciprog 1,587,016,100; sciqa 1,880,407,792 = **9,357,835,014** (was 9,377,070,998).
- Not re-checked: the share of scientific content in sciprog (50% of validation documents had a science term on 6 Nov); Namit's `check_leakage.py` (benchmark questions) on the bins; engineering (agent) source and licence.

## Update 9 Nov 2026 (later): Ravi's answers on licence and the sciprog filter, checked
**Ravi's statement (9 Nov, group; read).** Engineering bin: source "OpenHands SWE-bench execution traces (open_swe_traces_openhands)", licence Apache-2.0. Sciprog filter: final; 65.4% of validation documents match his numerical/scientific keyword regex (24% on 6 Nov, 7% for general code); the rest are C++/Fortran/Julia support files from scientific repositories (Trilinos/Belos, ArrayFire, LLNL DESUL, NumFOCUS/ITK).
**My checks on the validation files (run).**
- sciprog (9 Nov upload, 2,116 documents): **50.9%** of documents contain at least one term of my list (6 Nov: 50.3%), 11.2% three or more distinct terms. The difference to his 65.4% is the term list, not the file; samples without a term are R, Julia, Boost and Fortran exercise files. Accepted as final (his statement); the file itself passes every check.
- engineering, 374 validation documents (keyword counts, a document can match several): "OpenHands" 42, "opencode" 126, "Claude" 45, "SWE-bench" 60, "str_replace_editor" 71, "Apache" 77. The first four documents are OpenHands traces. **So the bin is not a single OpenHands dataset**: Apache-2.0 may cover only part of it, and traces written by other agents or models can carry other terms. The statement is therefore **not accepted as the licence card** yet. Asked of Ravi: every source dataset inside `engineering.train.bin` with Hugging Face name, revision, licence and token share. The training file (2.99B tokens) was not scanned.
