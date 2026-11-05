# Data cleaning report: hashes and token statistics (Raju's task 1), status 5 Nov 2026

**Evidence labels.** *Run* = I ran `check_bins` or my own analysis on the file. *Read* = taken from the owner's output, PDF or manifest (numbers re-added by me, files not opened). *Computed* = arithmetic from the numbers in the table. **This report is not final**: four of the 20 bins are not in my hands (engineering reasoning, chemistry, CAD, agent trajectories) and five have known defects (last column: sciqa, sciprog, chemistry, domain 9, domain 10). A cleaning report cannot be called final until those are closed.

## 1. Fixed for every bin
- Tokenizer: `vocab32k_v2`, SHA-256 `10b3c5aecf6b81746bf86f8c18f8c576bf716a3620e7632e8fc2684006b51cdc` (run: `sha256sum` of `tokenizers/vocab32k_v2/vocab32k.tok` on 5 Nov equals the frozen value; `n_vocab` 32,000, end-of-text id 31,936; guarded by a test).
- Integrity gate: `scripts/check_bins.py --vocab ...` (header, size, id range, end-of-text, decode and re-encode on 20 chunks per file). A bin passes only if all of this passes.
- Document boundaries: stated per bin in its build config and verified by mean document length (`eot`, `start`, `marker`, `blank` rules in `scripts/tokenize_text.py`).
- Rules for every new bin (unchanged): exact-duplicate removal; train and validation never from the same document; a 20 MB source sample first; licence and source recorded; the build manifest (tokenizer hash, commit, config hash, token counts) kept next to the bin.

## 2. Inventory: 20 domains of Raju's 30B allocation (train tokens)
| Domain | Owner | Allocation (B) | Train tokens held (M) | Evidence | Open defect |
|---|---|---|---|---|---|
| math | Prabanjan | 2.4 | 995.7 | run (whole file, 28 Oct) | none known |
| physics | Prabanjan | 2.1 | 637.2 | run | about 13% of documents with little physics |
| engineering reasoning | Prabanjan / Pankaj rebuild | 2.4 | 805.0 | run on 31 Oct; **file not on Pankaj's laptop on 5 Nov** (search of Downloads, Documents, Desktop) | file location: path or Drive upload needed for the dataset card |
| multistep | Prabanjan | 2.1 | 88.6 | run (24 Oct) | generated arithmetic, low information |
| astronomy | Prabanjan | 0.9 | 20.3 | run | tiny |
| finance | Prabanjan | 0.6 | 17.2 | run | tiny |
| generative engineering | Namit | 1.8 | 586.0 | run + his manifest (BIGPATENT) | none known |
| experimental validation | Namit | 1.2 | 628.9 | run + manifest | none known |
| technical communication | Namit | 0.6 | 534.5 | run + manifest | none known |
| scientific discovery | Gokul | 1.2 | 1,001.6 | run (whole file, 31 Oct) | none known |
| biology | Gokul | 1.2 | 44.6 | run | tiny against the allocation (26.7 passes) |
| earth | Gokul | 0.9 | 73.2 | run | tiny (12.3 passes) |
| code | Ravi | 1.8 | **1,517.4** (new bin; was 322.8) | read; validation file run | a licence banner starts 2.6% of validation documents |
| scientific Q&A (sciqa) | Ravi | 1.5 | **1,901.5** (was 188.6) | read; validation file run | **source labels still present** ("Expected Answer" 23.6%, "Augmentation strategy" 8.4%, "Benchmark source" 5.4% of validation documents) |
| scientific programming | Ravi | 1.5 | **1,672.4** (was 49.8) | read; validation file run | **mostly general code**: only 23.6% of validation documents contain a science or numerics term |
| chemistry | Ravi | 1.2 | **2,447.2** (was 127.6) | read only; **bin not seen** | mean document 262 tokens; near-duplicates not checked; dedup proof is arithmetic only |
| CAD | Ravi | 1.2 | **840.6** (was 78.4) | read only; **bin not seen** | templated (about 40% procedural) |
| engineering agent trajectories | Ravi | 1.2 | **2,986.7** (was 813.9) | read only; **bin not seen** | 2.5 times the allocation |
| numerical simulation (domain 9) | Swaraj, owner open | 2.1 | 39.8 (+0.09 validation) | run (v7, whole file) | volume: 52 passes at 30B |
| physics + AI (domain 10) | Swaraj, owner open | 2.1 | 13.1 | run (v7, whole file) | volume: 161 passes at 30B; **validation file empty** (0 of 930 documents) with the default split; `--val-frac 0.02` fixes it (run: 909 + 21 documents, PASS) |

## 3. Hashes of the files I hold (run, sha256 of the whole file)
| File | Bytes | SHA-256 |
|---|---|---|
| Ravi `code.val.bin` | 15,653,534 | `120c8ef0fc482388f639b41aeeed548f306e198ccd43a17c0bb12d691394c5ac` |
| Ravi `sciprog.val.bin` | 17,874,970 | `4ba3cb95e967ab216e020011c0a7fe8bd854aaab187fc0b56ff2002718cfe641` |
| Ravi `sciqa.val.bin` | 19,163,214 | `eb13cd3e7bc481306ec5c5c4dfd4f02eb6750703b9a99e781dec0b51b3f5b9b9` |
| Domain 9 corpus text (v7) | 162,417,282 | `352184264fca0dea1f302291199a49f13a08dfc606abee25f8345dc35f6db24e` |
| Domain 10 corpus text (v7) | 50,086,930 | `397196886a475804a0192bdd726396e78b8c593490359068a9a97d81d94e87c0` |
| Domain 9 train bin (built by me) | 79,606,662 | `0ceddc101ea2ebc51d06302ea273b693325d8062e7953fb4f0f3fa27451c0a3a` |
| Domain 9 val bin (built by me) | 189,218 | `365fd30faf9c1ef6792a86168df133a1da942cc8a8f6cf0ef663598e17796044` |
| Domain 10 train bin (built by me) | 26,127,036 | `5843211f84ab3d644599afdcad47ffda6d82b0e10a63fa700eac57eadeb1dc4e` |
| Domain 10 val bin (built by me, empty: header only) | 36 | `adc5fd0ac733f4a3c74b5194a5a99f226a9ca0feba879cd0e7cb396f0f296a7e` |
| `check_bins` result on these nine on 5 Nov | | PASS on all but the empty Domain 10 validation file (0 tokens: "FAIL no tokens", expected) |
The hashes of the 17 bins used in the 100M run are in each owner's `SHA256SUMS.txt` and in the run's preflight log; I do not hold those files (see `docs/A100_RUN_RESULT_2026-11-01.md`). The six new Ravi bins (about 17 GB) have been downloaded only as three validation files; their training-file hashes are not mine to give.

## 4. Statistics (computed)
- Verified unique train tokens before Ravi's new bins: **7.07B** (`DATA_PLAN_30B_2026-11-01.md`). If Ravi's six new bins are accepted as built: **16.85B** (7,068M minus his old 1,582M plus his new 11,366M).
- Passes over the data at the 30B allocation, new bins included: 7 of 20 domains above 4 passes (biology 26.7, earth 12.3, multistep 23.9, astronomy 45.0, finance 35.3, domain 9 52.5, domain 10 161.5), down from 12. Tokens trainable at no more than 4 passes: **21.3B of 30B**. New unique tokens needed to bring the seven domains to 4 passes: about **2.2B** (biology 0.26, earth 0.15, multistep 0.44, astronomy 0.21, finance 0.13, domain 9 0.49, domain 10 0.51).
- These figures hold only if the three unseen bins exist as reported and the four defects above are repaired without losing much volume (removing sciqa labels and filtering sciprog to scientific code will shrink those two bins; by how much is not known).

## 5. What is needed to make this final
1. Ravi: upload chem, cad and engineering bins (Raju's folder from Avinash does not exist yet, `docs/AVINASH_DRIVE_CHECK_2026-11-05.md`); remove sciqa source labels; filter sciprog; near-duplicate check and `build_manifest.json` for chem (`dedup_report.json` received 5 Nov evening, verified against the bins: `RAVI_1P5B_BINS_CHECK_2026-11-04.md`); remove validation documents that also occur in training (Namit's overlap: sciprog 25%, code 38%, chem 23% of validation passages).
2. Prabanjan or Pankaj: the engineering reasoning bin location.
3. Domain 9 / 10 owner: volume (`SWARAJ_COLLECTOR_REVIEW_2026-11-05.md`); rebuild both with `--val-frac 0.02` (tested for domain 10).
4. Then Pankaj re-runs `check_bins` on every file, recounts, hashes and issues the final report.
