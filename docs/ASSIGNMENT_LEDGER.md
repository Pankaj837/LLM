# Assignment ledger — what was allotted, to whom, when, and what has been delivered (25 Oct 2026; section "Since 25 Oct" added 5 Nov)

Built from the group messages and files Pankaj showed, from the first assignment (10 Oct) to now. "Done" means a file or result was
actually seen; "claimed" means the owner said so but it could not be verified; "not delivered" means nothing has been seen. Days late are
counted from the due date to 25 Oct. Work done in private chats that was not shown is not counted.

## The original split (Raju, 10 Oct: "Try to complete it by 10th Oct")
| Owner | Allotted | Delivered |
|---|---|---|
| Prabanjan | Mathematics, Physics, Multi-step, Astronomy & Space, Quant Finance | all five (late, see below) |
| Aarav | Chemistry, Scientific Q&A, Scientific Discovery, Biology, Earth & Environment | none: removed 15 Oct, domains reassigned 14 Oct |
| Afnan | Engineering Reasoning, Numerical Simulation, Physics + AI, Engineering | none: left the internship 24 Oct |
| Ravi | Code, CAD/STEP, Generative Engineering, Scientific Programming | three of four |
| Avinash | Experimental Validation, Technical Engineering Communication, Engineering-agent / Tool Use, dataset-wide dedup / quality / final validation | the index only |

## Per person
### Prabanjan — 5 domains, due 10 Oct
| Task | Status | Evidence |
|---|---|---|
| Mathematics | **Done, ~11 days late** | cleaned file (34 GB) on Drive since 21 Oct; tokenized 486.5M tokens 25 Oct (first 5% only: rerun with `spread` pending) |
| Physics | **Done, late** | 2.33 GB file; 25 Oct: count posted (632M est.); tokenization still to run |
| Multi-step | **Done 21 Oct** (rebuilt) | generated arithmetic; tokenized 88.6M 25 Oct |
| Astronomy & Space | **Done 21 Oct** | 66% catalog rows; tokenized 20.3M 25 Oct |
| Quant Finance | **Partly** | numbers-only finance file delivered and tokenized (17.2M); the earlier 463 MB instruction-text corpus (19 Oct) is not in the shared folder now |
| 50M models Bio + Earth, Physics | Done before 10 Oct | verified in the repo |
| 24 Oct board (upload math, physics count) | **Done** | link checked, files read |

### Ravi — 4 domains (+2 from Aarav on 14 Oct), due 10 Oct
| Task | Status | Evidence |
|---|---|---|
| Code | **Done, ~9 days late** | 2.69M files, report 24 Oct |
| CAD/STEP | **Done** (19 Oct) | 940,576 rows |
| Scientific Programming | **Done, dedup claimed** | 19 Oct; dedup "done" 24 Oct: not verifiable (link was the Pranag AI root, no cleaned files visible) |
| **Generative Engineering** | **NOT delivered, 15 days late** | nothing seen |
| Chemistry (from Aarav) | Done earlier | text exists; "chemistry is already done" (19 Oct); not integrated, folder no longer visible |
| Scientific Q&A (from Aarav) | **Done** (19 Oct) | 5.93M rows |
| **32K tokenizer** | **NOT delivered** | reported ready on 10 Oct; on 24 Oct "vocab only after the final dataset"; built by Pankaj |
| Tokenization / data readiness ("by today evening", Raju 19 Oct) | **NOT done by Ravi, 6 days late** | the first token files came from Prabanjan on 25 Oct |
| Training-text fields for the v4.0 files | Done 23 Oct | his rule adopted after checking it against the files |
| 50M Chemistry, Math, Engineering | Done before 10 Oct | verified |

### Gokul — joined 10 Oct; Scientific Discovery, Biology, Earth & Environment from 14 Oct
| Task | Status | Evidence |
|---|---|---|
| The three domains | **Done** (17–18 Oct) | 18-column files, ~8.5B tokens est. |
| Exact dedup | **Done** (22 Oct) | 150,229 duplicates removed; his own pending list (19 Oct): language check, MinHash near-dedup, exact token count — not reported |
| H100 availability / benchmark / infrastructure (21 Oct) | **NOT done** | no H100 session has happened (₹1,000 credit); only a laptop RTX 4050 run (small preset) |
| 24 Oct board: Drive links of the cleaned folders | **Not received** | |
| 24 Oct board: smoke test | **Partly** | laptop run worked (loss 9.71 → 7.57); not on the H100 |

### Om — architecture / infrastructure for 1.5–3B (10 Oct, "first benchmark as soon as the single-GPU run is done")
| Task | Status | Evidence |
|---|---|---|
| **Config, GQA check at scale, FSDP/ZeRO-3, FP8, FlashAttention-3, YaRN 4K→32K, benchmarks** | **NOT delivered, 15 days** | no code and no numbers received. Stated blocker: no H100/H200; his own plan said benchmarks could proceed with dummy tokens |
| 24 Oct board: smoke test numbers | **Not received** | only "I will share the first benchmark results" |
| 50M Quant Finance | **Done** (10 Oct) | the best-documented run in the project |

### Swaraj
| Task | Status | Evidence |
|---|---|---|
| Fix the 51M things and test with PINN and parser teams (10 Oct) | **Partly, ~14 days late** | 24 Oct: 3 written gaps (capacity, instruction following, entity/unit binding) |
| Gap analysis, at least 5 points (21 / 24 Oct) | **3 of 5** | |
| Verify Avinash's index | **Done** (19 Oct) | structurally valid, 23 columns |
| 3B focus (Raju, 24 Oct) | Open | |

### Avinash
| Task | Status | Evidence |
|---|---|---|
| **Experimental Validation (12), Technical Engineering Communication (14), Engineering-agent / Tool Use (15)** | **NOT delivered, 15 days late** | nothing seen |
| Dataset-wide dedup, quality filtering, final validation | **NOT done** | the 15 Oct index is not deduplicated across datasets; the merged file promised for 21–22 Oct was not received ("64M+ rows" mentioned 24 Oct, not uploaded) |
| Map tokens to the 15 capabilities (asked 19 Oct) | **NOT delivered** | on 24 Oct he asked what the list is |
| Universal index (15 Oct) | **Done** | 39.7M rows, verified structurally |
| Medical dataset | Not processed | laptop limits |
| Steps 4–6 (intrinsic properties, embeddings, cache) | Not done | postponed by Raju; needed later by the PINN side |

### Left or removed
| Person | Allotted | Delivered |
|---|---|---|
| Afnan (left 24 Oct) | capabilities 4, 9, 10 and Engineering | nothing; also did not train the Math / Engineering 51M models (compute issues) |
| Aarav (removed 15 Oct) | 5 domains, 51M Scientific Programming | domains reassigned; the checkpoint arrived only as Git LFS pointers |
| Shrot (left) | Mathematics preprocessing, 8K tokenizer | 39 GB cleaned text never reachable; the 8K trainer did not finish |

### Pankaj
| Task | Status |
|---|---|
| Fix the 51M things by 10 Oct | Done: repo, verification of every submission |
| "The next plan" (Raju, 19–20 Oct) | Done 20 Oct, revised since |
| Verify and freeze the 32K tokenizer | Done differently: built, verified and frozen (25 Oct) because none was delivered |
| Progress report, roles and milestones (21 Oct) | Done |
| "Complete training and testing by the end of this week" (Raju, 20 Oct) | **Not possible**: 3 GPU hours in total, no more allowed |

## Since 25 Oct (added 5 Nov 2026; sources: group messages as relayed by Pankaj, and the checks named in each row)
| Date | Assignment or event | Status on 5 Nov |
|---|---|---|
| 27 Oct (Raju) | Capability owners: 9 and 10 Swaraj; 4 Prabanjan; 11, 12, 14 Namit; 15 Gokul / Ravi; target 500M to 1B tokens each. Swaraj to own the LLM project with Pankaj | Namit's three domains are built and verified (generative engineering 0.59B, experimental validation 0.63B, technical communication 0.53B tokens; `DATA_PLAN_30B_2026-11-01.md`); Prabanjan's capability 4 is 0.81B tokens (bin not on Pankaj's laptop, see `PROGRESS_REPORT.md`); Ravi's engineering agent trajectories are 2.99B tokens (reported, file not seen) |
| 27 Oct | First H100 session (Gokul) | done, owner-reported (`FIRST_RUN_ANALYSIS_2026-10-27.md`) |
| 31 Oct to 1 Nov | 22-hour A100 slice (Prabanjan's account) | 100M run finished, evaluated (`A100_RUN_RESULT_2026-11-01.md`) |
| 1 Nov | Swaraj: final Domain 9 / 10 handover | delivered; v7 verified on 4 Nov (39.90M and 13.06M tokens) |
| 2 Nov (Raju) | 30B allocation over 20 domains | gap analysis in `DATA_PLAN_30B_2026-11-01.md` |
| 4 Nov (Raju) | Six pre-pilot tasks: cleaning report, three dataset cards, evaluation of "the 1.5B checkpoint", checkpoint reliability, reproducible evaluation, pilot plan | 4, 5 done; 6 drafted; 1, 2, 3 open (`PREPILOT_TASKS_STATUS_2026-11-04.md`) |
| 4 Nov | Swaraj leaves the internship (family medical situation) | Domains 9 and 10 have no owner; Raju to assign; Pankaj proposed Avinash |
| 4 Nov (Raju) | Domains 9 and 10 to 60 to 80M tokens each for the 1B run | open; the two named sources are exhausted (`SWARAJ_COLLECTOR_REVIEW_2026-11-05.md`) |
| 6 Nov (Pankaj, group) | Domain 9 (+485M) reassigned to Namit, Domain 10 (+512M) to Ravi; Ravi asked "do i have to do domain 10?"; **Raju: "no pankaj will cover it"** | **Domain 9: Namit (not yet confirmed by Namit); Domain 10: Pankaj (Raju), collection waits for Raju's decision on the allocation, else 9-10 Nov.** Swaraj and Avinash (laptop dead) own nothing now | Swaraj's delivered v7 files (39.90M and 13.06M tokens) are the only Domain 9 / 10 data held. 9 Nov: Pankaj told me to collect Domain 10; collector `scripts/collect_domain10_openalex.py` running |
| 4 Nov (Raju) | Avinash to create one Drive folder for all domain bins | not done: the 5 Nov link is the 24 Oct data lake (`AVINASH_DRIVE_CHECK_2026-11-05.md`) |
| 5 Nov | Namit's evaluation branch `namit/science-eval` pushed by Pankaj; evaluation reproduced | done (`EVAL_REPRODUCTION_2026-11-04.md`); two Windows bugs fixed by Namit in `eefe8cf` (diff reviewed 5 Nov, not run) |

## What is still open, by owner (as of 25 Oct; the current list is in `PROGRESS_REPORT.md`, "Open items by owner")
Ravi: Generative Engineering; verifiable cleaned files; his domains' token files. Avinash: capabilities 12, 14, 15; the merged and deduplicated file;
the 15-capability mapping; the 1M-row sample for the PINN tests. Om: the architecture and benchmark deliverables. Gokul: the H100 session, cleaned-folder
links, token files for his three domains. Swaraj: two more gaps, the `scientific_ir` examples, capabilities 9 and 10 if Raju confirms. Prabanjan: physics token
file, math rerun with `spread`. Raju: the shared Drive folder, Afnan's reassignment, owners for the engineering sources.
