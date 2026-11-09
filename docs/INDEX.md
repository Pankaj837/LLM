# Documentation index (5 Nov 2026)

Which document to open, and which ones are history. A document marked *history* is kept because it records how a
decision was reached or what was true at the time; **do not take its numbers as current**. Where a later document
supersedes a number, the older document carries a note that says so.

**Evidence labels used in the dated documents.** *Run* = executed by the author on this machine. *Read* = taken from a
file or message the author opened. *Owner-reported* = a team member's statement, no output seen. *Estimate* = computed
from a measured number with the assumption stated. *Not done* = no evidence yet.

## 1. Current: status, plans, decisions pending
| Document | What it is |
|---|---|
| `PROGRESS_REPORT.md` | Single source of truth for status, blockers, owners; newest update at the top |
| `PREPILOT_TASKS_STATUS_2026-11-04.md` | Raju's six pre-pilot tasks (4 Nov) and where each stands |
| `PILOT_1B_TOKEN_DIVISION_2026-11-05.md` | Raju's request: tokens per domain for the 1B pilot (20 domains, passes over held data, which are short) |
| `PILOT_PLAN_2026-11-04.md` | 1B to 3B-token pilot of the 1.5B model: compute, data feasibility, gates, success criteria, stop rules |
| `PILOT_RUN_CHECKLIST_2026-11-09.md` | Gokul's command checklist for the 1B pilot (rent, data, preflight, 30-min check, run, watch); mix file `configs/pilot_1b.json` |
| `DATA_PLAN_30B_2026-11-01.md` | From about 7B to 30B tokens; Raju's allocation of 2 Nov against the data held |
| `ASSIGNMENT_LEDGER.md` | Who was given what, when, and whether it was delivered |
| `TRAINING_PLAN_1p5B.md` | Recipe, budget and checklist for the 1.5B run (rev. 24 Oct; the GPU status lines are superseded by the pilot plan) |

## 2. Checks and reports (dated, final for their date)
| Document | What it shows |
|---|---|
| `A100_RUN_RESULT_2026-11-01.md` | The 100M run (step 8,250, validation loss 2.166), probes, per-domain loss |
| `CHECKPOINT_RELIABILITY_2026-11-04.md` | Task 4: what was wrong with checkpoints, the fix, tests, stop-and-resume demonstration |
| `EVAL_REPRODUCTION_2026-11-04.md` | Task 5: independent CPU re-run of Namit's evaluation, exact match |
| `DATA_CLEANING_REPORT_2026-11-05.md` | Task 1: inventory of the 20 bins with tokens, hashes, evidence and open defects; effect of Ravi's new bins on the 30B gap (not final) |
| `DATASET_CARDS_2026-11-05.md` | Task 2: cards for Engineering Reasoning, Domain 9 and Domain 10 |
| `RAVI_1P5B_BINS_CHECK_2026-11-04.md` | Check of Ravi's six 1.5B bins |
| `SWARAJ_V7_HANDOVER_CHECK_2026-11-04.md` | Check of Swaraj's final Domain 9 / 10 corpus (v7) |
| `SWARAJ_COLLECTOR_REVIEW_2026-11-05.md` | Static review of Swaraj's collection script; why re-running it cannot add volume |
| `AVINASH_DRIVE_CHECK_2026-11-05.md` | Avinash's Drive folder: a 24 Oct data lake, no bins; the truncated `entity_id` key |
| `FIRST_H100_SESSION_REPORT_2026-10-27.md`, `FIRST_RUN_ANALYSIS_2026-10-27.md` | First H100 session (owner-reported) and what it does and does not show |
| `share/` | Word versions of the status updates sent to the group |

## 3. Reference: how the code works and why
`ARCHITECTURE.md`, `COMPONENTS.md`, `DECISIONS.md`, `TEAM_INSIGHTS.md`, `REVIEW_FINDINGS.md`, `TESTING.md`,
`CPP_TOKENIZER_PARITY.md`, `REWARD_MODEL.md`, `HANDOFF_GPU.md` (how to run on a GPU), `HANDOFF_DATA_TEAM.md` (reward
model preference data), `CAPABILITIES.md` (the 15 capabilities), `A100_SESSION_RUNBOOK_2026-10-31.md` (the runbook used
for the A100 session).

## 4. History (superseded, kept for the record)
| Document | Superseded by |
|---|---|
| `DATA_STATUS_2026-10-20.md` | `DATA_PLAN_30B_2026-11-01.md` and `PROGRESS_REPORT.md` |
| `NEXT_PLAN_2026-10-28.md` | `PREPILOT_TASKS_STATUS_2026-11-04.md`, `PILOT_PLAN_2026-11-04.md` |
| `BIOLOGY_SMOKE_TEST.md`, `EARTH_ENVIRONMENT_SMOKE_TEST.md` | still valid as the record of the 6.3M smoke runs (September); not a quality signal for the current model |

## Rules for new documents
Name `TOPIC_YYYY-MM-DD.md`; start with the evidence legend; state what was *not* checked; when a number replaces an
older one, edit the older document with a one-line pointer and add the entry to `PROGRESS_REPORT.md` and `CHANGELOG.md`.
