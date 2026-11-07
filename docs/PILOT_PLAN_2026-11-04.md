# 1.5B pilot plan (Raju's task 6), draft of 4 Nov 2026

**Evidence labels.** *Measured* = from a run on our hardware. *Estimate* = computed from a measured number with the assumption stated. *Proposal* = a decision for Raju. Cost in rupees is not stated because the rental price is not known to me.

## 1. Purpose
Prove, on the real 1.5B model and the real data path, that training is stable, checkpoints survive interruptions and the evaluation is repeatable, before spending the 30B-token budget (Raju's staged plan: 1-3B pilot, 5B checkpoint evaluation, 10-15B only if the evaluation supports it). The pilot is **not** expected to beat the 100M model on benchmarks: 3B tokens for 1.5B parameters is 2 tokens per parameter, against about 22 for the 100M run, and compute-optimal training of a 1.5B model needs about 30B tokens.

## 2. Compute (estimate)
Measured: 1.5B model, sequence 2,048, 25,373 tokens/s on one H100, 60.8 GB peak, 4.5 minutes, loss 8.70 to 7.94, checkpoint and resume worked (27 Oct, small-batch smoke test, owner-reported run, numbers in `docs/FIRST_H100_SESSION_REPORT_2026-10-27.md`).

| Tokens | 1 x H100 hours | Notes |
|---|---|---|
| 1B | 10.95 | |
| 3B | 32.8 | |
| 5B | 54.7 | the 5B checkpoint evaluation in Raju's plan |

Assumptions: speed stays at the smoke-test value (a larger batch may be faster; shared or throttled GPUs were up to 2.7x slower in the A100 run, so keep a 20% margin). Multi-GPU speed-up is not measured; with 8 GPUs and 80% efficiency 3B tokens would take about 4 hours (estimate, untested). The 1.5B model needs a full 80 GB GPU; it does not fit the 20 GB A100 slice. Disk: one checkpoint with optimizer state is about 18 GB (estimate), so ckpt_last + 2 backups + ckpt_best is about 72 GB.

## 3. Data (measured token counts, 4 Nov)
- Per-domain token split for the 1B pilot (all 20 domains): `PILOT_1B_TOKEN_DIVISION_2026-11-05.md`.
- The pilot samples with Raju's 30B allocation weights. At 3B tokens each domain's share is 10% of its 30B allocation, e.g. math 0.24B, physics 0.21B, Domains 9 and 10 0.21B each.
- Passes over the data (pilot tokens / unique tokens held), limit 4: all held domains are within 4 passes except Numerical Simulation (39.9M tokens held, 5.3 passes), Physics+AI (13.1M, 16 passes) and Astronomy (4.5 passes). To stay within 4 passes the pilot needs about +13M and +39M tokens for Domains 9 and 10; or those two domains are down-weighted for the pilot.
- Ravi's six 1.5B bins (11.43B tokens) are the main supply but are **not yet usable**: sciqa still carries source labels, sciprog is mostly general code, chem, cad and engineering are not yet uploaded, and validation files contain text that also occurs in training (see `docs/RAVI_1P5B_BINS_CHECK_2026-11-04.md`). Gate 0 below blocks the pilot until they are fixed.

**8 Nov status of Gate 0:** sciqa labels fixed; sciprog improved; chem, cad and engineering uploaded; **chem rebuilt and verified on 8 Nov (the 6 Nov version contained the old sciqa data)**; validation-in-training overlap of the new bins not yet tested (training files not seen); Engineering Reasoning bin still missing. Cost (Spheron price page, 6 Nov): H100 80 GB $2.98 per hour on demand, $2.20 spot (can be reclaimed), 2.4 TB disk, 20-minute minimum; the 1B pilot is about 13 GPU-hours with the smoke test and evaluation, about $39 on demand.

## 4. Gates
| Gate | Check | Pass criterion | Owner |
|---|---|---|---|
| 0 Data | `check_bins` on every bin with the frozen tokenizer (`10b3c5ae...b51cdc`); no source labels; validation files free of training documents (Namit's `check_val_overlap.py`); benchmark leakage search (`check_leakage.py`) on all bins | all PASS; zero exact validation-in-training documents | Ravi, Namit |
| 1 Hardware smoke (about 30 min) | 200 steps on the rental GPU, real mix; stop-and-resume (`scripts/resume_demo.py`-style: STOP file, then `--resume auto`) | tokens/s within 20% of 25.4k; loss falling; resumed weights identical to an uninterrupted run | Gokul / Pankaj |
| 2 Pilot 1B tokens | training with evaluation every 250M tokens | criteria in section 5 | Gokul, Namit |
| 3 Pilot 3B tokens | continue if Gate 2 passed and the budget allows | criteria in section 5 | Raju decides |

## 5. Success criteria (proposal; thresholds are mine and need Raju's approval)
Training health (must all hold):
1. No NaN or Inf; no loss spike above 2x the running average that lasts more than 50 steps; global gradient norm below 5 after warm-up (100M run: 1.76 after warm-up, 2.60 at step 10).
2. Validation loss falls at every evaluation (the 100M run did).
3. Throughput stays above 80% of the Gate 1 measurement.
4. A checkpoint reloads and resumes with identical weights (demo above); `ckpt_last.pt` always carries a validation loss.

Model quality (assessed at 1B and 3B tokens on the frozen set, same scorer):
5. At 2.2B tokens (what the 100M saw) the pilot's validation loss is below the 100M's 2.166 on the same validation files, and lower on at least 80% of the domains (Namit's per-domain table is the reference). A larger model at equal tokens should win; if not, the mix, learning rate or data has a problem.
6. No regression against the 100M on Namit's frozen set (baseline: overall acc_norm 26.9% on the 3,544-item set v2.1, 95% range 25.5% to 28.4%; this number replaces the 24.5% quoted on 3 Nov, which was on 2,337 items with a different headline scorer). Do not expect gains on MMLU or GSM8K at 3B tokens.
7. No "Benchmark source:" or similar source labels in sampled text, and the repetition rate under sampling no worse than the 100M's (3/14 probes).

## 6. Stop rules
Stop and investigate if: non-finite loss; validation loss up for 3 evaluations in a row; throughput below 70% of Gate 1; more than 5% of sampled outputs are repetition loops under sampling; or any bin is found to contain benchmark questions.

## 7. Operations
- `--save-interval` about 15 minutes of training (an H100 at 25k tokens/s does about 22M tokens in 15 minutes), `--keep-backups 2`, `--resume auto`, run under a supervisor that restarts on exit; stop with the STOP file or SIGTERM, never kill -9 (see `docs/CHECKPOINT_RELIABILITY_2026-11-04.md`).
- Copy `ckpt_last.pt` and `metrics.jsonl` off the machine after each save (rental machines vanish; the 100M run's output was downloaded by hand).
- Evaluate with `scripts/eval_science.py` (branch `namit/science-eval`, not yet in `main`) on the frozen set and compare with `scripts/compare_eval.py` (see `docs/EVAL_REPRODUCTION_2026-11-04.md`). The 100M evaluation took 386 s on an Apple GPU; the 1.5B needs a GPU too (estimate: 10 to 20 minutes on an H100).

## 8. Open decisions for Raju
GPU type, number of GPUs and budget; pilot size (1B or 3B first); whether Domains 9 and 10 are down-weighted or waited for; Gokul's cover from 10 to 19 Nov.
