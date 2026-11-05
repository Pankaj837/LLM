# Raju's six pre-pilot tasks: status (4 Nov 2026; updated 5 Nov)

**Evidence labels.** *Run* = I ran it or opened the file. *Read* = taken from a team member's paste or PDF. *Estimate* = computed from a measured number with an assumption stated. *Not done* = no evidence yet.

| # | Task (Raju, 4 Nov 1:16 am) | Status | Evidence / what is left |
|---|---|---|---|
| 1 | Data cleaning report with hashes and token statistics | Draft written (5 Nov), not final: `DATA_CLEANING_REPORT_2026-11-05.md` | Done (run): the frozen tokenizer `vocab32k_v2` hash `10b3c5ae...b51cdc` is correct. The 17 bins of the 100M run have hashes and counts. Ravi's six new bins (11.43B tokens) are checked from his PDF (sums correct) and three validation files (`docs/RAVI_1P5B_BINS_CHECK_2026-11-04.md`). Left: chem, cad and engineering bins are not in Drive; sciqa still has source labels; sciprog is mostly general code; chem dedup unproven. A cleaning report cannot be called final until these are fixed. |
| 2 | Engineering Reasoning, Numerical Simulation and Physics+AI datasets (samples, method, token allocation) | Cards for Domains 9 and 10 done with measured numbers; Engineering Reasoning card from records only (file not found on 5 Nov): `DATASET_CARDS_2026-11-05.md` | Held: domain 9 about 40M tokens, domain 10 about 13M. At 4 passes that is about 0.16B and 0.05B against 2.1B each in Raju's table (5.2x and 16x over-pass at 30B). The Engineering Reasoning bin (verified 31 Oct) was said to exist only on Pankaj's laptop; a search of Downloads, Documents and Desktop on 5 Nov found no such file, so its path or a Drive upload is needed (source and licence are also not recorded). Swaraj handed over a cleaned v7 on 4 Nov (verified: 39.90M and 13.06M tokens, no emails or `# ` lines, 0 duplicates; see `SWARAJ_V7_HANDOVER_CHECK_2026-11-04.md`); the gap is volume. Ravi's claim to have collected 9 and 10 is unproven (his link showed none). |
| 3 | Evaluate the "existing 1.5B checkpoint" | Needs clarification | No 1.5B checkpoint exists. Only the 100M (`ckpt_best.pt`, step 8,250, validation loss 2.166). Question asked to Raju. Meanwhile Namit's evaluation of the 100M is the baseline (3 Nov report: 24.5% vs 22.6% chance on 2,337 questions; his 5 Nov branch: 26.9% vs 24.8% on 3,544 questions with the acc_norm scorer). |
| 4 | Checkpoint reliability and a stop-and-resume demo | **Done** (run, 5 Nov) | Validation loss always saved, rolling backups, `--resume auto`, graceful stop; 12 new tests; CPU demonstration reproduces an uninterrupted run bit for bit after a stop file, a signal and a hard kill: `CHECKPOINT_RELIABILITY_2026-11-04.md`. Not covered: multi-GPU resume, a 1.5B-size resume on a GPU. |
| 5 | Reproducible evaluation pipeline | **Reproduced; one gap left** | Namit's frozen set (now 3,544 questions in version 2.1, file hashes; baseline of the 100M checkpoint: overall acc_norm 26.9%, chance 24.8%) and scorer are the base. My reproduction and the comparison script: `docs/EVAL_REPRODUCTION_2026-11-04.md`. Result (run): my CPU re-run gave 26.95% in both runs, 0 of 3,544 answers changed. Namit fixed the two Windows bugs in `eefe8cf` (5 Nov evening; diff reviewed, not run). Left: one wrapper command is still to be written; sampled decoding to be fixed in the pipeline. Greedy decoding causes about 60% of repetition loops (Namit 56/102 vs 4, mine 8/14 vs 3/14), so sampled decoding must be fixed in the pipeline. |
| 6 | Pilot plan with GPU hours and success criteria | Draft below | See next sections. |

## Pilot compute (estimate)
Measured (run, 31 Oct smoke test): 1.5B model, 25,373 tokens/s on one H100, 60.8 GB. That gives about 10.95 H100-hours per 1B tokens, so about 32.8 for 3B and about 54.7 for 5B. This is small-batch smoke speed, so treat it as rough (real throughput may be higher with a larger batch). The 1.5B does not fit the 20 GB A100 slice.

## Pilot data feasibility (3B tokens at the 30B mix)
Possible with the bins held, except domain 9 (5.2x over-pass), domain 10 (16x) and astronomy (4.5x). Domains 9 and 10 should be limited to what is held, or the pilot mix re-weighted.

## Proposed success criteria (proposal, for Raju to approve)
1. Loss falls smoothly with no NaN and no spike above 2x the running average; gradient norm stays below 5.
2. The pilot checkpoint reloads and resumes with the same next-step loss (tolerance 1e-3).
3. At 2.2B tokens the pilot's validation loss is below the 100M's 2.166 and lower on at least 80% of domains; no regression on Namit's frozen set (26.9% acc_norm on 3,544 questions).
4. No "Benchmark source:" lines in generated text (no GSM8K or MMLU gains expected at 3B tokens).
5. Repetition rate under sampling no worse than the 100M.

## Open decisions for Raju
GPU and budget; licence decision on held CC BY-SA / public-domain records; Gokul cover for 10 to 19 Nov (endsems); who owns Multistep, Numerical Simulation, Physics+AI and Biology (still uncovered).
