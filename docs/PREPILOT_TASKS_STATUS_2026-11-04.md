# Raju's six pre-pilot tasks: status on 4 Nov 2026

**Evidence labels.** *Run* = I ran it or opened the file. *Read* = taken from a team member's paste or PDF. *Estimate* = computed from a measured number with an assumption stated. *Not done* = no evidence yet.

| # | Task (Raju, 4 Nov 1:16 am) | Status | Evidence / what is left |
|---|---|---|---|
| 1 | Data cleaning report with hashes and token statistics | Partly done | Done (run): the frozen tokenizer `vocab32k_v2` hash `10b3c5ae...b51cdc` is correct. The 17 bins of the 100M run have hashes and counts. Ravi's six new bins (11.43B tokens) are checked from his PDF (sums correct) and three validation files (`docs/RAVI_1P5B_BINS_CHECK_2026-11-04.md`). Left: chem, cad and engineering bins are not in Drive; sciqa still has source labels; sciprog is mostly general code; chem dedup unproven. A cleaning report cannot be called final until these are fixed. |
| 2 | Engineering Reasoning, Numerical Simulation and Physics+AI datasets (samples, method, token allocation) | Not done | Held: domain 9 about 40M tokens, domain 10 about 13M. At 4 passes that is about 0.16B and 0.05B against 2.1B each in Raju's table (5.2x and 16x over-pass at 30B). Engineering Reasoning bin exists only on Pankaj's laptop and must be uploaded to Drive. Swaraj handed over a cleaned v7 on 4 Nov (verified: 39.90M and 13.06M tokens, no emails or `# ` lines, 0 duplicates; see `SWARAJ_V7_HANDOVER_CHECK_2026-11-04.md`); the gap is volume. Ravi's claim to have collected 9 and 10 is unproven (his link showed none). |
| 3 | Evaluate the "existing 1.5B checkpoint" | Needs clarification | No 1.5B checkpoint exists. Only the 100M (`ckpt_best.pt`, step 8,250, validation loss 2.166). Question asked to Raju. Meanwhile Namit's evaluation of the 100M is the baseline (overall 24.5% vs 22.6% chance; ARC-Easy 37.7% is the only real signal). |
| 4 | Checkpoint reliability and a stop-and-resume demo | Not done | Known problem (run): the last checkpoint (step 8,200) has no validation loss. Plan: save on the last step and at every eval step, keep rolling backups, and demonstrate stop and resume on CPU with a tiny model. |
| 5 | Reproducible evaluation pipeline | Base exists | Namit's frozen set (2,337 scored questions, file hashes) and scorer are the base. Left: one command that reproduces his table from a checkpoint, plus decoding settings. Greedy decoding causes about 60% of repetition loops (Namit 56/102 vs 4, mine 8/14 vs 3/14), so sampled decoding must be fixed in the pipeline. |
| 6 | Pilot plan with GPU hours and success criteria | Draft below | See next sections. |

## Pilot compute (estimate)
Measured (run, 31 Oct smoke test): 1.5B model, 25,373 tokens/s on one H100, 60.8 GB. That gives about 10.95 H100-hours per 1B tokens, so about 32.8 for 3B and about 54.7 for 5B. This is small-batch smoke speed, so treat it as rough (real throughput may be higher with a larger batch). The 1.5B does not fit the 20 GB A100 slice.

## Pilot data feasibility (3B tokens at the 30B mix)
Possible with the bins held, except domain 9 (5.2x over-pass), domain 10 (16x) and astronomy (4.5x). Domains 9 and 10 should be limited to what is held, or the pilot mix re-weighted.

## Proposed success criteria (proposal, for Raju to approve)
1. Loss falls smoothly with no NaN and no spike above 2x the running average; gradient norm stays below 5.
2. The pilot checkpoint reloads and resumes with the same next-step loss (tolerance 1e-3).
3. On Namit's frozen set the pilot beats the 100M baseline (24.5%) with the 95% range above chance on at least ARC-Easy, math and one engineering domain.
4. GSM8K above 5% (100M: 1.5%), with no "Benchmark source:" lines in generated text.
5. Repetition rate under sampling no worse than the 100M.

## Open decisions for Raju
GPU and budget; licence decision on held CC BY-SA / public-domain records; Gokul cover for 10 to 19 Nov (endsems); who owns Multistep, Numerical Simulation, Physics+AI and Biology (still uncovered).
