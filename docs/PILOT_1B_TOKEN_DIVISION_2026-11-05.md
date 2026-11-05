# Token division for the 1B pilot run, all 20 domains (Raju's request, answered 5 Nov 2026)

**Rule.** Every domain gets the same share of the pilot as it has of Raju's 30B allocation (`token distribution.pdf`, 2 Nov): pilot tokens = allocation / 30 x 1B. Every result is a multiple of 10M. Raju's own 4 Nov ask for Domains 9 and 10 (60 to 80M each) matches this rule: it gives 70M each.

**Evidence labels.** Allocation = read (Raju's PDF). Held = verified unique train tokens: from `check_bins` for the bins we hold; Ravi's six bins (code, sciqa, sciprog, chem, cad, agent) are his reported numbers, and 2 of 6 training files were not seen by me (see `RAVI_1P5B_BINS_CHECK_2026-11-04.md`). Passes = pilot tokens / held tokens (computed by me).

| Domain | Owner | 30B allocation (B) | **1B pilot (M tokens)** | Held (M) | Passes in the pilot | Short? |
|---|---|---|---|---|---|---|
| Math | Prabanjan | 2.4 | **80** | 996 | 0.08 | no |
| Physics | Prabanjan | 2.1 | **70** | 637 | 0.11 | no |
| Scientific discovery | Gokul | 1.2 | **40** | 1,002 | 0.04 | no |
| Engineering agent trajectories | Ravi | 1.2 | **40** | 2,987 | 0.01 | no |
| Engineering reasoning | Prabanjan | 2.4 | **80** | 805 | 0.10 | no (bin not on Pankaj's laptop) |
| Generative engineering (11) | Namit | 1.8 | **60** | 586 | 0.10 | no |
| Experimental validation (12) | Namit | 1.2 | **40** | 629 | 0.06 | no |
| Technical communication (14) | Namit | 0.6 | **20** | 535 | 0.04 | no |
| Code | Ravi | 1.8 | **60** | 1,517 | 0.04 | no |
| Scientific Q&A | Ravi | 1.5 | **50** | 1,901 | 0.03 | no |
| Chemistry | Ravi | 1.2 | **40** | 2,447 | 0.02 | no |
| Earth | Gokul | 0.9 | **30** | 73 | 0.41 | no |
| CAD | Ravi | 1.2 | **40** | 841 | 0.05 | no |
| Multi-step reasoning | Prabanjan | 2.1 | **70** | 88 | 0.80 | no (generated numbers, low information) |
| Biology | Gokul | 1.2 | **40** | 45 | 0.89 | no (nearly all of it) |
| Scientific programming | Ravi | 1.5 | **50** | 1,672 | 0.03 | no |
| Finance | Prabanjan | 0.6 | **20** | 17 | 1.18 | **yes, +3M** |
| Astronomy | Prabanjan | 0.9 | **30** | 20 | 1.50 | **yes, +10M** |
| Numerical simulation (Domain 9) | open | 2.1 | **70** | 40 | 1.75 | **yes, +30M** |
| Physics + AI (Domain 10) | open | 2.1 | **70** | 13 | 5.38 | **yes, +57M** |
| **Total** | | **30.0** | **1,000** | 16,851 | | |

## What it says
- For 16 of 20 domains the data we hold already covers the 1B pilot with at most one pass. Biology (0.89 passes), multi-step (0.80) and Earth (0.41) use most of what exists.
- Four domains need more than one pass at 1B: Finance (1.2x), Astronomy (1.5x), Domain 9 (1.75x) and Domain 10 (5.4x). The first three are within the 4-pass limit; Domain 10 is the only one beyond it. The smallest fix is +57M tokens for Domain 10 and +30M for Domain 9, or accept the repeats for the pilot (the pilot checks stability and throughput, not capability).
- The pilot does not need the 20 Oct to 4 Nov gaps closed: it can start with the bins as they are, if Raju accepts the repeats above. At the 3B pilot the same table scales by 3 (Domain 10 = 16 passes).
- The Engineering Reasoning bin (805M) must be on the training machine: it is not on Pankaj's laptop (Prabanjan to share). Ravi's chem, CAD and engineering-agent training bins are also needed there.
- Compute: 1B tokens is about 10.95 H100-hours (estimate from the 25,373 tokens/s smoke test, owner-reported); GPU and budget are Raju's decision (`PILOT_PLAN_2026-11-04.md`).
