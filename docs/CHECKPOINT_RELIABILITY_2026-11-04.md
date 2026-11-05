# Checkpoint reliability and stop-and-resume (Raju's task 4), 4 Nov 2026

**Evidence labels.** *Run* = executed by me on CPU (Windows, Python 3.13, torch 2.12). Code: `src/llm/train.py`, `src/llm/checkpoint.py`; tests: `tests/test_checkpoint_reliability.py`; demo: `scripts/resume_demo.py`.

## What was wrong (found in the 100M A100 run, read from its outputs)
| Problem | Evidence |
|---|---|
| `ckpt_last.pt` was written only every `--save-interval` steps and carried no validation loss (`val_loss: null` at step 8,200) | `sample_outputs.json` of the run |
| The run was stopped by hand at step 8,380; the last save was step 8,200, so 180 steps (about 47M tokens) were lost | `metrics.jsonl` vs checkpoint step |
| After a resume the "best validation loss" restarted at infinity, so the first evaluation could overwrite `ckpt_best.pt` with a worse model | `train.py` before this change |
| No rolling backups: one damaged `ckpt_last.pt` meant restarting | `train.py` before this change |
| Resuming needed the exact file path of a checkpoint that opened cleanly | `train.py` before this change |

## What changed
1. `ckpt_last.pt` is written every `--save-interval` steps, **after every evaluation**, on the final step and on a graceful stop, and always carries the latest validation loss (plus the step it was measured at).
2. Writes were already atomic (temp file, rename). The previous `ckpt_last.pt` is now kept as `ckpt_step_<step>.pt` (rolling backups, `--keep-backups`, default 2).
3. `--resume auto` picks the newest checkpoint in `--out` that opens cleanly (a truncated or non-finite file is skipped with a warning; `ckpt_last.pt` is tried first, then the backups by step, and the search stops at the first good file so a 1.5B restart does not read three 18 GB files); with none present it starts fresh. Resuming restores the best validation loss, so `ckpt_best.pt` can no longer be overwritten by a worse model.
4. Ctrl-C, SIGTERM (SIGBREAK on Windows) or a file named `STOP` in `--out` finish the current step, write `ckpt_last.pt` and exit. All ranks agree on the stop in multi-GPU runs. A second signal quits at once. A hard kill loses at most `--save-interval` steps.
5. `verify_checkpoint`, `find_latest_checkpoint`, `prune_backups` in `llm/checkpoint.py`.

**Disk cost.** A checkpoint holds the weights and the AdamW state. For the 1.5B model (fp32 weights, two Adam moments) that is about 18 GB per file (estimate), so ckpt_last + 2 backups + ckpt_best is about 72 GB. Set `--keep-backups 1` on a small disk.

## Tests (run)
`tests/test_checkpoint_reliability.py`: 12 tests, all pass (the last two added on 5 Nov after a second review of the change: the newest-first search does not open older backups when `ckpt_last.pt` is good, and the signal handlers are restored even when training raises; the second was checked to fail when the `finally` is removed) (validation loss always recorded; save after each evaluation; backups pruned to N; STOP file and SIGTERM stop and resume reproduce the uninterrupted run exactly; corrupt latest checkpoint skipped; fresh start when nothing to resume; best checkpoint not overwritten after a resume). The existing training and checkpoint tests still pass.

## Stop-and-resume demonstration (run, `python scripts/resume_demo.py`)
Tiny model, synthetic data, 120 steps, checkpoint every 10 steps. Final weights compared bit for bit with the uninterrupted run.

| Scenario | What happens | Final step | Interruption | Final weights equal to uninterrupted run |
|---|---|---|---|---|
| A | uninterrupted run | 120 | - | reference |
| B | STOP file, then `--resume auto` | 120 | stopped and saved at step 48 | True |
| C | signal (CTRL_BREAK), then `--resume auto` | 120 | stopped and saved at step 54 | True |
| D | hard kill, then `--resume auto` | 120 | killed after step 67; last save was step 60, 7 steps redone | True |

## Not covered
- CUDA and NCCL paths are not tested (no GPU here); the two-process CPU (gloo) test of the trainer still passes, but the new stop agreement for several ranks (`all_reduce`) is not exercised by a test.
- Preemption by a scheduler that sends SIGKILL without warning is scenario D (at most `--save-interval` steps lost). For the pilot set `--save-interval` to about 15 minutes of training.
- Google Drive / remote copies of checkpoints are outside this code (copy `ckpt_last.pt` after each save).
