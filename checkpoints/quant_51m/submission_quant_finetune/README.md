# Run B: team base model fine-tuned on Quant Finance

This run starts from the team's existing 51.5M base checkpoint at step 5,000 and continues training it on
Quant Finance for 1,500 more steps. It is the same starting checkpoint the Scientific Programming fine-tune used,
recorded there as the Biology + Earth run. The SHA-256 match is the only part of that history checked here.

**Short version:** `ckpt_best.pt` (step 5,700) is the best Quant model from either run, with validation
perplexity 30.9.

## Files

| File | What it is |
|---|---|
| `ckpt_best.pt` | Model at the lowest validation loss: step 5,700, val loss 3.431, perplexity 30.9. **The best Quant model.** 591 MB. |
| `ckpt_last.pt` | Model at the final step, 6,500: val loss 3.749, perplexity 42.5. Starting to overfit. 591 MB. |
| `vocab.tok` | Same shared vocabulary as Run A (`final_vocab_51m.tok`, 8,193 tokens) |
| `base_checkpoint.json` | Where the starting model came from, and its SHA-256 (`60b3cb85…e06a`), checked before training |
| `eval_pretrained.json` | The starting model evaluated on Quant **before** fine-tuning: val loss 6.228, perplexity 507 |
| `eval.json`, `eval_best.json` | Full evaluation of `ckpt_last.pt` and `ckpt_best.pt` |
| `train_config.json` | Every training and model setting used |
| `metrics.jsonl` | Training log: one line every 10 steps (150 lines, steps 5,010–6,500). Validation every 100 steps. |
| `metrics_report.json` | Automatic check of the log: no duplicate steps, learning rate correct, no invalid values |
| `gen_check_final.json` | Text generated at step 6,500: 6 prompts x 3 samples, temperature 0.7, top-p 0.9 |
| `checkpoint_checks.json` | Each checkpoint reloaded in a fresh process and checked |
| `data_manifest.json` | Same data and split as Run A |
| `SHA256SUMS.txt` | SHA-256 of every file in this folder, written by the run |

The file formats are the same as Run A's; see `../submission_quant_base/README.md` for what each field means.

## Settings that differ from Run A

| Setting | Value |
|---|---|
| Starting point | the team base checkpoint at step 5,000, with its optimizer state |
| Steps | 5,000 → 6,500 (step numbers continue from the base model, so `max_steps` is 6,500) |
| Batch | 16 sequences x 512 tokens = 8,192 tokens per step |
| Learning rate | constant 1e-4, no warmup, no decay |

`train_config.json` shows `resume` pointing to a file called `ckpt_last.pt`. That is the run's working copy of
the base checkpoint, which training started from. It is not a restart of a crashed run.

## How the run went

| Step | Validation loss | Perplexity |
|---|---|---|
| 5,000 (before fine-tuning) | 6.228 | 507 |
| 5,100 | 4.024 | 55.9 |
| 5,400 | 3.514 | 33.6 |
| **5,700** | **3.431** | **30.9** (lowest) |
| 6,000 | 3.494 | 32.9 |
| 6,500 | 3.749 | 42.5 |

Starting from a model already trained on other text gives a better Quant model than training from scratch: 0.31 lower validation loss than Run A's best. It still begins to overfit after about 700 steps.
At step 6,500, 4 of 18 generated samples get stuck repeating themselves, and only 22% end on their own.
It does not copy training text, though: the longest verbatim run in any sample is 7 words.
