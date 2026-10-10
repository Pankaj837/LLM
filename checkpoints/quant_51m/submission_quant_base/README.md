# Run A: 51.5M model trained from scratch on Quant Finance

This is the submission the training brief asked for: the `base` model, trained from random initialization on
the Quant Finance data for 5,000 steps.

**Short version:** use `ckpt_best.pt` (step 500). `ckpt_last.pt` (step 5,000) memorized the training data.

## Files

| File | What it is |
|---|---|
| `ckpt_best.pt` | Model at the lowest validation loss: step 500, val loss 3.740, perplexity 42.1. **The usable model.** 591 MB. |
| `ckpt_last.pt` | Model at the final step, 5,000: val loss 6.075, perplexity 435. Memorized training text; kept because it was requested. 591 MB. |
| `vocab.tok` | Tokenizer vocabulary, identical to the team's `final_vocab_51m.tok` (8,193 tokens, end-of-text id 8,192) |
| `train_config.json` | Every training and model setting used (see below) |
| `metrics.jsonl` | Training log: one line every 10 steps (500 lines). Validation every 250 steps. |
| `metrics_report.json` | Automatic check of `metrics.jsonl`. See the note on `lr_schedule_ok` below. |
| `eval.json` | Full evaluation of `ckpt_last.pt` on the validation set |
| `eval_best.json` | Full evaluation of `ckpt_best.pt` |
| `gen_check_step2000.json` | Text generated at step 2,000: 6 prompts x 3 samples, temperature 0.7, top-p 0.9 |
| `gen_check_final.json` | The same prompts at step 5,000 |
| `checkpoint_checks.json` | Each checkpoint reloaded in a fresh process and checked |
| `data_manifest.json` | Exactly which data was used: sources, train/validation split by file name, token counts, file hashes |
| `SHA256SUMS.txt` | SHA-256 of every file in this folder, written by the run |

## Settings (from `train_config.json`)

| Setting | Value |
|---|---|
| Model | 11 layers, hidden size 640, 8 query heads / 2 key-value heads (GQA), FFN 1,664, RoPE, 2,048 max positions. 51,665,920 parameters. |
| Steps | 5,000, starting from scratch (`resume: null`) |
| Batch | 32 sequences x 512 tokens = 16,384 tokens per step |
| Learning rate | peak 6e-4 after 200 warmup steps, cosine decay to 6e-5 at step 5,000 |
| Optimizer | AdamW, betas 0.9 / 0.95, weight decay 0.1, gradient clipping 1.0 |
| Precision | fp16 with gradient scaling, plus torch.compile. The T4 GPU has no native bf16. |
| Seed | 0 |

## Data

Quant Finance from the team dataset registry:
- arXiv quantitative-finance abstracts (927 files)
- Wikipedia finance articles (139)
- older public-domain economics book text, including *The Wealth of Nations* (25)
- finance code from GitHub, such as Microsoft QLib (92 after removing duplicates)

Stack Overflow is excluded, pending legal review.
Training: 1,133,567 tokens. Validation: 97,354 tokens. The split is by whole file, so no document is in both.

## How the run went

| Step | Training loss | Validation loss |
|---|---|---|
| 250 | 3.71 | 4.354 |
| **500** | 2.55 | **3.740** (lowest) |
| 1,000 | 0.58 | 4.493 |
| 2,000 | 0.13 | 5.530 |
| 5,000 | 0.04 | 6.075 |

The data is small for this model: one pass over it takes about 69 steps, so 5,000 steps is about 72 passes.
After step 500 the model learned the training text by heart and got worse on new text. In the final generation
check, 9 of 18 samples copy 20 or more words straight from the training data (the longest copy is 79 words).
See `../checks/` and `../RESULTS.md`.

## Reading the JSON files

- `metrics.jsonl`: `step`, `loss` (training), `lr`, `grad_norm`, `tokens_per_s` (average since the start),
  `loss_scale` (fp16 scaling factor), `max_mem_gb` (peak GPU memory). Every 250 steps it also has `val_loss` and
  `val_ppl`.
- `eval*.json`:
  - `val`: loss, perplexity and bits per byte on the whole validation set.
  - `val_by_source`: the same per source (arxiv, books, github, wikipedia).
  - `control_non_finance`: the same on unrelated text, to show how specialized the model is.
  - `unigram_baseline_loss` and `beats_unigram_by_nats`: comparison with a model that only knows word frequencies.
  - `samples`: a few greedy and sampled completions.
- `gen_check*.json`:
  - `summary`: `eos_stop_rate` (share of samples that ended themselves), `mean_distinct_2` (variety of word
    pairs, 1.0 = no repetition), `samples_with_looping_4gram_ge3` (samples stuck in a loop).
  - `samples`: every generated text.

## Known false alarm

`metrics_report.json` says `lr_schedule_ok: false`. The schedule is correct: every logged learning rate matches
the 5,000-step cosine exactly. The checker flagged step 20, where warmup passes through 6e-5 on its way up, and
steps 4,970–4,990, which are within 0.1% of the floor as they should be at the very end. The corrected checker
passes this file: see `../checks/metrics_report_runA_rechecked.json`.
