# Quant Finance: 51.5M base model results

## Short version

- The full 5,000-step run is done, with the requested settings. All four requested files are in
  `submission_quant_base/`: `ckpt_last.pt`, `vocab.tok`, `train_config.json` and `metrics.jsonl`.
- Quant is too small for 5,000 steps on its own. It has 1.13M training tokens, and 5,000 steps at batch 32 x 512
  is 82M tokens, about 72 passes. Validation loss was lowest at step 500 and rose for the rest of the run.
  By step 5,000 the model was repeating training text word for word.
- **Use `submission_quant_base/ckpt_best.pt` (step 500)** as the from-scratch Quant model, not `ckpt_last.pt`.
- A second run fine-tuned the team's step-5,000 base checkpoint on Quant. Its `ckpt_best.pt` (step 5,700)
  is the best Quant model overall.
- Both runs use the shared vocabulary `final_vocab_51m.tok` (8,193 ids), so token ids match the other domains.

## What was trained

| | Run A: from scratch | Run B: fine-tune |
|---|---|---|
| Starting point | random init, `--preset base` | team base checkpoint, step 5,000 (SHA-256 `60b3cb85…e06a`) |
| Parameters | 51,665,920 | 51,665,920 |
| Steps | 0 → 5,000 | 5,000 → 6,500 |
| Batch x sequence length | 32 x 512 | 16 x 512 |
| Learning rate | 6e-4 peak, 200 warmup steps, cosine decay to 6e-5 at step 5,000 | 1e-4 constant |
| Precision | fp16 + torch.compile | fp16 + torch.compile |
| Throughput | ~34,800 tokens/s | ~25,400 tokens/s |
| Evaluation | every 250 steps | every 100 steps |

Data: Quant Finance (arXiv abstracts, Wikipedia articles, older public-domain economics books including The Wealth of Nations, and finance code from GitHub such as QLib). Stack Overflow is excluded
pending legal review. 1,133,567 training tokens and 97,354 validation tokens, split by file, tokenized with
`final_vocab_51m.tok`.

Hardware: Kaggle, 2 x Tesla T4, PyTorch 2.11 (CUDA 12.8). The two runs trained at the same time, one per GPU.
The whole job took 49 minutes. The repo test suite ran on the GPU first: 278 passed, 8 skipped (only tests
that need files not in the kit).

## Results on the Quant validation set

| Checkpoint | Step | Val loss | Perplexity | Bits/byte | Non-finance control loss |
|---|---|---|---|---|---|
| Team base model, before any Quant training | 5,000 | 6.228 | 507 | 2.656 | 7.076 |
| Run A best | 500 | 3.740 | 42.1 | 1.595 | 5.801 |
| Run A last | 5,000 | 6.075 | 435 | 2.591 | 9.475 |
| **Run B best** | **5,700** | **3.431** | **30.9** | **1.463** | 5.423 |
| Run B last | 6,500 | 3.749 | 42.5 | 1.599 | 6.175 |

The unigram baseline loss is 6.975. The earlier 6.3M Quant model scored 1.754 bits/byte. Bits per byte is the
fair comparison across vocabularies; loss and perplexity are not comparable with that model or with other
domains' runs.

Run A validation loss by step: 4.354 (250), **3.740 (500)**, 4.021 (750), 4.493 (1,000), 5.530 (2,000),
6.066 (4,000), 6.075 (5,000). Training loss was 0.58 at step 1,000 and 0.04 at step 5,000.

Run B was lowest at step 5,700 and rose steadily after that.

## Status of each requested change

| Request | Result |
|---|---|
| 5,000–10,000 steps | 5,000 steps |
| Scheduler total steps match the run | LR follows the 5,000-step cosine exactly (every logged value checked). It is 4.3e-4 at step 2,000 and reaches 6e-5 only at step 5,000. |
| `resume: null`, larger seq_len | `resume` is null; seq_len 512 |
| bf16 + torch.compile, above 2,000 tokens/s | torch.compile on; ~34,800 tokens/s. fp16 was used instead of bf16, because the T4 has no native bf16. In a 60-step benchmark, fp16 gave the same losses as fp32 to 3 decimal places. |
| No duplicate log lines | 500 lines, steps 10–5,000, no repeats. The trainer now drops log lines past a resume point; re-logging steps after a resume is the likely source of the duplicates seen in earlier runs. |
| Generation check at step 2,000 (T 0.7, top-p 0.9) | Done, see below |

`metrics_report.json` in Run A says `lr_schedule_ok: false`. That is a false alarm in the checking script.
It flagged step 20, where warmup passes through 6e-5 on the way up, and steps 4,970–4,990, which sit within
0.1% of the floor as a cosine schedule should at the very end. With the corrected script
(`checks/clean_metrics.py`), the same file passes (`checks/metrics_report_runA_rechecked.json`).

## Generation and memorization

18 samples per checkpoint, 120 new tokens each, temperature 0.7, top-p 0.9. The "verbatim copy" column is the
longest run of generated words (prompt excluded) that appears word for word in the training text.

| Checkpoint | Stops at end-of-text | Distinct-2 | Looping samples | Longest verbatim copy | Samples copying 20+ words |
|---|---|---|---|---|---|
| Run A step 2,000 | 56% | 0.965 | 0 of 18 | 12 words | 0 of 18 |
| Run A step 5,000 (last) | 50% | 0.978 | 0 of 18 | 79 words | 9 of 18 |
| Run B step 6,500 (last) | 22% | 0.838 | 4 of 18 | 7 words | 0 of 18 |

Run A's last checkpoint reads fluently because half its samples are quoting training abstracts, some of them
for 50–79 words. This matches its validation loss. At step 2,000 the text is a mix of finance phrases from
different sources, and it is not copying.

The two `ckpt_best.pt` files were not generation-checked. Their validation scores are the evidence for them.

## Recommendation

For Quant on its own, about 500 steps at batch 32 is the useful length. Longer training only makes sense once
Quant is part of the combined corpus, or after more Quant data is collected. Fine-tuning the shared base model
(Run B) gave the best result: 0.31 nats lower validation loss than the best from-scratch checkpoint.

## Files

`submission_quant_base/` (Run A): `ckpt_last.pt`, `ckpt_best.pt`, `vocab.tok`, `train_config.json`,
`metrics.jsonl`, `metrics_report.json`, `eval.json`, `eval_best.json`, `gen_check_step2000.json`,
`gen_check_final.json`, `checkpoint_checks.json`, `data_manifest.json`, `SHA256SUMS.txt`.

`submission_quant_finetune/` (Run B): the same, plus `eval_pretrained.json` and `base_checkpoint.json`, and
without the step-2,000 check.

Checkpoint SHA-256 (from the run's own `SHA256SUMS.txt`):

| File | SHA-256 |
|---|---|
| `submission_quant_base/ckpt_best.pt` | `e33746a9dee6d4d2d6b6a9e72149f00921413491b893cc6e36e128b6f73dc94c` |
| `submission_quant_base/ckpt_last.pt` | `550eea596b0985ed3a691582ef4387b47fd42019265ed6486921ede1614e9b78` |
| `submission_quant_finetune/ckpt_best.pt` | `d354fa041ede39059fdec5db3c3df7708274fb0d68660f58bb9b3e061cc01456` |
| `submission_quant_finetune/ckpt_last.pt` | `74d1996b130817e539fc4c80001f0086e7dacc8fefdf2406fe068085d1296993` |

`vocab.tok` SHA-256: `7a56c090cfbfc8913c21cdf412ed0aae1c806ab6fd6fa09c6376ab29566fd362`
(identical to `final_vocab_51m.tok`).

Every checkpoint was reloaded in a fresh process after training. All four pass the following:
- strict load with the vocabulary hash
- 51,665,920 parameters
- no torch.compile key prefix
- all weights finite
- optimizer state included

## Not covered

- Generation checks on the two `ckpt_best.pt` files.
- C++ / Python tokenizer parity (compiled binary, 32k/64k vocabulary). This is a separate task.
- 10,000 steps. Given the overfitting at 5,000, it was not run.
