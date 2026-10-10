# Quant Finance: 51.5M base model training run

This folder holds everything from training the team's 51.5M-parameter model on the Quant Finance dataset:
the checkpoints, the vocabulary, the training logs, evaluations and checks. The run took 49 minutes on
Kaggle (2 x Tesla T4).

**New here? Read `RESULTS.md` first.** It has the results and what they mean. This file explains what is
where and how to use it.

## Which checkpoint should I use?

| You want | Use |
|---|---|
| The best Quant model | `submission_quant_finetune/ckpt_best.pt` (step 5,700, validation perplexity 30.9) |
| The best model trained from scratch on Quant only | `submission_quant_base/ckpt_best.pt` (step 500, perplexity 42.1) |
| The exact file the training brief asked for | `submission_quant_base/ckpt_last.pt` (step 5,000). **Don't use it as a model.** It memorized the training data (perplexity 435). It's included because it was requested and as evidence of the overfitting. |
| To keep training from a checkpoint | any `ckpt_*.pt`; all four include the optimizer state |

All four checkpoints use the team's shared vocabulary `final_vocab_51m.tok` (8,193 tokens). A copy is in each
submission folder as `vocab.tok`.

## What is in this folder

| Path | What it is |
|---|---|
| `RESULTS.md` | The results report: settings, scores, overfitting, generation checks, recommendation |
| `submission_quant_base/` | **Run A**: the model trained from scratch on Quant (5,000 steps). The requested submission. |
| `submission_quant_finetune/` | **Run B**: the team's existing 51M base model fine-tuned on Quant (1,500 more steps) |
| `checks/` | Extra checks run after training: memorization test, corrected log check |
| `logs/` | Full console output of every stage (tests, benchmark, training, evaluation, generation) |
| `reproduce/` | The kit that ran this training on Kaggle, so the run can be repeated |
| `environment_and_plan.json` | The GPU, PyTorch version, speed benchmark, and the settings the run chose from it |
| `notes.json`, `notes_runB.json` | Anything that went wrong or changed during the run. Both are empty: nothing did. |
| `run_log.txt` | Timeline of the run, one line per stage, with minutes since start |
| `quant-51m-training.log` | The complete Kaggle notebook log (everything in `logs/`, interleaved) |

Each folder has its own README with the details.

## How to load a checkpoint

Use the team's LLM repo. These checkpoints were made with commit `5fb3edf` and load with the unmodified repo.

```bash
git clone https://github.com/Pankaj837/LLM.git
cd LLM
pip install -e .
```

```python
from llm.pipeline import LLMPipeline

pipe = LLMPipeline.from_pretrained(
    "submission_quant_finetune/ckpt_best.pt",
    "submission_quant_finetune/vocab.tok",
    device="cuda",          # or "cpu"
)
print(pipe.generate("The price of a European call option",
                    max_new_tokens=80, temperature=0.7, top_p=0.9))
```

`from_pretrained` refuses to load if the vocabulary file doesn't match the one the checkpoint was trained with,
so a wrong `vocab.tok` fails loudly instead of producing nonsense.

## How to check the files arrived intact

Each submission folder has a `SHA256SUMS.txt` written by the training run itself.

- Windows (PowerShell): `Get-FileHash submission_quant_base\ckpt_best.pt -Algorithm SHA256`
- Mac / Linux: `cd submission_quant_base && sha256sum -c SHA256SUMS.txt`

All 25 listed files were checked after download and all matched.

## Terms used in these files

- **Validation loss**: how surprised the model is by Quant text it never trained on. Lower is better.
- **Perplexity**: the same thing on a friendlier scale (e^loss). Roughly, the number of tokens the model is
  choosing between at each step. Lower is better.
- **Bits per byte**: loss measured per byte of raw text instead of per token. Use this to compare models with
  different vocabularies.
- **Step**: one update of the model's weights. Run A sees 16,384 tokens per step, Run B sees 8,192.
- **Overfitting / memorization**: the model learns its training text by heart instead of general patterns.
  Training loss keeps falling while validation loss rises.
