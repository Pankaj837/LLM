# Scientific evaluation set (frozen)

Owner: Namit (readiness plan §6, scientific evaluation and model capability validation).
This folder answers the question "what can the model actually do", which validation loss on its own does not. It is the downstream
benchmark that `docs/TRAINING_PLAN_1p5B.md` §7 lists as missing. The same frozen questions are scored on every checkpoint, so results
from different checkpoints can be compared directly.

## What is in it (3,619 questions in 19 domains, version 2.1)

| domain | basic | intermediate | advanced | sources |
|---|---|---|---|---|
| math | 150 | 150 | 100 | MMLU elementary / high-school / college mathematics |
| multistep | – | 200 (numeric) | – | GSM8K word problems, 4 worked examples shown first |
| physics | 159 | 157 | 102 | MMLU conceptual / high-school / college physics; hand-written units, formulas and insufficient-information items |
| chemistry | 3 | 152 | 100 | MMLU high-school / college chemistry; hand-written numeric |
| bio | – | 150 | 140 | MMLU high-school / college biology |
| astro | – | 150 | – | MMLU astronomy |
| earth | 150 | 124 | – | ARC-Easy / ARC-Challenge questions that mention an earth-science term (`EARTH_TERMS`: rocks, weather, climate, erosion, atmosphere, ...) |
| finance | – | 150 | 210 | MMLU high-school macroeconomics, econometrics, professional accounting (MMLU has no finance subject; these are the nearest) |
| scidisc | – | – | 200 | PubMedQA: a research abstract and a yes/no question about it; 100 yes and 100 no, so always answering "yes" scores 50% |
| sciqa | 300 | 300 | – | ARC-Easy, ARC-Challenge (the questions not in `earth`) |
| code | 3 | 103 | 100 | MMLU high-school / college computer science; hand-written Python/NumPy reading |
| electrical_eng | – | 144 | – | MMLU electrical engineering (the only engineering subject in MMLU) |
| engineering | 19 | 16 | 5 | hand-written mechanics, heat, fluids, circuits, design, generative design |
| simulation | 7 | 9 | 6 | hand-written CFD / FEA / PDE numerics |
| sciml | 3 | 8 | 5 | hand-written PINNs, Fourier neural operators, DeepONet |
| exp_val | 10 | 11 | 3 | hand-written uncertainty, statistics, design of experiments, validation |
| tech_comm | 8 | 7 | – | hand-written reports, RFC 2119 keywords, requirements, reporting results |
| sciprog | 4 | – | – | hand-written: smallest programs (print, add two numbers, NumPy sum); a starting point only |
| cad | 1 | – | – | hand-written: one CAD operation question (extrude); a starting point only |

`sciprog` and `cad` have only a handful of questions so far (a full code-writing test that runs the model's code is planned for
the 1.5B model; CAD has no public benchmark), so their scores show direction only. Engineering agent / tool use has no questions. SciQ was not used
for scidisc because its licence (CC BY-NC 3.0) forbids commercial use.

Files: `public/*.jsonl` (made by `scripts/build_eval_set.py` from pinned dataset revisions), `handwritten/*.jsonl` (written for
PRANA-G), `public/fewshot_gsm8k.json` and `handwritten/fewshot_science_numeric.json` (worked examples shown before numeric
questions; they are never scored). `MANIFEST.json` records the SHA-256 of every file: `scripts/eval_science.py` refuses to run
on a changed set unless `--allow-modified` is given. To change the set on purpose, edit it, bump the version and run
`python scripts/build_eval_set.py --manifest`.

Licences: ARC CC BY-SA 4.0; MMLU MIT; GSM8K MIT; PubMedQA MIT; hand-written items are internal. Every item carries its `source` and `licence`.
Public items were picked without randomness: the items whose source ids have the smallest SHA-1 hashes. MMLU rows that repeat an
answer option are left out (2 rows).

Version history:
* 2.1: added 4 small `sciprog` and 1 `cad` hand-written questions.
* 2.0: added bio, astro, finance (MMLU), earth (ARC, by keyword) and scidisc (PubMedQA); MMLU electrical engineering moved from
  `engineering` into its own domain `electrical_eng`, so `engineering` is now only the hand-written questions. ARC questions that
  match `EARTH_TERMS` moved from `sciqa` to `earth`, and `sciqa` was refilled from the other ARC questions. Every other file is
  byte-for-byte unchanged.
* 1.1: in 43 hand-written multiple-choice questions the right answer was clearly the longest option (58% of them overall), which a
  model can exploit without knowing anything. Options were reworded so that it is the longest in 22%, close to the 25% expected by
  chance. Meanings and `distractor_types` are unchanged, and edits were chosen from option lengths only, not from model answers.
  `tests/test_eval_science.py` checks the balance.
* 1.0: first frozen set.

## Item format

```json
{"id": "hw-eng-01", "domain": "engineering", "level": "basic", "type": "mcq", "skill": "formula_selection",
 "question": "A steel rod (E = 200 GPa) is 2 m long, ... Its elongation is",
 "choices": ["0.1 mm", "1 mm", "10 mm", "0.5 mm"], "answer": 1,
 "distractor_types": ["unit_error", "correct", "unit_error", "wrong_formula"],
 "source": "hand-written (Namit, PRANA-G)", "licence": "PRANA-G internal"}
```

`type` is `mcq` (answer = index into choices) or `numeric` (answer = number, `tolerance` = allowed relative error, `fewshot` =
which worked examples to show). Hand-written multiple-choice items label every wrong option (`wrong_formula`, `unit_error`,
`calculation_error`, `misconception`, `unsupported_claim`, `false_insufficient_info`), so a wrong pick shows *which* mistake the
model prefers. `skill` tags: `formula_selection`, `unit_conversion`, `multi_step`, `insufficient_info` (the right answer is that
the question cannot be answered), `unsupported_claim`, `concept`, `design`, `validation`, `code_reading`, `formatting`.

## How it is scored (`scripts/eval_science.py`)

* **Multiple choice**: each choice is appended to `Question: ...\nAnswer:` and the model's log-probability of the choice is
  computed. `acc_norm` (reported) picks the choice with the highest log-probability per character; `acc` (also stored) uses the
  plain sum. Nothing is generated, so a small base model that cannot follow instructions is still measured fairly. Chance level
  is 1 / number of options.
* **Numeric**: worked examples, then the question; the model continues greedily and the first number of its answer (for GSM8K,
  the number after "The answer is", strictly: a GSM8K answer without that line is `no_number`) must be within the tolerance. Chance level is 0. Wrong answers are classified as
  `no_number`, `unit_error (off by 10^k)`, `sign_error`, `repetition_loop` or `wrong_value`.
* Every accuracy is reported with a 95% Wilson interval next to the chance level; "above chance" means the lower end of the
  interval is above chance. With 15–25 items, the small hand-written domains can only show large differences.
* There is no sampling, and the script re-scores the first items and requires identical results (`repeat_check`).

## Leakage (`scripts/check_leakage.py`)

Questions are tokenized with the frozen tokenizer, and every 13-token window is searched for in the training `.bin` files.
A question counts as `leaked` when at least 20% of its windows (and at least 2, so one shared stock phrase such as "Which of the following statements is true?" does not count), or at least 10 of them, occur in training data (forum copies
often change punctuation, so only part of a copied question matches exactly). Pass the result to
`eval_science.py --exclude` so leaked questions are reported separately rather than scored. Public benchmarks (ARC, MMLU,
GSM8K) are widely copied on the web, so this check has to be run against every training file before a result is trusted.

## Running it

```bash
python scripts/check_leakage.py --vocab tokenizers/vocab32k_v2/vocab32k.tok --bins "bins/*.train.bin" --out results/leakage
python scripts/eval_science.py --ckpt session_main/run/ckpt_last.pt --vocab tokenizers/vocab32k_v2/vocab32k.tok \
    --exclude results/leakage.json --out results/eval_science_<step> --device auto
```

A 100M checkpoint runs on a laptop (Apple MPS or CPU); `--limit 5` gives a quick run over a few items per file.

## Limitations

* The hand-written domains (simulation, sciml, exp_val, tech_comm) are small. They show direction, not fine differences.
* Multiple-choice log-likelihood measures which answer the model finds most plausible, not whether it can produce or explain
  it. Free-form reasoning (derivations, explanations) is covered only by the numeric items and needs an instruction-tuned model
  to be measured properly.
* No chemistry or code items at the basic level apart from a few hand-written ones; no advanced items for sciqa or tech_comm.
