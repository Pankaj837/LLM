# Independent reproduction of Namit's evaluation (Raju's task 5), 5 Nov 2026

**Evidence labels.** *Run* = executed by me. Namit's code and results are from his branch `namit/science-eval` (commit `71027ca`, now also in `Pankaj837/LLM`).

## Result (run): reproduced exactly
I ran Namit's `scripts/eval_science.py` (on branch `namit/science-eval`; it is not in `main` yet, so the paths below exist only after checking out that branch) on the 100M checkpoint `ckpt_best.pt` (step 8,250) on a different machine and device (Windows, CPU, Python 3.13, torch 2.12) from his Mac run (Apple GPU, Python 3.14, torch 2.15), then compared the two result files with `scripts/compare_eval.py --reproduce`.

| Check | Result |
|---|---|
| Checkpoint SHA-256 | `54252dfe...867050`, equal to the one in his `results/inputs_sha256.txt` |
| Tokenizer SHA-256 | `10b3c5ae...b51cdc`, the frozen one |
| Question set | 3,544 scored items (version 2.1; 75 more excluded for overlap with training data), manifest matches |
| Overall acc_norm | 26.95% in both runs (difference 0.00 points); 95% range 25.5% to 28.4%, chance 24.8% |
| Items whose correctness changed | **0 of 3,544** |
| Multiple-choice picks changed | 0; log-probability difference median 0.0001, max 0.0007 |
| Every domain | identical (table in the comparison output) |
| Greedy repeat check inside the script | identical |
| Time | about 66 minutes on 8 CPU threads (386 s on his Apple GPU) |

Verdict of the script: **REPRODUCED** (limits: 0.5 points overall, 2% of items). Per-item and per-domain comparison: `python scripts/compare_eval.py --ref <his json> --new <my json> --reproduce`.

## What the baseline really is
The 3 Nov report (24.5% against 22.6% chance) used 2,337 questions and the plain `acc` score. His 5 Nov branch uses 3,544 questions and `acc_norm` (log-probability per character): 26.9% against 24.8% chance; plain `acc` on the same set is 24.3%. Any later checkpoint must be scored on the same set with the same scorer to be comparable. Domains with the whole 95% range above chance: astronomy 32.7%, earth 32.1%, electrical engineering 32.2%, sciqa 30.4% (and multistep 1.5% against 0). Everything else, including math, physics, chemistry, code and scidisc (50.0%, which is chance), is not above chance.

## Two problems found when running it on Windows (fixed by Namit in `eefe8cf` on 5 Nov, evening: diff read by me, same two changes as below plus a test; not run)
1. `eval_science.py` compares file names from `os.path.relpath`, which uses backslashes on Windows, with the forward-slash names in `evals/MANIFEST.json`; every file is reported as modified and the run stops. Fix (two lines): `os.path.relpath(p, evals_dir).replace(os.sep, "/")` at the places where `rel` and `it["_file"]` are set.
2. With Git's `core.autocrlf=true` (the Windows default) the question files are checked out with CRLF line endings, so their SHA-256 no longer matches the manifest. Fix: add `evals/** -text` to `.gitattributes`. I extracted the branch with LF endings and applied fix 1 in a scratch copy to run it.

## How to reproduce (one checkpoint, one command)
```
python scripts/eval_science.py --ckpt <ckpt_best.pt> --vocab tokenizers/vocab32k_v2/vocab32k.tok --exclude results/leakage_all.json --out my_run --device cpu
python scripts/compare_eval.py --ref results/eval_science_100m_step8250.json --new my_run.json --reproduce
```
Namit's `scripts/reproduce_analysis.sh` rebuilds the full analysis and report from the checkpoint and the bins. His `results/inputs_sha256.txt` lists every input hash.

## Not covered
- Only `ckpt_best.pt` was re-run; `ckpt_last.pt` (step 8,200) and the other analysis scripts (`analyze_checkpoint.py`, `check_leakage.py`, `check_val_overlap.py`) were not.
- Sampled decoding is not part of this score (the benchmark is greedy and likelihood-based), so repetition under sampling still needs the separate probe script.
