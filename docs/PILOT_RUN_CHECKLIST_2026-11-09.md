# 1B pilot run: command checklist for Gokul (9 Nov 2026)

**Evidence.** The commands are built from the repo's own command-line options (read: `scripts/gpu_session.py`, `src/llm/train.py`, `docs/PILOT_PLAN_2026-11-04.md`). The mix file `configs/pilot_1b.json` was parsed by `gpu_session.py preflight` on a CPU (run, 9 Nov). **Nothing here has been run on an H100 with the real bins.** Numbers that come from the 27 Oct smoke test (owner-reported): 1.5B model, micro-batch 4 fits (55 GB), micro-batch 8 does not, 25.4k tokens/s.

**What the pilot is:** 1.5B model, 1,000M tokens, 1,907 steps of 524,288 tokens (256 sequences of 2,048 = micro-batch 4 x accumulation 64), learning rate 3e-4, evaluation every 477 steps (= 250M tokens), about 11 hours. The mix is the 20-domain split in `PILOT_1B_TOKEN_DIVISION_2026-11-05.md`.

## A. Rent the machine
1 x H100 80 GB (not a MIG slice or A100 20 GB), on demand (spot can be reclaimed), at least 300 GB free disk (checkpoints are about 18 GB each, about 72 GB kept). Ubuntu image with CUDA and PyTorch. Raju asked for the amount before booking: about 13 GPU-hours at $2.98 = about $39, so $45.

## B. Set up (about 15 min)
```
nvidia-smi                      # must show one H100 with 80 GB
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"     # True
git clone https://github.com/Pankaj837/LLM.git && cd LLM
pip install -e ".[dev]"
```

## C. Data (put every file in `bins/`)
Names must be `<name>.train.bin` and `<name>.val.bin`. Domains whose train file is missing are left out and the others are renormalised, so a missing bin does not stop the run, but it changes the mix: **write down which are missing**.

| name | pilot tokens | where |
|---|---|---|
| cad, chem, code, engineering (agent trajectories), sciqa | 40M, 40M, 60M, 40M, 50M | Drive folder `1.5B Bins` (Ravi) |
| sciprog | 50M | same Drive folder `1.5B Bins`, **new de-duplicated files of 9 Nov** (verified by me: sha256, `check_bins` PASS, 0 duplicates) |
| math, physics, multistep, astro, finance, bio, earth, scidisc | 80, 70, 70, 30, 20, 40, 30, 40M | Prabanjan's and Gokul's earlier bins (the 34 files of the A100 run), renamed to these names |
| eng_reasoning | 80M | **missing** (Prabanjan): the run goes ahead without it unless Raju says wait |
| numsim (Domain 9), physai (Domain 10) | 70M, 70M | Swaraj's collected files; **not tokenized yet**: left out unless somebody delivers them |
| gen_eng, exp_val, tech_comm | 60, 40, 20M | no data exists: left out |

Also copy Ravi's `build_manifest.json` into `bins/` (preflight warns without it).

Check the files against the hashes (`docs/evidence/`); every line must say OK:
```
cd bins
grep -v sciprog ../docs/evidence/ravi_bins_sha256_2026-11-08.txt | sha256sum -c --ignore-missing
sha256sum -c ../docs/evidence/ravi_sciprog_sha256_2026-11-09.txt     # both hashes match the files I downloaded on 9 Nov; if one differs on your machine the download is incomplete, download again
cd ..
```

## D. Preflight (free, no training)
```
python scripts/gpu_session.py preflight --bins bins --config configs/pilot_1b.json --preset 1p5b --global-seqs 256 --min-free-gb 150 --out pilot
```
It must end with `PREFLIGHT: PASS`. It checks the GPU, the frozen tokenizer hash, every bin with `check_bins`, builds `bins/val_mix.bin` and checks the disk. Post the table in the group. A `WARN` for manifests is fine; any `FAIL` stops here.

## E. Gate 1: the 30-minute check (small steps, 60 steps)
```
python scripts/gpu_session.py all --bins bins --config configs/pilot_1b.json --preset 1p5b --global-seqs 64 --lr 3e-4 --bench-batches 2 4 --eval-interval 30 --save-interval 20 --max-steps-cap 60 --minutes 60 --out pilot_check
```
Then test stop and resume: start the same command again, wait for the first `[train]` step lines, and in a second terminal run `touch pilot_check/run/STOP`. It writes `ckpt_last.pt` and exits; run the same command again: it must say `resuming the interrupted run` and continue from that step.
Pass: loss finite and falling; tokens/s within 20% of 25.4k (at least 20k); resume continues the same loss curve. Post `pilot_check/session_report.md` to Raju. **Do not start Section F if anything here fails.**

## F. The pilot (1,907 steps, about 11 hours)
```
nohup python scripts/gpu_session.py all --bins bins --config configs/pilot_1b.json --preset 1p5b --global-seqs 256 --lr 3e-4 --bench-batches 2 4 --eval-interval 477 --save-interval 40 --max-steps-cap 1907 --minutes 800 --out pilot > pilot.log 2>&1 &
tail -f pilot.log
```
It benchmarks, measures speed on the real mix (about 10 min), prints `plan: 1907 steps x 524,288 tokens`, trains, evaluates every domain and writes `pilot/session_report.md`. If it stops or the machine restarts, run the same command again: it resumes from `pilot/run/ckpt_last.pt`. Stop on purpose with `touch pilot/run/STOP` (never `kill -9`).

## G. Watch and copy off
- Every hour: `tail -5 pilot/run/metrics.jsonl`. Stop and tell the group if: loss is NaN or Inf; validation loss rises at 3 evaluations in a row; tokens/s falls under 17k; any "Benchmark source" labels appear in sampled text.
- Every few hours copy `pilot/run/ckpt_last.pt`, `pilot/run/metrics.jsonl` and `pilot/session_report.md` off the machine (a rental machine can vanish).
- Success criteria: `PILOT_PLAN_2026-11-04.md` section 5 (proposal, Raju has not approved the numbers yet). The one to quote: validation loss at 1B tokens at most 2.35, the 100M model's value at the same point.

## H. Report to Raju
Post `pilot/session_report.md`, the five lines of `metrics.jsonl` around each evaluation, the missing-domain list from Section C, the hours used and the cost.
