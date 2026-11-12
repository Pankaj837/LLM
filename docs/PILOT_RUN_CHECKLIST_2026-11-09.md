# 1B pilot run: command checklist for the run owner (written 9 Nov 2026 for Gokul; on 11 Nov, 9:24 pm, Om Ahire wrote in the group that he will run the 1B pilot - owner-reported, still to be confirmed with Raju together with the GPU approval)

**Evidence.** The commands are built from the repo's own command-line options (read: `scripts/gpu_session.py`, `src/llm/train.py`, `docs/PILOT_PLAN_2026-11-04.md`). The mix file `configs/pilot_1b.json` was parsed by `gpu_session.py preflight` on a CPU (run, 9 Nov). **Nothing here has been run on an H100 with the real bins.** Numbers that come from the 27 Oct smoke test (owner-reported): 1.5B model, micro-batch 4 fits (55 GB), micro-batch 8 does not, 25.4k tokens/s.

**What the pilot is:** 1.5B model, 1,000M tokens, 1,907 steps of 524,288 tokens (256 sequences of 2,048 = micro-batch 4 x accumulation 64), learning rate 3e-4, evaluation every 477 steps (= 250M tokens), about 11 hours. The mix is the 20-domain split in `PILOT_1B_TOKEN_DIVISION_2026-11-05.md`.

## A. Rent the machine
1 x H100 80 GB (not a MIG slice or A100 20 GB), on demand (spot can be reclaimed), at least 300 GB free disk (checkpoints are about 18 GB each, about 72 GB kept). Ubuntu image with CUDA and PyTorch. Raju asked for the amount before booking: about 13 GPU-hours at $2.98 = about $39, so $45 (9 Nov estimate, now out of date). **12 Nov update (group chat, owner-reported by Om Ahire, 7:06 pm):** Spheron on-demand H100 is $4.20/h (13 h = about $55; I read the Spheron page on 12 Nov: on-demand $4.20, spot $2.60, spot can be reclaimed so not for an 11-hour run); NeevCloud on-demand 1 x H100 is Rs 192.48/h (13 h = about Rs 2,500 + GST; the NeevCloud page returned HTTP 403 to me, so not checked). Om suggested NeevCloud if the account can be opened today, otherwise Spheron on-demand; H100 SXM 80 GB in both cases (PCIe may not reach the 20k tokens/s gate). Raju at 7:13 pm: "ok then book it and run". Billing starts when the machine is rented, so do Sections B-D first and copy `ckpt_best.pt`, `ckpt_last.pt` and the logs off the machine before releasing it.

## B. Set up (about 15 min)
```
nvidia-smi                      # must show one H100 with 80 GB
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"     # True
git clone https://github.com/Pankaj837/LLM.git && cd LLM
pip install -e ".[dev]"
```

## C. Data (put every file in `bins/`) - updated 11 Nov 2026 after the Drive audit
Names must be `<name>.train.bin` and `<name>.val.bin`. Domains whose train file is missing are left out and the others are renormalised, so a wrong name does not stop the run but silently changes the mix: **write down which are missing**. All 40 files are on the shared Drive folder `1B pilot training` (`https://drive.google.com/drive/folders/15rfwpGo3fhH8wvRYOhYQSaGJB7xEeLUX`). Token counts are the file headers read on 11 Nov (run: header + 4 samples per file; whole-file `check_bins` where marked).

| name the program reads | pilot plan (M) | Drive sub-folder / file | train tokens | note |
|---|---|---|---|---|
| cad, chem, code, engineering, sciprog, sciqa | 40, 40, 60, 40, 50, 50 | `ravi bins` | 840.6M, 545.7M, 1,517.4M, 2,986.7M, 1,587.0M, 1,880.4M | whole-file hashes in `docs/evidence/ravi_*`; `engineering` hash identical to 8 Nov (run 11 Nov) |
| bio, earth, scidisc | 40, 30, 40 | `Gokkul Bins/llm_dataset_bins` | 44.6M, 73.2M, 1,001.6M | header + samples |
| math, physics, multistep, gen_eng, exp_val, tech_comm | 80, 70, 70, 60, 40, 20 | `prabanjan/bins` | 995.7M, 637.2M, 88.2M, 586.0M, 628.9M, 534.5M | header + samples |
| **astro, finance** | 30, 20 | **`finance/astro updated`** (not the older files of the same name in `prabanjan/bins`) | 28.2M, 82.0M | whole-file `check_bins` PASS on 11 Nov; finance is about 94% numeric records, about half of astro is not astronomy |
| **eng_reasoning** | 80 | rebuilt 11 Nov, on the Drive in `eng_reasoning_upload` (https://drive.google.com/drive/folders/13c0qTtHjc0WlhaNYTM7QC3cS36eGNij8; sha256 re-checked after download). **Do not use the older `prabanjan/bins/engineering_reasoning.*.bin` (wrongly split: 13.5M pieces, median 26 tokens).** Check the sha256 below before use | 805.0M (train 804,961,418; val 4,185,109) | rebuilt from Prabanjan's original text with `tokenize_text.py --eot-domains --dedup`: 259,525 whole documents, `check_bins` PASS, identical to the 31 Oct build. sha256 train `7f8e21d4...0221a`, val `962934a6...50f6c` (`evidence/eng_reasoning_sha256_2026-11-11.txt`) |
| numsim | 70 | `namit bins` | 118.0M | whole-file hash and `check_bins` PASS on 11 Nov; 51% OpenWebMath (Raju approved 9 Nov, 10:15 pm) |
| physai | 70 | `domain10_v2_upload` (https://drive.google.com/drive/folders/1wa71m1WzhVjYOnuB923sY3NE-oPjVSn0), version 2, sha256 re-checked after download; **the old files in `pankaj bins` (version 1, 51.3M) are obsolete** | 68.8M | version 2 (11 Nov): 3,704 CC BY papers + 10,102 files from 1,043 MIT/Apache/BSD GitHub repositories; train 68,775,755 + val 1,330,502 = 70.1M tokens, `check_bins` PASS, sha256 train `70b40a2d...e47fb`, val `8ef52426...16b80` (`evidence/physai_v2_sha256_2026-11-11.txt`) |

## Final composition of the pilot (11 Nov, for Chandrika and the run owner)
All 20 domains have a train and a validation bin on the Drive. With the verified train-token counts every domain supplies its planned share, so **nothing is renormalised and the mix is the planned 1,000M**. Highest repeats (calculation, not run on a GPU): astro 1.07 passes, physai 1.02, bio 0.90, multistep 0.79, numsim 0.59, earth 0.41; all others below 0.25. If `eng_reasoning` is left out (wrong name), the other 19 domains are drawn about 1.09 times their planned share (920M of 1,000M planned present). The 1.5B full-run composition stays open until Biology (+255M), Earth (+152M), real finance text and on-topic astronomy arrive.

Also copy Ravi's `build_manifest.json` into `bins/` (preflight warns without it).

**Single source (11 Nov, night):** Ravi built one Drive folder `pilot_bins` with the 40 files + `SHA256SUMS.txt` (https://drive.google.com/drive/folders/1Tos80Itd1NOBxz7Wm2Hhe_6Uj4CW_J3h). His `SHA256SUMS.txt` is identical to `docs/evidence/pilot_bins_sha256_2026-11-11.txt` on all 40 lines (read: compared line by line, 12 Nov). Download all 40 files from it into `bins/`.

Check the files against the hashes; all 40 lines must say OK:
```
cd bins
sha256sum -c ../docs/evidence/pilot_bins_sha256_2026-11-11.txt
cd ..
```
(The older per-file hash lists `ravi_bins_sha256_2026-11-08.txt` and `ravi_sciprog_sha256_2026-11-09.txt` are superseded by this one.)

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
