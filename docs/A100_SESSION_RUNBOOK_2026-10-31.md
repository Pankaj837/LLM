# A100 session runbook (slot: Sat 31 Oct 10:00 -> Sun 1 Nov 08:00, 22 hours)

Written by Pankaj's assistant on 31 Oct (02:30). Everything marked *estimate* is a guess to be replaced by the measurement the session makes in its first minutes; everything marked *checked* was run on the actual files.

## 0. What is known and what is not
- **Confirmed on the machine (31 Oct, Prabanjan's preflight output): `1 x NVIDIA A100-SXM4-40GB MIG 3g.20gb, 21 GB, bf16 supported`.** It is a 3/7 slice of an A100, shared with other tenants (speed varied between 41k and 15k tokens/s). A 20 GB device **cannot run the 1.5B smoke test** (about 60 GB on the H100 run), so the 1.5B benchmark, smoke test and resume gates of Raju's readiness plan cannot be met on it; they need a full GPU.
- The platform note (Raju's screenshot): the file-copy option opens about **1.5 hours before the slot (about 08:30)**; copy the files into the GPU environment before GPU time starts; **download the produced artifacts before the session ends: the notebook workspace is deleted afterwards**.
- Confirmed 31 Oct: the machine **has internet** (the bins were downloaded from Drive with `dl.sh`); it had **no `git clone` checkout** (the repo came as a zip, so `--expect-tag` fails: leave it out). Still unknown: whether a notebook stays alive when the browser is closed.

## 1. The data (all `check_bins` PASS with the frozen tokenizer, sha256 10b3c5ae...; train tokens in millions)
| name in `bins/` | owner | train M | checked |
|---|---|---|---|
| math | Prabanjan | 995.7 | whole file; spread over the corpus, 0% duplicates in a 3,000-document sample |
| physics | Prabanjan | 637.2 | whole file; marker rule + dedup, 0% duplicates in sample (about 13% of documents have little physics: known) |
| gen_eng | Namit | 586.0 | whole file + his manifest/README/licences (BIGPATENT) |
| exp_val | Namit | 628.9 | whole file + manifest (peS2o, Common Pile, DOAB, FineWeb-Edu) |
| tech_comm | Namit | 534.5 | whole file + manifest (Stack Exchange, RFC, NASA NTRS, DOE OSTI) |
| engineering | Ravi | 813.9 | whole file (OpenHands agent trajectories, 13,933 documents of 58k tokens) |
| eng_reasoning | Pankaj rebuild (Prabanjan's corpus) | 805.0 | rebuilt 31 Oct on rc9 from `engineering_reasoning_corpus_old.txt` (3,114,429,124 bytes, separators throughout) with `"eot": true, "dedup": true`: 259,525 whole documents (mean 3,102 tokens), PASS, 0% duplicates in a 3,000-document sample, manifest clean at commit f541cad |
| code, sciqa, cad, sciprog | Ravi | 323 / 189 / 78 / 50 | whole files (28 Oct) |
| chem | Ravi | 128 | passes; still cut line by line (rebuild with `"start": "chempile-"` pending) |
| bio, earth, scidisc | Gokul | 45 / 73 / 1,002 | whole files, PASS (scidisc.train: 1,001,617,800 tokens, 139,297 documents, checked in full 31 Oct) |
| multistep, astro, finance | Prabanjan | 88 / 20 / 17 | whole files (24 Oct) |
Not in the mix: domains 9 and 10 (Swaraj): the 20 MB samples are mostly unrelated text (96% / 64% of documents contain none of the domain words) and about half the documents are exact duplicates. Prabanjan's `engineering_colab_training.zip` bin (737M tokens, `"blank": true` on the deduplicated file) is **not** used: it is cut at every paragraph.

## 2. Commands (paste one block at a time; send the output of each)
```bash
nvidia-smi
df -h .
curl -sI https://github.com | head -1
curl -sI https://drive.google.com | head -1
```
```bash
git clone https://github.com/Pankaj837/LLM.git && cd LLM && git checkout scale-run-rc11
python -m pip install -e ".[dev]"
mkdir -p bins            # copy every <name>.train.bin and <name>.val.bin from the shared folder into bins/
```
Free check, no GPU time used (it refuses wrong-vocabulary or damaged files):
```bash
python scripts/gpu_session.py preflight --bins bins --config configs/run_a100_22h.draft.json --expect-tag scale-run-rc11 --skip-smoke --smoke-preset 100m --min-free-gb 25 --out session
```
The session itself (the run gets 18 hours; the rest is benchmark, checks, evaluation and the time to download artifacts):
```bash
nohup python scripts/gpu_session.py all --bins bins --config configs/run_a100_22h.draft.json --expect-tag scale-run-rc11 --skip-smoke --smoke-preset 100m --bench-batches 4 --min-free-gb 25 --minutes 1020 --save-interval 200 --out session_main > session_main.log 2>&1 &
tail -f session_main.log
```
If it is interrupted, run the same command again: it resumes from `session_main/run/ckpt_last.pt` with the same step count. Use a **new `--out` folder for every fresh start** (an old `session.json` carries its step plan and would be resumed). `--minutes` is the training budget: the script stops training at 1.1 x that, so choose it so that start + 1.1 x minutes + 1.5 h is before the end of the slot. On a 20 GB slice add `--skip-smoke --smoke-preset 100m` (the 1.5B benchmark would only run out of memory) and `--bench-batches 4` (micro-batch 8 gave a hidden out-of-memory error during real training on the slice).

### 2b. If the machine has no GitHub / Google Drive (Prabanjan, 31 Oct 9:02 am: "I am manually uploading zips")
Upload `LLM_rc10_offline.zip` (the code at tag rc10 without `.git`; torch and numpy must already be installed, nothing else is needed for the session) and the bins (raw `.bin` files or zips of them). No `git clone`, no `pip install`:
```bash
unzip -q LLM_rc10_offline.zip -d LLM && cd LLM
python -c "import torch, numpy; print(torch.__version__, torch.cuda.is_available())"
mkdir -p bins            # put every <name>.train.bin and <name>.val.bin here
python scripts/gpu_session.py preflight --bins bins --config configs/run_a100_22h.draft.json --min-free-gb 25 --out session
nohup python scripts/gpu_session.py all --bins bins --config configs/run_a100_22h.draft.json --min-free-gb 25 --minutes 1080 --out session > session.log 2>&1 &
```
(`--expect-tag` is left out: without `.git` there is no tag to compare; the preflight still checks the tokenizer hash and every token file.) To prove the copied files are intact, compare `sha256sum bins/*.bin` with `SHA256SUMS.txt`.

## 3. Which model size (decided from the measurement, rule fixed now)
The session trains the 100M preset. A 300M model needs about 6B tokens (20 per parameter): at an *estimated* 40k tokens/s on a full A100 that is 42 hours, so it does not fit in 22 hours on any A100 variant; on a 20 GB slice the speed is lower still. The 1.5B model does not fit. The benchmark table in `session/session_report.md` gives the real tokens/s; if the 100M speed is above about 100k tokens/s, the 18-hour run sees more than 6B tokens (about one pass over everything we have); below 40k, about 2.6B.

## 4. Before the session ends (artifacts: the workspace is deleted)
Download `session/` (report, `session.json`, `metrics.jsonl`, `eval_domains.json`, `train.log`) and `session/run/ckpt_last.pt` plus `ckpt_best.pt`. Then run, and post, the real inputs/outputs Raju asked for:
```bash
python scripts/sample_outputs.py --ckpt session/run/ckpt_last.pt --vocab tokenizers/vocab32k_v2/vocab32k.tok --val math=bins/math.val.bin physics=bins/physics.val.bin gen_eng=bins/gen_eng.val.bin exp_val=bins/exp_val.val.bin tech_comm=bins/tech_comm.val.bin code=bins/code.val.bin chem=bins/chem.val.bin --out session/sample_outputs
```

## 5. What happened on 31 Oct (owner-reported unless marked; the log lines are Prabanjan's pasted output)
- 10:00 session started on Prabanjan's AIKosh account; bins downloaded with `dl.sh` (34 files, 12 GB, all pass `check_bins`; `eng_reasoning` was not in Drive, so 17 of the 18 planned domains train).
- The first preflight failed three checks, two of them too strict in `gpu_session.py` (GPU memory below 24 GB although the 1.5B smoke test was skipped; zero manifests): fixed in rc11 (tests added). The third (`--expect-tag`) cannot pass without a git clone. Prabanjan patched his copy with the equivalent `sed`, copied in one manifest (`math_physics_manifest.json`, covering math and physics only), and raised the calibration limit. His run is therefore **commit `46277fced2` (a local git init of the repo zip) plus local edits**, not an exact tag.
- A first start with the placeholder `--minutes 100` (947 steps) was cancelled; the real run started about 12:45 pm: **`plan: 9134 steps x 262,144 tokens = 2,394,423,296 tokens; measured 41,185 tokens/s (benchmark said 41,678)`**, micro-batch 4 x accumulation 32, peak memory 9.25 GB, `--save-interval 200`.
- Speed fell during the day: about 41k tokens/s at calibration, 36k at step 300, 32.8k at step 3,140 (probably other tenants on the same A100; not verified).
- Loss (his pasted lines): training loss 4.94 at step 250 (validation 5.375), 2.51 at step 3,140, 2.47 at step 4,250; validation loss 2.329 (step 4,250, perplexity 10.27), 2.308 (step 4,500, perplexity 10.06).
- Tokens so far = steps x 262,144 (step 3,140 = 823M). Prabanjan first reported 10^8 by multiplying steps by tokens/s; corrected in the group.
- Projection (estimate): 8.0 s per step leaves about 9.5 hours after step 4,850 (11:37 pm), so the plan would finish about 09:00 am, after the slot ends at 08:00. The script's own limit (1.1 x 1020 min after the start) would stop training at about 07:35 without the evaluation.

## 6. Finishing the session on time (stop at about 06:30, not at the script's limit)
The checkpoint is written atomically (`.tmp` then `os.replace`), so `ckpt_last.pt` is always complete and the run can be stopped at any moment; it is refreshed every 200 steps (about 27 minutes). The cosine schedule has `min_lr_ratio` 0.1: at step 8,000 of 9,134 the learning rate is 8.2e-5 against 6.0e-5 at the end, so stopping near step 8,000 loses little.
```bash
cd ~/LLM/LLM
mkdir -p ~/safe && cp session_main/run/ckpt_last.pt ~/safe/ckpt_$(date +%H%M).pt && ls -l ~/safe      # insurance copy; also test the platform's download with it
tail -n 1 session_main/run/metrics.jsonl; ls -l --time-style=full-iso session_main/run/ckpt_last.pt; date
pkill -f "gpu_session.py all"; sleep 3; pkill -f llm.train; sleep 10
pgrep -af "llm.train|gpu_session" || echo stopped
python scripts/gpu_session.py eval   --bins bins --config configs/run_a100_22h.draft.json --out session_main
python scripts/gpu_session.py report --bins bins --config configs/run_a100_22h.draft.json --out session_main
python scripts/sample_outputs.py --ckpt session_main/run/ckpt_last.pt --vocab tokenizers/vocab32k_v2/vocab32k.tok --val math=bins/math.val.bin physics=bins/physics.val.bin gen_eng=bins/gen_eng.val.bin exp_val=bins/exp_val.val.bin tech_comm=bins/tech_comm.val.bin code=bins/code.val.bin chem=bins/chem.val.bin --out session_main/sample_outputs
```
Then download `session_main/` (report, `session.json`, `metrics.jsonl`, `eval_domains.json`, `train.log`, `sample_outputs/`) and `session_main/run/ckpt_last.pt` and `ckpt_best.pt`, finishing before 08:00.
