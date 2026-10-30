# A100 session runbook (slot: Sat 31 Oct 10:00 -> Sun 1 Nov 08:00, 22 hours)

Written by Pankaj's assistant on 31 Oct (02:30). Everything marked *estimate* is a guess to be replaced by the measurement the session makes in its first minutes; everything marked *checked* was run on the actual files.

## 0. What is known and what is not
- Prabanjan reports **"GPU memory is 20GB, NVIDIA A100"** (his message, 30 Oct 7:37 pm). A full A100 has 40 or 80 GB, so this is probably a slice (MIG) of one A100. Unconfirmed: `nvidia-smi` output not yet seen. A 20 GB device **cannot run the 1.5B smoke test** (it needs about 60 GB on the H100 run); the session script skips it automatically with a note, and the 1.5B parameter-count / forward-backward checks in Raju's readiness plan must be done on CPU or another machine.
- The platform note (Raju's screenshot): the file-copy option opens about **1.5 hours before the slot (about 08:30)**; copy the files into the GPU environment before GPU time starts; **download the produced artifacts before the session ends: the notebook workspace is deleted afterwards**.
- Unknown: whether the environment can reach GitHub and Google Drive; free disk; whether a notebook stays alive when the browser or internet is off (ask the platform, or keep the tab open).

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
| bio, earth, scidisc | Gokul | 45 / 73 / 1,002 | bio, earth whole; scidisc validation whole, training 3 slices |
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
git clone https://github.com/Pankaj837/LLM.git && cd LLM && git checkout scale-run-rc10
python -m pip install -e ".[dev]"
mkdir -p bins            # copy every <name>.train.bin and <name>.val.bin from the shared folder into bins/
```
Free check, no GPU time used (it refuses wrong-vocabulary or damaged files):
```bash
python scripts/gpu_session.py preflight --bins bins --config configs/run_a100_22h.draft.json --expect-tag scale-run-rc10 --min-free-gb 25 --out session
```
The session itself (the run gets 18 hours; the rest is benchmark, checks, evaluation and the time to download artifacts):
```bash
nohup python scripts/gpu_session.py all --bins bins --config configs/run_a100_22h.draft.json --expect-tag scale-run-rc10 --min-free-gb 25 --minutes 1080 --out session > session.log 2>&1 &
tail -f session.log
```
If it is interrupted, run the same command again: it resumes from `session/run/ckpt_last.pt` with the same step count.

## 3. Which model size (decided from the measurement, rule fixed now)
The session trains the 100M preset. A 300M model needs about 6B tokens (20 per parameter): at an *estimated* 40k tokens/s on a full A100 that is 42 hours, so it does not fit in 22 hours on any A100 variant; on a 20 GB slice the speed is lower still. The 1.5B model does not fit. The benchmark table in `session/session_report.md` gives the real tokens/s; if the 100M speed is above about 100k tokens/s, the 18-hour run sees more than 6B tokens (about one pass over everything we have); below 40k, about 2.6B.

## 4. Before the session ends (artifacts: the workspace is deleted)
Download `session/` (report, `session.json`, `metrics.jsonl`, `eval_domains.json`, `train.log`) and `session/run/ckpt_last.pt` plus `ckpt_best.pt`. Then run, and post, the real inputs/outputs Raju asked for:
```bash
python scripts/sample_outputs.py --ckpt session/run/ckpt_last.pt --vocab tokenizers/vocab32k_v2/vocab32k.tok --val math=bins/math.val.bin physics=bins/physics.val.bin gen_eng=bins/gen_eng.val.bin exp_val=bins/exp_val.val.bin tech_comm=bins/tech_comm.val.bin code=bins/code.val.bin chem=bins/chem.val.bin --out session/sample_outputs
```
