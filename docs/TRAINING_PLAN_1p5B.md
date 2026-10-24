# Training plan — scale-up from the 51M models (rev. 24 Oct 2026)

Owner of this plan: Pankaj. Numbers marked *est.* are my calculations and must be replaced by Om/Gokul's measured
benchmark. **Nothing below has been run on a GPU yet.** Decisions that are not mine are listed in §9.

## 1. What is ready in the repo (all tested on CPU)
| Item | Status |
|---|---|
| Presets `100m` (100.1M), `300m` (303.4M), `1p5b` (1.509B), `2b` (2.012B): `1p5b`/`2b` have 32 layers, d=2048, 16 query / 4 KV heads, FFN 5632 / 8192, 4K context, tied embeddings, vocab 32000 | parameter counts checked against an exact formula that reproduces a real built model |
| Multi-GPU: `torchrun --nproc_per_node=N -m llm.train ...` (DistributedDataParallel; `--batch-size` is per rank; rank 0 alone logs, evaluates and writes checkpoints; 60-minute collective timeout) | a 2-process CPU run reproduces one process with accumulation 2 (weights match); **never run on CUDA/NCCL** |
| Weighted domain mix: `--train-mix math.bin=0.2 code.bin=0.12 ...` (weight = share of training tokens; prints the epochs each file will be repeated and warns above 4) | tested: sampling frequencies, determinism, bit-exact resume |
| Atomic checkpoint writes (a crash or pre-emption never leaves a corrupt `ckpt_last.pt`) | tested |
| bf16 autocast, AdamW, cosine schedule, grad clip, bit-exact resume, per-step metrics | tested |
| `scripts/tokenize_parquet.py` — Parquet (v4.0 files included) → TOK1 bins; big files split by row group across all cores; repairs unencodable text; unique job names | tested |
| `scripts/tokenize_text.py` — `.txt`/`.txt.gz` (line or blank-line documents; Windows `\r\n` handled; invalid bytes repaired; streams gzip) → TOK1 bins. Needed for math, Multi-step, Astronomy, Finance, chemistry text | tested; measured 1.3M tokens/s on 8 cores |
| `scripts/dedup_text.py`, `scripts/dedup_parquet.py` — exact de-duplication across files; `--expect-files` catches incomplete downloads | tested; run by Prabanjan (math) and Gokul (3 domains) |
| `scripts/merge_bins.py` — concatenates a domain's per-job bins into one TOK1 file (needed for `--train-mix` and per-domain evaluation; refuses mixed vocabularies) | tested |
| `scripts/check_bins.py` — header/size/ids/EOS/vocabulary check of every token file | tested; PASS on the smoke data |
| `scripts/train_vocab.py`, `scripts/verify_vocab.py`, `tokenizers/vocab32k_v2/` — the 32K vocabulary (candidate) and its verification (round-trip, C++ parity 3003/3003 lines) | tested |
| `scripts/eval_domains.py` — validation loss per domain | tested |
| Real layer shapes of `1p5b` / `2b` (d=2048, 16/4 heads, head_dim 128, FFN 5632/8192) | forward + backward + full 4K context verified on CPU with 1–2 layers instead of 32 (`tests/test_scale_shapes.py`); the full-depth model has not been instantiated yet |
| `scripts/pubmed_to_parquet.py` — PubMed XML → Parquet (for Medical) | tested, also on a real 48 MB slice from the Drive zip |

**Not in the repo:** FSDP/ZeRO-3, FP8 (Transformer Engine), FlashAttention-3, `torch.compile`, activation
checkpointing (Om's items). They are speed-ups, not blockers: with fp32 master weights and Adam the 1.51B model needs
~24 GB of state, so it fits on one 80 GB H100 with DDP.

## 2. Compute: what is possible *(est.)*
Cost ≈ 10.7 GFLOP per token for 1.51B (6 × 1.509B plus attention at seq 2048). At 35% of an H100's bf16 peak with this
plain PyTorch code: ≈ 3×10⁴ tokens/s per GPU ≈ **2.8B tokens per GPU-day**. 30B tokens (≈ 20 tokens/parameter) ≈ 11
GPU-days ≈ 257 H100-hours. *(An earlier version of this plan said 8 GPU-days; that was wrong.)*

| GPUs | tokens in 3 days | tokens/parameter (1.51B) |
|---|---|---|
| 1 | ~8B | 5 |
| 4 | ~33B | 22 |
| 8 | ~65B | 43 |

### Against the budget we actually have
Gokul (21 Oct): an H100 costs ₹330 per hour; credit about ₹1,000 = **about 3 H100-hours**. Same assumptions:

| Model | Tokens at 20/param | H100-hours | Cost at ₹330/h | Tokens in ₹1,000 |
|---|---|---|---|---|
| ~100M | 2.0B | 1.3 | ₹440 | 4.6B |
| ~300M | 6.1B | 11.8 | ₹3,900 | 1.6B |
| ~650M | 13.1B | 51 | ₹16,700 | 0.8B |
| **1.51B** | **30.2B** | **257** | **₹85,000** | **0.35B** |

1.51B on 5B tokens (3 tokens/parameter, an under-trained baseline) ≈ 43 h ≈ ₹14,000; on 10B ≈ 86 h ≈ ₹28,000.
**With ₹1,000: the 1.51B smoke test (~40 min ≈ ₹220) and one ~100M compute-optimal run (~1.3 h ≈ ₹440); not the
1.51B training.**

**3B:** the configurations in `llm update.pdf` stop at 2.08B (`2b` preset = 2.012B with 4 KV heads). A true 3B needs a
new configuration (d=2560 gives ~2.6B) and ~60B tokens at 20 tokens/parameter, against ~26–28B *estimated* today.

## 3. Data: what exists *(estimates until every owner's recount and the manifests are in)*
| Source | Tokens (est.) | Notes |
|---|---|---|
| Math (Prabanjan, 4,924,517 docs after his dedup) | ~9.5B | `tokenize_text.py`, line documents; his cleaned file must be uploaded |
| Biology / Earth & Env. / Sci. Discovery (Gokul, 1,642,491 docs after dedup) | ~8.5B | Parquet with `text` |
| Code (Ravi) | 3.28B (his report) | 83% TypeScript; Python only 6 files |
| Scientific Programming (Ravi) | 1.2–2.1B | 73% of rows repeat a file; dedup pending |
| Scientific Q&A (Ravi) | ~2.5B | checked against the rows; model-generated math |
| CAD/STEP (Ravi) | ~0.75B (he reported 2.0B) | 40.5% procedural, 59.5% zerotocad; the estimate uses the agreed text rule |
| Physics (Prabanjan) | ~0.6B | not read in full |
| Multi-step / Astronomy / Finance (Prabanjan) | 0.43B / 0.11B / 0.06B | synthetic arithmetic / 66% catalog rows / numbers only |
| Chemistry text | ~0.5B | ~2.1 GB, sampled |
| Medical (PubMed abstracts, optional) | 6–8B | not one of the 15 capabilities |
| **Engineering (6 capabilities), general language** | **none** | nobody has delivered; the model will be weak there |

Total ≈ 26–28B without Medical. "Verified" tokens exist only after: tokenizer frozen → tokenization → `manifest.json`
per-domain counts → `check_bins.py` PASS. Until then everything above is an estimate.

### Proposed first-run mix (for Raju's approval; weights are shares of training tokens, normalised automatically)
Math 0.20 · Code 0.12 · Scientific Q&A 0.10 · Biology 0.10 · Scientific Programming 0.08 · Earth & Env. 0.07 ·
Scientific Discovery 0.07 · Chemistry 0.05 · Physics 0.05 · CAD/STEP 0.04 · Multi-step 0.03 · Astronomy 0.005 ·
Finance 0.002 · Medical 0.06 (only if included). With a 30B-token run no file is repeated more than ~3×
(math 0.6 epochs, code 1.1, Chemistry 3); Astronomy and Finance are kept small because they would repeat 5× at
larger weights. The trainer prints the epochs per file and warns above 4. The draft of 20 Oct (Engineering 20%,
general language 8%) cannot be filled: there is no such data.

## 4. Training recipe (starting point; the smoke test confirms memory and speed)
| | 100M run | 1.51B |
|---|---|---|
| Sequence length | 2048 | 2048 |
| Global batch | 262,144 tokens = 128 sequences | 524,288 tokens = 256 sequences |
| How on N GPUs | 1 GPU: `--batch-size 16 --grad-accum 8` | 8 GPUs: `--batch-size 4 --grad-accum 8`; 1 GPU: `--batch-size 4 --grad-accum 64` |
| Steps | tokens / batch (2B → 7,629) | tokens / batch (5B → 9,537; 30B → 57,220) |
| Peak LR / floor | 6e-4 / 10% | 3e-4 / 10% |
| Warmup | 300 steps | max(200, 2% of steps) |
| Other | AdamW (0.9, 0.95), weight decay 0.1, grad clip 1.0, bf16 autocast, seed fixed | same |
| Eval / save | eval every 250 steps; save every 500 | eval every 500; save every 500 (a checkpoint with optimizer state is ~18 GB: keep `ckpt_last` and `ckpt_best` only) |

Example (100M, single GPU, mixed domains):
```
python -m llm.train --train-mix bins/math.train.bin=0.20 bins/code.train.bin=0.12 ... --val-bin bins/val.bin \
  --vocab tokenizers/vocab32k_v2/vocab32k.tok --out runs/100m --preset 100m --seq-len 2048 \
  --batch-size 16 --grad-accum 8 --max-steps 7629 --warmup-steps 300 --lr 6e-4 --eval-interval 250 --save-interval 500
```

## 5. Smoke test (before any real run) — pass criteria
Data: the 59M-token smoke set (9 domains, per-domain validation files) built from the tokenizer sample. It was tokenized with an
earlier throwaway vocabulary (its own `vocab32k.tok` is in the zip), so it tests speed, memory and resume only; it is not tied to
the vocabulary that will be frozen. Run `--preset 1p5b
--seq-len 2048` for ~200 steps:
1. Loss starts between 10.4 (= ln 32000) and about 11 (measured 10.88 at init on the real layer shapes) and falls steadily; no NaN.
2. Tokens/s per GPU and peak memory per GPU recorded (these replace the estimate in §2); micro-batch size chosen.
3. A checkpoint is written; resuming from it continues the loss curve (same loss at the next step).
4. On several GPUs: tokens/s scales close to N×, and only rank 0 writes files.
5. `scripts/eval_domains.py` gives a finite validation loss on at least two domains.

```
torchrun --nproc_per_node=8 -m llm.train --train-bin data/train.bin --val-bin data/val.bin --vocab data/vocab32k.tok \
  --out runs/1p5b_smoke --preset 1p5b --seq-len 2048 --batch-size 4 --grad-accum 4 --max-steps 200 \
  --warmup-steps 20 --lr 3e-4 --eval-interval 50 --save-interval 100
```
(on one GPU drop `torchrun ...` and use `python -m llm.train`). Stop at 40 minutes: GPU time is the scarce resource.

## 6. Pre-launch checklist (every item, in order, before paying for GPU time)
1. `python scripts/verify_vocab.py <vocab> --holdout ... --cli encode_cli` → PASS, and the SHA-256 equals the frozen hash.
2. Every source has its dedup report (Gokul ×3, Ravi Scientific Programming, Prabanjan math) and its inputs are complete (`--expect-files`).
3. Tokenization manifests: per-domain tokens within 5% of the owners' recount; `tokenizer_sha256` equals the frozen hash.
4. One train and one validation file per domain (`scripts/merge_bins.py`), then `python scripts/check_bins.py <all bins> --vocab <vocab>` → PASS.
5. Smoke test (§5) passed, including resume.
6. Disk: space for the bins plus ~40 GB of checkpoints; the per-domain validation files are present.
7. Budget and run length confirmed in writing (§9).

## 7. Evaluation suite
- Per-domain held-out loss/perplexity with `scripts/eval_domains.py` (one validation file per domain: tokenize with
  `--no-merge --val-frac 0.005`, which keeps `<job>.val.bin` per job; merge a domain's files with `scripts/merge_bins.py --split val`, or pass a wildcard to `--val`; do **not** use `--delete-shards`). Use **exact** file patterns when merging (`code.rg*.train.bin`, not `code*`): a pattern that also matches another domain's files would silently mix them. Same windows
  every time, so checkpoints are comparable.
- Regression, parser and structured-output tests: Swaraj. Gaps against the project requirements, found *before*
  training: Swaraj (start from `docs/REVIEW_FINDINGS.md`, `docs/HANDOFF_GPU.md`).
- Not defined yet: downstream task benchmarks (needs an owner). Validation sets are held out by document, but near-duplicate
  documents across train and validation can still make validation loss optimistic.

## 8. Responsibilities (Raju's list, 21 Oct)
Pankaj: pipeline, model configuration, checkpoints, progress report. Ravi: dataset processing, tokenization, integrity checks,
training-data readiness. Gokul: H100 availability, benchmarking, infrastructure. Om: compute setup with Gokul, performance,
stability (FSDP/FP8/FA3 if time allows). Swaraj: inference, parser integration, structured-output and regression tests,
requirement gaps. Avinash: dataset integration, dedup and cross-domain checks. Prabanjan: Math and his other domains,
quality validation, domain evaluation.

## 9. Open decisions and risks
Decisions for Raju: (1) choose and freeze one of the two 32K tokenizers (`vocab32k_v2`, or `vocab32k_digits_v2` with numbers encoded digit by digit at ~5–6% more tokens — recommended, see `tokenizers/README.md`); (2) compute: budget/another machine for the 1.51B run, or a
smaller first model, or 1.51B on 5B tokens; (3) the data mix in §3; (4) include Medical or not; (5) who delivers
engineering and general-language data; (6) true 3B or 2B as the ceiling.
Risks: GPU time and count; the DDP path and the 1.51B model have never run on CUDA; all numbers in §2–§3 are estimates;
the tokenizer sample has no physics, engineering or general-language text; data licences (code, peS2o, PubMed, OpenWebMath)
are the owners' claims and unchecked: do not redistribute weights or data before they are (see `docs/HANDOFF_GPU.md` §6).
