# Training plan — 1.51B model (21 Oct 2026)

Owner of this plan: Pankaj. Numbers marked *est.* are my calculations and must be replaced by Om/Gokul's
measured benchmark. Nothing below has been run on a GPU yet.

## 1. What is ready in the repo (tested on CPU)
| Item | Status |
|---|---|
| `--preset 1p5b` (1.509B) and `--preset 2b` (2.012B): 32 layers, d=2048, 16 query / 4 KV heads, FFN 5632 / 8192, 4K context, tied embeddings, vocab 32000 | parameter counts checked against an exact formula that reproduces a real built model |
| Multi-GPU: `torchrun --nproc_per_node=N -m llm.train ...` (DistributedDataParallel; `--batch-size` is per rank) | 2-process run reproduces one process with accumulation 2 (weights match); **not yet run on CUDA/NCCL** |
| bf16 autocast, AdamW, cosine schedule, grad clip, bit-exact resume, per-step metrics | tested |
| `scripts/tokenize_parquet.py` — Parquet (incl. v4.0 files) → TOK1 bins, count-only mode, per-shard train/val | tested |
| `scripts/pubmed_to_parquet.py` — PubMed XML → Parquet (for Medical) | tested, also run on a real 48 MB slice from the Drive zip |
| `scripts/eval_domains.py` — validation loss per domain | tested |

**Not in the repo yet:** FSDP/ZeRO-3, FP8 (Transformer Engine), FlashAttention-3, `torch.compile`, activation
checkpointing (Om's items). They are speed-ups, not blockers: the optimizer state of 1.51B is ~24 GB, so the model
fits on one 80 GB H100 with DDP.

## 2. Compute: what "by end of week" can mean *(est.)*
Cost ≈ 10.7 GFLOP per token (6 × 1.509B plus attention at seq 2048). At 35% of an H100's bf16 peak with this plain
PyTorch code: ≈ 3×10⁴ tokens/s per GPU ≈ **2.8B tokens per GPU-day**.

| GPUs | tokens in 3 days | tokens/parameter |
|---|---|---|
| 1 | ~8B | 5 |
| 4 | ~33B | 22 |
| 8 | ~65B | 43 |

30B tokens (≈ 20 tokens/parameter) ≈ 11 GPU-days (≈ 257 H100-hours). *(An earlier version of this plan said 8 GPU-days; that was wrong.)*

### Cost against the budget we actually have *(est.)*
Gokul's reply (21 Oct): one H100 costs ₹330 per hour and he has about ₹1,000 of credit, i.e. **about 3 H100-hours**.
Same assumptions as above (35% of bf16 peak, seq 2048, vocab 32K, ~20 tokens/parameter):

| Model | Tokens at 20/param | H100-hours | Cost at ₹330/h | Tokens in ₹1,000 |
|---|---|---|---|---|
| ~100M | 2.0B | 1.3 | ₹440 | 4.6B |
| ~300M | 6.1B | 11.8 | ₹3,900 | 1.6B |
| ~650M | 13.1B | 51 | ₹16,700 | 0.8B |
| **1.51B** | **30.2B** | **257** | **₹85,000** | **0.35B** |

1.51B on 5B tokens (3 tokens/parameter, under-trained baseline) ≈ 43 h ≈ ₹14,000; on 10B ≈ 86 h ≈ ₹28,000.
**With ₹1,000 we can run the smoke test of the 1.51B pipeline (~40 min ≈ ₹220) and one small (~100M) compute-optimal
run, not the 1.51B training.** The 1.51B run needs compute from outside this budget; the model size and token budget
are a decision for Raju.

**3B:** the configurations in `llm update.pdf` stop at 2.08B (32 layers, d=2048, 8 KV heads, FFN 8192; `2b` preset = 2.012B with 4 KV heads). A true 3B needs a new configuration (for example d=2560 gives ~2.6B), and at ~20 tokens/parameter it needs ~60B tokens against the ~25–30B we can verify today. Decision for Raju: first run 1.51B, then scale.

## 3. Data gate for the first run
Usable tokens are "verified" only after: tokenizer frozen → every owner's recount → dedup. Reported pools after my
corrections (est.): Gokul ~8.8B, Prabanjan math ~9B (after cleaning), Ravi: Code 3.28B (his report), Scientific Programming
1.2–1.9B, Scientific Q&A and CAD up to 2.5B and 2.0B but both overstated (3× storage; 41% procedural), Physics 0.6B,
Medical (PubMed abstracts, if included) ~6–8B. That is roughly 25–30B without Medical.

## 4. Milestones
| Day | What | Owner |
|---|---|---|
| Tue 20 | Plan and owners posted; GPU count and hours confirmed | Pankaj, Gokul |
| Wed 21 | 32K vocab approved and hash frozen (candidate ready); every owner's recount posted; Math cleaned; dedup keys | Pankaj, Raju, owners, Avinash, Prabanjan |
| Wed 21 | Smoke test (§5) on the real GPUs | Gokul, Om |
| Thu 22 | Tokenize everything on the training machine (~2 h per 25B tokens on 16 cores) | Ravi, Gokul |
| Thu 22 night / Fri 23 | Launch the full run; per-domain eval at every checkpoint | Gokul, Om, Pankaj |
| Fri–Sun | Run, monitor, eval, regression and parser tests on each checkpoint | Gokul, Swaraj |
| Daily | One shared progress report: status, failures, blockers, owner, deadline | Pankaj |

## 5. Smoke test (before the full run) — pass criteria
Run `--preset 1p5b --seq-len 2048` for ~200 steps on a tokenized sample (any single domain is enough):
1. Loss starts near ln(32000) ≈ 10.4 and falls steadily; no NaN.
2. GPU utilization and tokens/s/GPU recorded (this replaces the estimate in §2); peak memory per GPU recorded;
   micro-batch size chosen.
3. Checkpoint saves; resume from it continues the loss curve (same loss at the next step).
4. Multi-GPU: the same number of tokens/s scaling (≈ N×) and rank 0 alone writes files.
5. Validation loss computed by `scripts/eval_domains.py` on at least two domains.

Command (per node):
```
torchrun --nproc_per_node=8 -m llm.train --train-bin bins/train.bin --val-bin bins/val.bin --vocab vocab32k.tok \
  --out runs/1p5b_smoke --preset 1p5b --seq-len 2048 --batch-size 4 --grad-accum 4 --max-steps 200 \
  --warmup-steps 20 --lr 3e-4 --eval-interval 50 --save-interval 100
```

## 6. Evaluation suite
- Per-domain held-out loss/perplexity with `scripts/eval_domains.py` on Math, Engineering, Physics, Chemistry, Code,
  Scientific Programming and the rest (one validation file per domain: `tokenize_parquet.py --no-merge --val-frac 0.005` writes one
  per input file). Same windows every time, so checkpoints are comparable.
- Regression, parser and structured-output tests: Swaraj.
- Gaps against the project requirements, found *before* training: Swaraj (use `docs/REVIEW_FINDINGS.md` and
  `docs/HANDOFF_GPU.md` as the starting list).
- Not yet defined: downstream task benchmarks (needs an owner).

## 6b. Responsibilities (Raju's list, as I will post it)
Pankaj: pipeline, model configuration, checkpoints, progress report. Ravi: dataset processing, tokenization,
integrity checks, training-data readiness. Gokul: H100 availability, benchmarking, infrastructure. Om: compute setup
with Gokul, performance, stability (FSDP/FP8/FA3 if time allows). Swaraj: inference, parser integration,
structured-output and regression tests, requirement gaps. Avinash: dataset integration, dedup and cross-domain
checks. Prabanjan: Math and his other domains, quality validation, domain evaluation.

## 7. Open risks
1. GPU count/time unknown (decides run length).
2. The 32K tokenizer (`tokenizers/vocab32k_v1/`) is a candidate until approved and frozen; it was trained without physics/engineering/general-language text (none delivered), so retrain it before tokenizing if that data arrives.
3. Numbers in §2 and §3 are estimates until the smoke test and the recount.
4. The DDP path has only been exercised on CPU.
