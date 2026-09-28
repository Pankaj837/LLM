# Handoff for GPU testing

Audience: whoever runs this on a CUDA machine. Everything below was developed and verified **on CPU only**
(Python 3.13, torch 2.12, 8 threads). GPU behaviour is therefore the main unknown; this page lists exactly what to run,
what "good" looks like, and what we need back.

## 0. What has been tested already (CPU)
See `docs/TESTING.md` for the current counts. In short: every component against an independent reference, the team's
original tests, cached ≡ uncached decoding at logit level, bit-exact training resume, mutation checks, two real-data
runs (biology and Earth & Environment — `docs/BIOLOGY_SMOKE_TEST.md`, `docs/EARTH_ENVIRONMENT_SMOKE_TEST.md`), and
the C++ tokenizer built and verified against the Python port (`docs/CPP_TOKENIZER_PARITY.md`). **Not tested
anywhere:** any CUDA path, bf16/fp16 numerics on a trained model, torch.compile, multi-GPU, structured pruning.

## 1. Setup
```bash
git clone https://github.com/Pankaj837/LLM && cd LLM
pip install -e ".[dev]"                        # installs torch/numpy/pytest and this package (editable, src/ layout)
python -m pytest -q                            # CPU suite must be green first
python -c "import torch;print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name())"
```
Record: GPU model, driver, CUDA, torch version. (`torch >= 2.5` needed for the `enable_gqa` experiment in §3.)

## 2. Automated GPU checks (`tests/test_gpu_handoff.py`, currently skipped)
```bash
python -m pytest tests/test_gpu_handoff.py -q -s      # -s prints the throughput / memory report
```
| Test | Pass criterion |
|---|---|
| CPU↔GPU fp32 logit parity | atol 2e-3 |
| KV-cache decode ≡ full forward on GPU | identical greedy tokens |
| bf16 / fp16 logits vs fp32 | ≤ 5% of the largest logit (**tolerance is a starting point; tune on a trained checkpoint**) |
| left-padded batch with fused SDPA kernels | no NaN (report which kernel was used) |
| YaRN beyond the trained length | finite output |
| decode throughput + KV memory report | prints tok/s and peak MB |

## 3. Questions only a GPU can answer
| # | Ask | Why | Report |
|---|---|---|---|
| 1 | Profile one decode step with `torch.profiler` | the `.item()` host syncs were removed on CPU-reasoning; confirm no `cudaStreamSynchronize` remains from the model path (an `eos_token_id` check still syncs once per step by design) | sync count, ms/step |
| 2 | `SDPA(..., enable_gqa=True)` instead of `repeat_interleave` in `gqa.py` | avoids materialising expanded K/V | speed ×, max abs diff vs current |
| 3 | `torch.compile(model)` on prefill and decode | is the code graph-friendly now that host syncs are gone? | works? speed-up? recompiles per step? |
| 4 | RoPE in bf16 at positions > 1k | tables are fp32 by design and cast to activation dtype; is the cast enough? | logit diff bf16 vs fp32 |
| 5 | bf16 autocast training vs fp32 on the biology data | loss curves should overlay | curve, throughput |
| 6 | Train the **base** (51.5M) preset on the prepared biology data | CPU speed is 743 tok/s, so this was never run beyond 6 steps | val loss vs step; time to reach the CPU `small` run's val loss |
| 7 | Determinism: same seed twice | training is deterministic on CPU by construction | bit-identical? which flags needed |
| 8 | Throughput at seq 256 / 512 / 2048 and batch sizes | sizing | tokens/s, peak memory |
| 9 | Dynamic NTK / YaRN quality beyond 2048 **after** a short training run | only meaningful with trained weights | perplexity vs length curve (512, 1k, 2k, 4k) for `rope`, `dynamic_ntk`, `yarn` |
| 10 | Reward model with `LMBackboneAdapter` on the trained LM | backbone d=640 | step time, memory, condition-sensitivity gap over ≥ 5 seeds |

## 4. Commands

**Dataset 1 — biology** (main target; largest corpus, 5.39M train tokens):
```bash
# data (once; ~1.5 min). --vocab-size 8000 uses the pure-Python stand-in trainer (fine for this run).
python scripts/prepare_biology.py --zip "Biology Dataset.zip" --out data/bio --vocab-size 8000

# real 51.5M model on GPU (bf16 autocast is automatic on CUDA)
python -m llm.train --train-bin data/bio/train.bin --val-bin data/bio/val.bin --vocab data/bio/vocab.tok \
    --out runs/bio_base --preset base --seq-len 512 --batch-size 32 --max-steps 5000 --warmup-steps 200 --lr 6e-4 \
    --eval-interval 250 --save-interval 1000

# evaluation (per source, control, bits-per-byte, unigram baseline, samples)
python scripts/eval_bio.py --run runs/bio_base/ckpt_last.pt --data data/bio --device cuda --out runs/bio_base/eval.json

# resume after a pre-emption (bit-exact)
python -m llm.train ... --resume runs/bio_base/ckpt_last.pt
```

**Dataset 2 — Earth & Environment** (smaller, templated data; already run at `small` size on CPU —
`docs/EARTH_ENVIRONMENT_SMOKE_TEST.md`; repeating at `base` size on GPU is optional, lower priority than biology):
```bash
python scripts/prepare_earth.py --zip "Earth and Environment Processed Dataset.zip" --out data/earth
python -m llm.train --train-bin data/earth/train.bin --val-bin data/earth/val.bin \
    --out runs/earth_base --preset base --seq-len 512 --batch-size 32 --max-steps 3000 --warmup-steps 150 --lr 6e-4
python scripts/eval_earth.py --run runs/earth_base/ckpt_last.pt --data data/earth --device cuda --out runs/earth_base/eval.json
```

**Optional — the real C++-trained vocabulary** instead of the Python stand-in, now that it's verified
(`docs/CPP_TOKENIZER_PARITY.md`): build `native/tokenizer/` (`cmake -B build && cmake --build build`, or see
`native/README.md` for a no-CMake build) and run `./build/prepare_dataset <corpus.txt> vocab.tok 8000` on the
biology corpus text, then point `--vocab` at that file instead.

**The rest of the team's datasets — the real target for the final `base` checkpoint.** Biology and Earth &
Environment (above) are only the two datasets that happened to be ready for CPU smoke-testing. The full team is
training in parallel on more domains:

| Person | Dataset(s) |
|---|---|
| Ravi | Astronomy and Space, Biology |
| Prabanjan | Chemistry, Earth & Science |
| Afnan | Engineering and Mathematics |
| OM | Quant |
| Aarav | Scientific Programming |

The reference `base` model should ultimately train on the **combined** corpus across all of these, not any one
domain in isolation — a model trained on biology alone will not generalise to astronomy or engineering prompts.
Once each person's data is prepared into a `.bin` file (following `scripts/prepare_biology.py` /
`scripts/prepare_earth.py` as the pattern — one `prepare_*.py` per source, one shared tokenizer/vocabulary across
all of them so ids line up, see `docs/DECISIONS.md` D4), concatenate the resulting train/val token streams (or
interleave batches across sources) before the final GPU run. **Do not tokenize each domain with a different
vocabulary** — that was exactly the mistake in finding F-29; agree on one `vocab.tok` for everyone first.

## 5. Reference numbers from the CPU run (for sanity comparison)
Throughput on CPU (8 threads, fp32, batch 16 × seq 256): `tiny` 13.0k tok/s, `small` 4.5k tok/s, `base` 0.74k tok/s.
The biology CPU results (loss/perplexity/bits-per-byte per source) are in `docs/BIOLOGY_SMOKE_TEST.md`; a GPU run of the
same `small` preset with the same seed should land close to them (not identical: different kernels).

## 6. Known limitations to carry into GPU testing
1. **No trained reference checkpoint.** The delivered `pretrain_model.pt` is random init (the loader warns). Two
   **toy** placeholders (`checkpoints/toy_bio_small_6.3M.pt`, `checkpoints/toy_earth_small_6.3M.pt` — 6.3M params,
   single-domain, CPU-trained) are pushed so integration work isn't blocked, but they are explicitly not the real
   model — see `checkpoints/README.md`. **Replace both with the real combined-dataset checkpoint once trained.**
2. Left-padded batches unsupported in generation and in reward pooling.
3. Dual Chunk Attention not implemented (not needed at 2048).
4. Biology data licences vary per source (`pes2o`: “unknown”). Do not redistribute models or data before checking.
5. Real preference data for the reward model is still pending from the data team — out of scope for this GPU run
   (see `docs/HANDOFF_DATA_TEAM.md`).

## 7. What to send back
A table: GPU / torch / CUDA; results of §2; answers to §3 (numbers, not adjectives); the `eval.json` and
`metrics.jsonl` of the training run; anything that failed, with the traceback.
