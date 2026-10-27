# GPU session report

- Hardware: NVIDIA H100 80GB HBM3; torch 2.14.1+cu130, CUDA 13.0
- Code: commit `bc1788f5ee`; tokenizer SHA-256 `10b3c5aecf6b8174...`
- Domains in the mix: math, code, sciqa, bio, sciprog, earth, scidisc, chem, physics, cad, multistep, astro, finance

## Benchmark (synthetic tokens)

| preset | micro-batch | tokens/s | step s | peak GB | MFU | Rs per 1B tokens | status |
|---|---|---|---|---|---|---|---|
| 100m | 1 | 66,049 | 0.031 | 3.5 | 5.5% | 1,388 | ok |
| 100m | 2 | 123,602 | 0.033 | 5.5 | 10.3% | 742 | ok |
| 100m | 4 | 178,226 | 0.046 | 9.4 | 14.9% | 514 | ok |
| 100m | 8 | 198,368 | 0.083 | 17.3 | 16.6% | 462 | ok |
| 100m | 16 | 216,330 | 0.151 | 33.2 | 18.1% | 424 | ok |
| 100m | 32 | 225,631 | 0.29 | 64.9 | 18.9% | 406 | ok |
| 300m | 1 | 31,467 | 0.065 | 8.4 | 7.7% | 2,913 | ok |
| 300m | 2 | 59,608 | 0.069 | 12.1 | 14.6% | 1,538 | ok |
| 300m | 4 | 73,236 | 0.112 | 19.6 | 18.0% | 1,252 | ok |
| 300m | 8 | 82,034 | 0.2 | 34.5 | 20.1% | 1,117 | ok |
| 300m | 16 | 88,475 | 0.37 | 64.6 | 21.7% | 1,036 | ok |
| 300m | 32 | | | | | | OOM |
| 1p5b | 1 | 14,831 | 0.138 | 30.5 | 16.0% | 6,181 | ok |
| 1p5b | 2 | 20,089 | 0.204 | 38.3 | 21.7% | 4,563 | ok |
| 1p5b | 4 | 24,645 | 0.332 | 55.3 | 26.6% | 3,719 | ok |
| 1p5b | 8 | | | | | | OOM |

## Smoke test

- PASS
- loss 8.696 -> 7.936, 25,373 tokens/s, peak 60.7694208 GB, 4.5 min

## Training

- 3,215 steps x 262,144 tokens = 842,792,960 tokens (micro-batch 32, accumulation 4); measured 224,031 tokens/s
- wall time 63.1 min; exit code 0
- training loss 9.812 (step 10) -> 2.216 (step 3215)
- validation loss 4.295 -> 2.162 (best 2.162 at step 3215)

## Per-domain validation, ckpt_last (step 3215), mean 2.1182

| domain | val loss | perplexity |
|---|---|---|
| math | 2.9813 | 19.7 |
| code | 1.2841 | 3.6 |
| sciqa | 1.9921 | 7.3 |
| bio | 2.5920 | 13.4 |
| sciprog | 1.8042 | 6.1 |
| earth | 2.9228 | 18.6 |
| scidisc | 1.7451 | 5.7 |
| chem | 2.7711 | 16.0 |
| physics | 2.9609 | 19.3 |
| cad | 0.1456 | 1.2 |
| multistep | 1.6348 | 5.1 |
| astro | 3.3008 | 27.1 |
| finance | 1.4018 | 4.1 |

## Per-domain validation, ckpt_best (step 3215), mean 2.1182

| domain | val loss | perplexity |
|---|---|---|
| math | 2.9813 | 19.7 |
| code | 1.2841 | 3.6 |
| sciqa | 1.9921 | 7.3 |
| bio | 2.5920 | 13.4 |
| sciprog | 1.8042 | 6.1 |
| earth | 2.9228 | 18.6 |
| scidisc | 1.7451 | 5.7 |
| chem | 2.7711 | 16.0 |
| physics | 2.9609 | 19.3 |
| cad | 0.1456 | 1.2 |
| multistep | 1.6348 | 5.1 |
| astro | 3.3008 | 27.1 |
| finance | 1.4018 | 4.1 |

Total session clock: 69.0 min = about Rs 379 at Rs 330/h.
