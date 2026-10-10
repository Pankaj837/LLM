# Results

## Dataset size and language yield

The final collection contains **22,632,151 tokens** from **4,579 documents** across 22 repositories. The source-level accounting is in `dataset/scientific_programming_manifest.json`.

| Language | Tokens | Provisional allocation |
|---|---:|---:|
| Python | 10,019,995 | 10,020,000 |
| C/C++ | 5,009,998 | 5,010,000 |
| R | 1,817,020 | 3,340,000 |
| SQL | 40,659 | 2,672,000 |
| MATLAB | 12,671 | 2,672,000 |
| Bash | 48,121 | 2,338,000 |
| Julia | 2,003,996 | 2,004,000 |
| Fortran | 1,669,991 | 1,670,000 |
| LaTeX | 63,318 | 1,670,000 |
| CUDA | 1,001,975 | 1,002,000 |
| Rust | 610,575 | 668,000 |
| Verilog/VHDL | 333,832 | 334,000 |

The 33.4M total target is provisional; the collection-plan README says internal language weights can be adjusted once clean-data yield is known.

## Same-split model comparison

Both checkpoints were evaluated at sequence length 512 over all **1,146,491 held-out tokens**.

| Checkpoint | Step | Loss (nats/token) | Perplexity |
|---|---:|---:|---:|
| Supplied checkpoint | 5,000 | 6.8352 | 930.03 |
| Fine-tuned checkpoint | 7,020 | **2.0537** | **7.80** |
| Training-token unigram baseline | — | 6.1639 | — |
| Uniform-token baseline | — | 9.0110 | — |

The fine-tuned model improves over the original checkpoint by 4.782 nats/token and over the unigram baseline by 4.110 nats/token on this held-out split.

## Runtime and tests

- Device: NVIDIA GeForce RTX 5060 Laptop GPU, 7.93 GiB visible memory.
- Fine-tuning: CUDA 12.8, bfloat16, batch size 4, sequence length 512, learning rate 1e-4.
- Upstream pytest suite: exit code 0 (with skips and two legacy warnings).
- GPU handoff suite: exit code 0; cached decode 87.3 tokens/s, peak memory 245 MB in the handoff test.

## Qualitative note

Sampled continuations became more code-shaped after fine-tuning, but still repeat patterns and occasionally emit identifiers from unrelated languages. The JSON files in `reports/` preserve the sampled outputs and full evaluation numbers.
