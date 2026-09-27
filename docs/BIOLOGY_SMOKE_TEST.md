# Biology real-data smoke test

**Purpose.** Show that the whole path works on *real* text — raw documents → tokenizer → `.bin` → trainer → checkpoint →
pipeline → evaluation — and that what the model learns is real. **Not** a claim about model quality: the model is 6.3M
parameters trained for one epoch on 5.4M tokens on a CPU.

Data: `Biology Dataset.zip` (7 gzipped JSONL shards). Everything below is reproducible with the commands at the end;
the exact files and hashes are in `data/bio/manifest.json`.

## 1. Data profile (before touching a model)

| Shard | Docs | Text | Median doc | Bio-term density* | Notes |
|---|---:|---:|---:|---:|---|
| `pes2o` | 6,499 | 196 MB | 28k chars | 0.021 | licence field: **“unknown” for all records** |
| `qbio_full` | 3,116 | 193 MB | 53k | 0.012 | arXiv full papers; 22 exact duplicates |
| `libretexts` | 15,501 | 125 MB | 5k | 0.017 | textbooks, **366 exact duplicates and many republished chapters**; includes non-biology chapters |
| `pmc_oa` | 2,417 | 70 MB | 26k | 0.022 | PubMed Central open access |
| `bhl` | 5,995 | 14 MB | 2.3k | 0.002 | 18th–19th century botany OCR, noisy (“Sea Gj/lrfoiuer”) |
| `qbio` | 2,881 | 3 MB | 1.1k | 0.026 | arXiv abstracts |
| `gutenberg` | ≫ 2,400 | 349 MB (gz) | 287k | **0.0002** | general books; only 7% of documents mention any biology term |

\*fraction of the first 3,000 characters' words that are biology terms (cell, protein, gene, …). Gutenberg is 50–100× lower, so it is
**excluded from training** and used as a **control**.

## 2. Decisions (what, why)

| Decision | Reason |
|---|---|
| Train on 6 sources, ~21.5 MB text; Gutenberg (2 MB) = control only | Gutenberg is not biology; as a control it shows the model is not merely modelling English |
| Fixed character budget per source, deterministic **stride** over the whole shard | “first N documents” would bias toward early records |
| Documents cut at 10,000 chars (at a word boundary) | one book/paper must not dominate; more distinct documents per MB |
| Exact **and near-duplicate** removal before splitting | exact-hash dedupe alone left **25.5% of LibreTexts validation windows in training** (4.9% overall); see D8 |
| Split by **document**, stratified per source (5%) | never by chunk; validation text is unseen in training |
| Tokenizer trained on **training documents only**, vocab 8000 | the vocabulary itself cannot leak validation text; 8000 matches the reference model |
| Per-source validation + control + bits-per-byte | an average would hide the noisy OCR source; bpb is tokenizer-independent so results compare across datasets |
| `small` preset (6.3M), 1,400 steps × 16 × 256 tokens ≈ 1.07 epoch | one pass ⇒ no repeated data ⇒ honest validation; ~25 min on CPU |

## 3. Data integrity checks (all recorded in the manifest)

| Check | Result |
|---|---|
| Round-trip exactness `decode(encode(doc)) == doc` for **every** document | **0 failures** |
| Ids < vocab (8000) and < 65536 (uint16) | yes (max id 7999) |
| EOS count == document count | yes |
| Validation 32-token windows also present in training | **0.275%** (was 4.88% before near-duplicate filtering; libretexts 25.5% → ≈0) |
| Vocabulary tokens used in training data | 98.7% |
| Training tokens / validation tokens | 5,388,291 / 273,913 (4,077 / 214 documents) |
| Bytes per token (train) | 3.80; 4.0–4.4 on abstracts/textbooks, **2.42 on BHL OCR** (noisy text tokenises poorly) |
| Single-byte tokens (fragmentation) | 18.5% train; 28.9% BHL |

Sampling detail (kept documents after stride / dedupe): pes2o 591, pmc_oa 484, libretexts 787 (75 near-duplicates dropped),
qbio_full 388, qbio 1,441, bhl 600, gutenberg control 201. Some budgets were not fully reached (pmc_oa 4.5 of 5.0 MB, qbio 1.7 of
3.0 MB — the shard is only 3.3 MB) because the stride is an integer.

## 4. Results (small preset, seed 0)

Training: validation loss 8.99 (uniform) → 6.32 → 5.79 → 5.48 → 5.27 → 5.10 → 4.99 → **4.93** at steps 200 … 1400, monotone; no
divergence. Final evaluation on the **whole** validation set (`scripts/eval_bio.py`; the 4.93 above used 40 batches):

| Set | Loss (nats/token) | Perplexity | Bits per byte |
|---|---:|---:|---:|
| **Validation, all sources** | **4.958** | **142** | **1.856** |
| pmc_oa | 4.776 | 119 | 1.742 |
| qbio_full | 4.929 | 138 | 1.858 |
| pes2o | 4.941 | 140 | 1.769 |
| bhl (noisy OCR) | 5.057 | 157 | 3.010 |
| libretexts | 5.097 | 164 | 1.789 |
| qbio | 5.111 | 166 | 1.663 |
| **Control: non-biology books (Gutenberg)** | 5.910 | 369 | 3.013 |
| *Baseline: uniform* | 8.987 | 8000 | – |
| *Baseline: unigram fitted on train* | 7.396 | 1629 | – |

Reading it:
- **Beats the unigram baseline by 2.44 nats** — it uses context, not only word frequency.
- **Control is 0.95 nats worse than in-domain** (and 1.16 bits/byte worse): the model specialised to scientific text.
- Per-source spread is small for the scientific sources (bpb 1.66–1.86); the OCR source is ~1.2 bits/byte worse, as expected.
  (Loss per token and bits per byte rank sources differently because bytes/token differs — use bpb to compare *across* sources.)

## 5. Negative control: the gain is not a leak or a bug

Same model family (`tiny`, 300 steps, seed 0), one run on real token order and one on the **same tokens shuffled**
(all sequence structure destroyed; only unigram statistics remain):

| Training data | Val loss @100 / 200 / 300 steps |
|---|---|
| real order | 7.02 / 6.43 / **6.23** (still falling) |
| **shuffled tokens** | 7.43 / 7.41 / **7.41** = the unigram baseline (7.396); cannot go lower |

## 6. The real 51.5M configuration

`--preset base` on the same data, 30 steps (batch 8 × 256): loss 8.18 → 7.49, val 7.57, **680 tokens/s on CPU**.
This only shows the reference configuration trains on real text; a meaningful run needs the GPU (`docs/HANDOFF_GPU.md`).
CPU throughput for the other presets: `tiny` 13k tok/s, `small` 4.2k tok/s.

## 7. Samples (small preset; qualitative only)

Prompt “DNA replication begins when”, sampled (T=0.8, top-p 0.9): *“…the PKT pathway, and a TC-1 gene, with the D-terminal
membrane-dependent DNA-binding protein-1, and IL-1-p-mediated…”* — plausible **vocabulary and style** of biology papers, not
meaning. Greedy decoding loops (“the cell is the most common, the cell is the cell…”). Both are what a 6M-parameter, one-epoch
model does. Some outputs contain markdown/figure residue (`###`, `.gif)`, `:::`) inherited from the LibreTexts/arXiv text —
a cleaning decision for a real run.

## 8. What this does *not* show
- Nothing about quality at scale; nothing about the 51.5M model beyond “it trains”.
- The tokenizer is the **Python stand-in**, not the team's C++ vocabulary (parity unverified).
- Near-duplicate removal is at the 12-word-shingle level; paraphrases and translations are not caught.
- Licences differ per record (`pes2o`: unknown). Do not redistribute derived models or data without checking.
- Single seed; differences between sources of a few percent are within what another seed could change.

## 9. Comparing with other domains (chemistry, earth science)
No results for other domains were available to this run, so nothing is merged or compared. To compare fairly, run the **same protocol**
on each domain and fill one row per domain: tokenizer (vocab size, trained on that domain's train docs only), tokens, model preset,
steps, **val bits-per-byte**, **unigram baseline in the same units**, control-set gap, validation-window leakage %, round-trip
failures. Compare **bits per byte and the gap to the unigram baseline**, not raw loss or perplexity (they depend on the tokenizer).

## 10. Reproduce
```bash
python scripts/prepare_biology.py --zip "Biology Dataset.zip" --out data/bio --vocab-size 8000      # 80 s
python -m llm.train --train-bin data/bio/train.bin --val-bin data/bio/val.bin --vocab data/bio/vocab.tok \
    --out runs/bio_small --preset small --seq-len 256 --batch-size 16 --max-steps 1400 --warmup-steps 100 \
    --lr 2e-3 --eval-interval 200 --eval-batches 40 --seed 0                                         # ~25 min CPU
python scripts/eval_bio.py --run runs/bio_small/ckpt_last.pt --data data/bio --out runs/bio_small/eval.json
BIO_ZIP="Biology Dataset.zip" python -m pytest tests/test_bio_smoke.py -s                          # ~1 min opt-in test
```
