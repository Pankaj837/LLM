# Insights for the team (by component, not by person)

Principle used throughout: **reuse what already works, change only what is wrong, and say why.** This page lists,
per component, what was kept as delivered, what was changed, and what is still the team's to decide.

## Kept as delivered (verified, not rewritten)
| Component | Evidence |
|---|---|
| GQA + KV cache (`gqa.py`) | equals an independent naive attention to 1e-7; the team's 10 unit tests pass unchanged; one optional argument added |
| RMSNorm / SwiGLU (`modules.py`) | ≡ `torch.nn.RMSNorm`; formula test |
| Team's `DynamicNTK` formula and `RopeCache` idea | integrated as the spec; the report's own numbers are tests |
| Tokenizer encode/decode logic | lossless on curated + fuzzed text; agrees with the earlier second copy on 515/515 cases |
| Reward-model mechanics (FiLM, pooling, BT loss) | padding-invariant to 1e-8; opposing-criteria mock task learns and flips; ablation control fails as it should |
| The 29-check smoke script and integration script | kept, ported to the package, still pass |

## Changed, and why
| Change | Reason (evidence) |
|---|---|
| YaRN frequencies and logit multiplier | unification had drifted from the paper and from the team's earlier `yarn.py` (frequencies off up to 71%; logits dampened instead of sharpened) |
| Dynamic NTK actually implemented per the report | the unified `rope.py` implemented a static factor active even below the trained length |
| `RopeCache` key includes θ | stale tables for the same length with a different θ |
| Weight init N(0, 0.02) | initial loss 573 vs 9 with the tied head |
| Pipeline / loader fail loudly | a typo silently produced a random model and a byte tokenizer |
| Prompts and documents cannot inject `<|endoftext|>` | user text became a control token |
| Self-describing checkpoints | the delivered file could not be verified or reproduced; it was random weights under a “pretrain” name |
| One tokenizer | two copies had diverged |
| Host-sync removal in the decode path | every layer synchronised with the host every step |
| Near-duplicate filter in data prep | 25% of LibreTexts validation windows were also in training |

## Still the team's decision
1. **Train the reference model.** Nothing quality-related can be concluded until a 51.5M checkpoint is trained (the
   trainer, `.bin` reader and data pipeline are ready and proven on real text).
2. **Deliver the C++ tokenizer source / `encode_cli` / trained `vocab.tok`** so C++ parity can be verified
   (`tests/test_cpp_parity.py`). Until then every result uses the Python stand-in vocabulary.
3. **Reward model:** move from synthetic data (a 5-line rule solves it) to real preference data with several
   quality axes; report several seeds with a length-only baseline. Decide standalone backbone vs shared LM backbone.
4. **Structured pruning:** not in the repository; when it arrives, evaluate the reward model's condition-sensitivity gap
   before/after, not only LM loss.
5. **Dynamic NTK semantics:** this repo fixes θ per planned context (decision D1). If per-step recomputation is wanted,
   cached keys must be stored un-rotated.
6. **The header `split` field** in `.bin` files is unspecified in the tokenizer report (0 = train / 1 = val assumed).

## Observations worth sharing
- **Read the data before training.** Profiling the biology shards found: 366 exact duplicates in LibreTexts, republished
  chapters that defeat exact-hash dedupe, OCR-noisy 18th-century botany (BHL), a general-books shard (Gutenberg) that is
  50× less biological, and a shard whose licence is “unknown”. Each changed a decision.
- **A validation set that shares text with training looks great and means nothing.** Measure overlap explicitly (the
  manifest records it).
- **Report bits-per-byte alongside loss** so runs with different tokenizers (or different datasets such as chemistry and
  earth science) can be compared honestly.
- **A test that can't fail proves nothing.** Two of the first policy tests passed on broken code; mutation testing exposed it
  (`scripts/mutation_check.py`).
- The RoPE integration report says the KV-repeat must follow RoPE “or the position maths breaks”. Both orders give identical
  results (tested); the order is an efficiency choice.
