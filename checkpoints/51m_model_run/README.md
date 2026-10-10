# 51M model run — training config and metrics (submitted by Prabanjan)

**Status: COMPLETE. The missing `ckpt_best.pt` and `final_vocab_51m.tok` arrived on 2026-10-10 and are
independently verified below — strict load, spectral test, and tokenizer round-trip all pass.** Everything
below is independently verified, not taken on the submitter's word.

## Checkpoint verification (added 2026-10-10)

- **Strict load**: 0 missing / 0 unexpected keys against this repo's `base` preset, 51,665,920 params.
- **Tokenizer works**: `final_vocab_51m.tok` is valid byte-level BPE; "The mitochondria are", "DNA replication
  begins when", "Natural selection acts on", "The greenhouse effect occurs when" all encode/decode exactly.
- **Spectral (Marchenko–Pastur) test confirms genuine training**: ratios 2.1–3.7× the random-matrix edge.
- **Forward pass finite**, not degenerate.
- This checkpoint (SHA-256 `60b3cb85d52e37dfa9f3f5776888a1b5f1aed7f3644c55f911d5acef5de1e06a`, not committed —
  618 MB, exceeds GitHub's 100 MB push limit) turns out to also be the team's shared base model: Om's Quant
  fine-tune (`../quant_51m/`) and Aarav's Scientific Programming fine-tune (`../scientific_programming_51m/`)
  both independently reference this exact hash as their starting checkpoint — strong cross-corroboration this
  is real. `final_vocab_51m.tok`'s SHA-256 (`7a56c090cfbfc8913c21cdf412ed0aae1c806ab6fd6fa09c6376ab29566fd362`)
  is the one shared vocabulary used across all of those submissions.
- `metrics.jsonl` sent alongside is byte-identical to the copy already in this folder — confirmed via diff,
  so this really is the completion of the same run, not a different one reusing the same folder name.

## Is it done?

**Yes, this specific run (5,000 steps) completed cleanly.**
- `metrics.jsonl` has exactly 500 lines, steps 10 → 5000 in strict, unbroken order (`log_interval=10`), no
  duplicate or missing steps.
- Loss and perplexity fall monotonically with no NaN/Inf anywhere: val loss 6.68 → 3.84, val perplexity
  **794.5 → 46.4**.
- The LR schedule behaves exactly as configured: peaks at `0.0006` (the configured `lr`), and floors at
  `6.00e-05` precisely at step 5000 — that's `min_lr_ratio (0.1) × peak (0.0006)`, confirming the cosine decay
  is correctly stretched across the *full* 5,000-step run rather than floor-decaying early. This is the exact
  fix Raju sir asked for (§2 below) — it's applied and it worked.

## Cross-checked against Raju sir's Sept 2026 feedback (verified where possible from this data alone)

| His point | Verified? | What the data actually shows |
|---|---|---|
| Increase to 5,000–10,000 steps; fix scheduler so decay doesn't hit the floor early | ✅ Done | `max_steps: 5000`, `resume: null` in the config; LR floors exactly at step 5000, not earlier |
| `seq_len` too low vs `max_position_embeddings` | ✅ Done | `seq_len: 512` in this config (was 256) |
| Enable bf16 + `torch.compile()`, target >2,000 tok/s (from ~540) | ✅ **Exceeded** | `dtype: bf16`, `compile: true`; measured throughput **~5,245 tok/s sustained** — 2.6× his target, ~9.7× the old baseline |
| "Perplexity (127.45) verifies loss convergence" | ⚠️ **Doesn't match this file** | No step in this run has ppl 127.45. Closest is step 1400 (ppl 126.69) — an **interim** point. By step 5000 (this run's actual end) perplexity is **46.4**, far better. Either his comment was made on an earlier/partial view of this run, or on a different checkpoint — ask which, don't assume 127.45 is the final number. |
| Duplicate print logs between steps 1010–1250 | ❓ **Can't verify from this file** | `metrics.jsonl` itself has no duplicate step entries in that range. If the duplication was in console/stdout output, that log wasn't included here — ask for it if it still matters. |
| Run generation quality checks (T=0.7, top-p 0.9) at step 2,000 | ❌ Not included | No sample generations in either file submitted |
| Acquire compiled C++ binary + 32k/64k vocab for parity testing | ❌ Separate, unrelated task | Not something this training run produces; still outstanding (see `docs/CPP_TOKENIZER_PARITY.md` for the 8k-vocab parity work already done — this asks for a *larger* vocabulary, not yet acquired) |

## Config as submitted

`vocab_size: 8193` (`../final_vocab_51m.tok`, not one of the vocabularies verified so far in this repo — its
provenance/hash isn't recorded here; ask for it alongside the checkpoint). Model architecture otherwise matches
this repo's `base` preset exactly (11 layers, d=640, 8 query/2 KV heads, SwiGLU 1664, `max_position_embeddings=2048`).

Dataset: **Biology + Earth & Science** (confirmed by Prabanjan — matches his current roster assignment in
`docs/HANDOFF_GPU.md` §4). Not yet the full team-combined corpus, so this is still a 2-domain checkpoint, not
the final cross-domain one described in that section.

This now replaces `checkpoints/toy_bio_small_6.3M.pt` and `checkpoints/toy_earth_small_6.3M.pt` as the best
available checkpoint for those two domains — but it still isn't the combined-corpus model the final handoff
needs (that requires every domain's data, see `checkpoints/README.md`).
