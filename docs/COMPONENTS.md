# Components

One section per component, same template: **Purpose · Interface · Spec source · Verified by · Known limits**.
"Spec source" names the team document the behaviour comes from; where none exists the behaviour is
defined by this repository and the section says so.

---

## C1 Configuration — `llm/model_config.py`
- **Purpose:** one validated, serialisable dataclass for every hyperparameter.
- **Interface:** `ModelConfig(...)`, `.to_dict()`, `ModelConfig.from_dict(d)`, `.rope_mode`, `.trained_context_len`.
  Defaults = the 51.5M-parameter reference model (11 layers, d=640, 8 query / 2 KV heads, SwiGLU 1664, vocab 8000, ctx 2048).
- **Spec source:** GQA integration status report (“single ModelConfig dataclass”).
- **Verified by:** `test_c1_config_blocks.py`. Validation uses explicit exceptions (survives `python -O`); rejects
  head/dim mismatch, odd `head_dim`, `yarn_scale_factor < 1`, `yarn_original_max_position > max_position_embeddings`,
  YaRN + Dynamic NTK together.
- **Limits:** the removed `ntk_scale_factor` (a static factor invented during integration, in no spec) is gone;
  use `use_dynamic_ntk`.

## C2 Core blocks — `llm/modules.py`, `llm/model.py`
- **Purpose:** RMSNorm (weight-only), SwiGLU feed-forward, pre-norm residual block, tied-embedding output head.
- **Interface:** `RMSNorm(dim)`, `SwiGLUFeedForward(hidden, intermediate)`, `TransformerModel(config)`.
- **Verified by:** RMSNorm ≡ `torch.nn.RMSNorm` (1e-6); SwiGLU ≡ its formula; fresh-model initial loss ≈ ln V
  (GPT-style N(0, 0.02) init — without it the tied head starts at CE ≈ 570).
- **Limits:** full-vocabulary logits for every position (memory grows with batch × seq × vocab).

## C3 Positional encoding — `llm/rope.py`, `llm/dynamic_ntk.py`, `llm/rope_cache.py`
- **Purpose:** RoPE with optional context extension.
- **Interface:** `RotaryEmbedding(config).get_cos_sin(position_ids, dtype, context_len)`; `DynamicNTK.get_theta(L)`;
  `RopeCache.build(seq_len, head_dim, theta, …)`.
- **Spec source:** *Dynamic NTK and RoPE Cache work report* (strict threshold, exponent `d/(d−2)`, cache reuse,
  instance-local state; reference values reproduced in `test_report_reference_values`); *RoPE integration report*
  (RoPE inside every attention block on Q and K); Peng et al. 2023 for YaRN.
- **Verified by:** `test_c3_rope_yarn.py`, `test_dynamic_ntk.py` — relative-position property, norm preservation,
  YaRN frequencies vs a reference implementation, YaRN logit multiplier, Dynamic NTK values from the report,
  cached ≡ uncached decode across the trained length (logit level), no host↔device sync in the decode path.
- **Limits:** Dual Chunk Attention (mentioned in the RoPE report for >128K tokens) is **not** implemented — out of
  scope for a 2048-token model. Dynamic NTK is defined by the *planned* context (see `docs/DECISIONS.md` D1). YaRN and
  Dynamic NTK have not been evaluated on a *trained* model (perplexity vs length) — GPU handoff item.

## C4 Attention — `llm/gqa.py`
- **Purpose:** grouped-query attention with an optional pre-allocated KV cache; RoPE applied to Q and K before the KV
  heads are repeated.
- **Interface:** `GroupedQueryAttention(d_model, n_heads, n_kv_heads, head_dim, …)(x, position_ids, kv_cache, attention_mask,
  use_causal_mask, rope_cos_sin)`; `KVCache(batch, max_len, n_kv_heads, head_dim)`.
- **Spec source:** GQA integration report; the team's 10 original unit tests run unchanged in `tests/legacy/test_gqa.py`.
- **Verified by:** output equals an independent per-head naive attention (1e-7); causality; incremental and chunked
  prefill ≡ full forward; KV memory is ¼ of MHA; padding mask; bf16 finite; rotate-then-repeat ≡ repeat-then-rotate.
- **Limits:** `repeat_interleave` materialises expanded K/V each step (`SDPA(enable_gqa=True)` can avoid it — GPU
  benchmark item). Left-padded batches are not supported by generation.

## C5 Tokenizer — `llm/tokenizer.py`, `llm/bpe_trainer.py`
- **Purpose:** byte-level BPE compatible with the C++ tokenizer's `.tok` files.
- **Interface:** `BPETokenizer.from_file(path)`, `.encode(text, allowed_special=…)`, `.decode(ids)`, `.n_vocab`, `.eos_id`,
  `BPETokenizer.file_sha256(path)`; `bpe_trainer.train_bpe / write_tok`.
- **Spec source:** *Tokenizer report* — `.tok` = Base64 token bytes + ids + specials; use one vocabulary end to end.
- **Verified by:** round-trip lossless on curated + fuzzed text (unicode, emoji, all whitespace runs, contractions);
  save/load; unknown id decodes to `<unk:N>`; special-token injection blocked for prompts and documents; trainer
  invariants (unique tokens, every merge = two earlier tokens, determinism, agreement with the encoder).
- **Limits (important):** parity with the **C++** implementation is **unverified** — its source, `encode_cli` and
  the trained `vocab.tok` are not part of this repository. `tests/test_cpp_parity.py` runs it as soon as
  `TOK_CLI` and `TOK_VOCAB` are set. `bpe_trainer` output is *format*-compatible only; never present it as the
  team's vocabulary.

## C6 Generation — `TransformerModel.generate`
- **Purpose:** sampling with temperature / top-k / top-p, KV-cached or uncached, per-row EOS.
- **Verified by:** cached ≡ uncached (greedy, batched, across the trained length in all three RoPE modes);
  seed-deterministic sampling; `top_k=1` ≡ greedy; top-k / top-p semantics; `max_new_tokens=0`; empty prompt and
  negative temperature raise; finished rows stay finished; warning beyond the configured context.
- **Limits:** no repetition penalty / min-p / stop sequences / streaming; no left-padded batches.

## C7 Checkpoint — `llm/checkpoint.py`, `scripts/make_scaffold_checkpoint.py`
- **Purpose:** self-describing checkpoints (config, step, losses, vocabulary sha256, optimizer) and safe loading.
- **Interface:** `save_checkpoint(...)`, `load_checkpoint(path, model, strict, expected_vocab_sha256)`, `read_checkpoint`,
  `looks_untrained`.
- **Spec source:** defined here. The delivered `pretrain_model.pt` is a bare `state_dict`; the integration report
  itself states the files were dummies to be replaced. It is statistically indistinguishable from random weights.
- **Verified by:** `test_checkpoint.py` — round-trip, `weights_only=True`, config / vocabulary mismatch rejected,
  legacy file loads with a warning, untrained flag and heuristic, format-version guard. Release gate:
  `LLM_CKPT=<file> pytest tests/test_checkpoint.py -k trained`.
- **Limits:** **no trained checkpoint exists yet** for the reference model. The included biology run trains a
  smaller model (see `docs/BIOLOGY_SMOKE_TEST.md`).

## C8 Pipeline — `llm/pipeline.py`
- **Purpose:** text in → text out.
- **Interface:** `LLMPipeline.from_pretrained(checkpoint, vocab, config=None, device=…).generate(prompt, …)`.
- **Verified by:** fails loudly on missing checkpoint / vocab / key or config mismatch / vocabulary-hash mismatch /
  tokenizer larger than the embedding; prompts cannot inject control tokens; empty prompt raises;
  `test_e2e.py` runs text → `.bin` → train → checkpoint → pipeline → reward model.
- **Limits:** returns prompt + continuation (a documented convention); no chat template yet.

## C9 Conditional reward model — `llm/reward/`
See `docs/REWARD_MODEL.md`. Mechanism verified (FiLM identity init, padding invariance, Bradley–Terry loss, opposing-criteria
mock task, ablation control). **Its accuracy numbers are not evidence of quality** (see that document).

## C10 Token data — `llm/data.py`
- **Purpose:** the tokenizer team's `.bin` dataset format plus training-data hygiene.
- **Interface:** `read_header/read_tokens/write_bin`, `TokenDataset.get_batch / sequential_batches`,
  `document_split`, `encode_documents`, `NearDuplicateFilter`.
- **Spec source:** *Tokenizer report* — 36-byte little-endian header (`<IIIIQIQ`, magic `TOK1`), `uint16` ids.
  The exact snippet from the report reads our files (`test_file_layout_matches_the_report_exactly`).
- **Verified by:** `test_data_bin.py`, `test_e2e.py`. The near-duplicate filter exists because the biology data
  had 25% of LibreTexts validation windows also in training after exact-hash dedupe (now 0.3% overall).
- **Limits:** the meaning of the header `split` field is not specified in the report; this repo uses 0 = train, 1 = val.

## C11 Trainer — `llm/train.py`
- **Purpose:** reference next-token trainer so a real checkpoint can exist.
- **Verified by:** `test_train.py` — schedule, weight-decay groups, loss starts at ln V and falls, one-batch overfit,
  i.i.d. tokens are *not* learnable (leak guard), determinism, **bit-exact resume**, grad accumulation, non-finite
  loss aborts, checkpoint provenance.
- **Limits:** single device; no multi-GPU / FSDP; no gradient checkpointing; conventional hyper-parameters, not tuned.

## C12 Team scripts — `scripts/`
`smoke_test.py` (the team's 29-check script, kept), `make_scaffold_checkpoint.py`, `prepare_biology.py`, `eval_bio.py`.
