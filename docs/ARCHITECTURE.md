# Architecture and interface contracts

## Data flow

```
                       ┌──────────────── training ────────────────┐
raw text ─► tokenizer.py ─► ids ─► data.py (.bin, uint16) ─► train.py ─► checkpoint.py (.pt, self-describing)
 (.tok BPE, 8000)                                                  │
                                                                   ▼
                       ┌──────────────── inference ───────────────────────────────────────────┐
text ─► tokenizer ─► ids ─► model.py ─► logits ─► sampling (model.generate) ─► ids ─► text
                               │
   embed ─► [ RMSNorm ─► GQA (RoPE on Q,K; KV-cache) ─► + ] ─► [ RMSNorm ─► SwiGLU ─► + ]  × N ─► RMSNorm ─► tied head
                               ▲
                     rope.py: RoPE | Dynamic NTK | YaRN   (cos/sin computed once per forward, shared by all layers)

model(…, return_hidden=True) ─► LMBackboneAdapter ─► ConditionalRewardModel ─► r(prompt, response, condition)
text ─► (other tokenizer allowed) ─► ConditionalRewardModel                     (text hand-off)
```

## Modules

| Module | Responsibility | Depends on |
|---|---|---|
| `src/llm/model_config.py` | the single configuration dataclass (validated, serialisable) | – |
| `src/llm/modules.py` | `RMSNorm`, `SwiGLUFeedForward` | – |
| `src/llm/dynamic_ntk.py` | `DynamicNTK.get_theta(seq_len)` (team spec) | – |
| `src/llm/rope_cache.py` | `RopeCache` cos/sin tables keyed by (length, theta, device, dtype) | – |
| `src/llm/rope.py` | `RotaryEmbedding`: RoPE / Dynamic NTK / YaRN, `get_cos_sin()` | config, dynamic_ntk, rope_cache |
| `src/llm/gqa.py` | `GroupedQueryAttention`, `KVCache` | – |
| `src/llm/model.py` | `TransformerModel` (forward, `generate`, `load_pretrained`) | all of the above |
| `src/llm/tokenizer.py` | byte-level BPE, `.tok` reader/writer, `BPETokenizer` | – |
| `src/llm/bpe_trainer.py` | pure-Python BPE trainer (tooling/tests; not the C++ vocabulary) | tokenizer |
| `src/llm/data.py` | `.bin` reader/writer, `TokenDataset`, document split, near-duplicate filter | – |
| `src/llm/checkpoint.py` | save/load with config + provenance + safety checks | config |
| `src/llm/pipeline.py` | `LLMPipeline`: text → text | model, tokenizer, checkpoint |
| `src/llm/train.py` | reference trainer (`python -m llm.train`) | model, data, checkpoint |
| `src/llm/reward/` | conditional reward model, preference data, reward training | tokenizer |

## Interface contracts (the things that break at integration time)

| Contract | Rule | Enforced by |
|---|---|---|
| **Vocabulary identity** | model `vocab_size` == tokenizer `n_vocab` == `.bin` header `vocab_size`; the vocabulary file's sha256 is stored in the checkpoint | `data.check_compatible`, `LLMPipeline.from_pretrained`, `checkpoint.load_checkpoint` |
| **Id space** | ids are `int64` tensors in memory, `uint16` on disk (so every id < 65536) | `data.write_bin` |
| **Two id spaces** | the LM (8000) and the reward model may use different vocabularies; hand off **text**, never ids | documented; `test_text_level_handoff_lm_to_reward_model` |
| **Padding** | right-pad. Pooling in the reward model reads the last *real* token. Left padding is unsupported | `ConditionalRewardModel.encode` |
| **Attention mask** | `(B, T)` 0/1 (or bool) padding mask, or a broadcastable 4-D bool/float mask | `TransformerModel.forward` |
| **Hidden states** | `forward(ids, return_hidden=True)` → `(B, T, hidden_dim)` after the final RMSNorm | `LMBackboneAdapter` |
| **RoPE placement** | inside every attention layer, on Q and K only, before KV heads are repeated; never added to embeddings | `gqa.py`; test vs naive reference |
| **Planned context** | frequencies are fixed per call from the planned total length (Dynamic NTK); `generate` passes prompt+new tokens to every forward | `rope.py`, `model.py`; logit-level tests |
| **dtype** | RoPE tables are computed in fp32 and cast to the activation dtype; `module.to(bf16)` does not corrupt frequencies | `test_casting_the_module_to_bf16…` |
| **Untrained weights** | a checkpoint flagged `untrained`, or a legacy file that looks random-initialised, emits a warning on every load | `checkpoint.py` |

## Positional-encoding modes (`ModelConfig.rope_mode`)

| Mode | Enabled by | Behaviour |
|---|---|---|
| `rope` | default | θ = `rope_theta` |
| `dynamic_ntk` | `use_dynamic_ntk=True` | θ = base for L ≤ L_orig, else `base·(L/L_orig)^(d/(d−2))`, L = planned context, L_orig = `yarn_original_max_position` |
| `yarn` | `yarn_scale_factor > 1` | NTK-by-parts frequencies (ramp linear in dimension index) and logits × (0.1·ln s + 1)² |

`dynamic_ntk` and `yarn` are mutually exclusive.

## Why RoPE-before-repeat

Rotating K before repeating KV heads and repeating first then rotating are numerically identical (the rotation is
per position and identical for copies of a head). Rotate-then-repeat is used because it rotates `n_kv_heads`
(2) instead of `n_heads` (8) key tensors and it is the layout the KV cache stores. This is an efficiency choice,
not a correctness requirement (the RoPE integration report states it as a correctness one; both orders give the
same result, verified by the naive-reference test).
