# Design decisions

Short records of the choices that are not obvious from the code. Each states the decision, the reason, and
what would make us revisit it.

## D1 — Dynamic NTK uses the *planned* context length, not the running length
**Decision.** θ is a function of the total context planned for the call (`prompt + max_new_tokens` in `generate`,
the KV-cache capacity when a cache is supplied, otherwise the input length). It is constant for the whole call.
**Why.** Dynamic NTK as specified makes θ depend on the current sequence length. With a KV cache the keys of earlier
tokens are stored already rotated; recomputing θ at each step would rotate new keys with a different θ than the
cached ones and make cached and uncached decoding disagree. Fixing θ per call keeps every token of a sequence on the
same frequencies (verified at logit level, `test_planned_context_changes_logits_and_cached_uncached_logits_agree`).
**Consequence.** The same prefix decoded with two different `max_new_tokens` beyond the trained length uses
different θ, so its logits differ slightly. Within the trained length nothing changes (θ = base).
**Revisit if** the team wants per-step recomputation (then cached keys must be stored un-rotated).

## D2 — The team's `DynamicNTK` / `RopeCache` are used as delivered
**Decision.** Their formula and cache semantics are preserved; only argument validation and a correct cache key were added.
**Why.** The delivered `RopeCache` keyed on sequence length only, so the same length with a different θ returned stale
tables (`test_cache_is_keyed_on_theta_not_only_length`). The unified `rope.py` that preceded this repository implemented a
different, static scaling and ignored the report; the report's own numbers are now regression tests.

## D3 — YaRN follows the paper / reference implementation
**Decision.** Frequency blend ramp linear in the frequency-dimension index; attention logits multiplied by
(0.1·ln s + 1)².
**Why.** The first integration's `yarn.py` used a dimension-index ramp (as the reference does); the later unification replaced it with a ramp linear
in wavelength (frequencies off by up to 25–71%) and applied the reciprocal square root as the logit multiplier
(dampening instead of sharpening). Both are tested against an independent reference.

## D4 — One configuration object, one tokenizer
**Decision.** `ModelConfig` is the single source of truth; `src/llm/tokenizer.py` is the only tokenizer; the reward model imports it.
**Why.** Two Python ports of the same C++ tokenizer had already diverged (unknown-id decoding). One implementation cannot drift.

## D5 — Checkpoints are self-describing; legacy files still load
**Decision.** New format stores config, step, losses, vocabulary sha256 and optimizer state; loading verifies config and
vocabulary identity; legacy raw state dicts load with a warning.
**Why.** A bare `state_dict` cannot be checked against the model or tokenizer it belongs to, which is how an
untrained scaffold named `pretrain_model.pt` could circulate as if it were a pretrained model.
`torch.load(weights_only=True)` is always used (no arbitrary unpickling).

## D6 — The reward model stays outside the LM forward pass
**Decision.** `TransformerModel.forward` has no reward path; the reward model consumes `return_hidden=True` through
`LMBackboneAdapter`, or text.
**Why.** Matches the integration report (“optional scoring path outside the core forward pass”), keeps the LM's tests and
checkpoints independent of the reward work, and avoids two id spaces leaking into one module.

## D7 — Weight initialisation N(0, 0.02)
**Why.** With a tied output head and default `nn.Embedding` (N(0,1)) the initial loss is ≈ 570 instead of ln V ≈ 9.
The delivered checkpoint's statistics show the same 0.02 scale.

## D8 — Data hygiene is part of the library, not the script
**Decision.** Document-level splits, EOS-terminated documents, forged-boundary protection and near-duplicate filtering
live in `src/llm/data.py` with tests.
**Why.** On the biology data, exact-hash dedupe left 25% of LibreTexts validation windows also present in training
(republished chapters). Content-defined shingle sampling removed them (overall leakage 4.9% → 0.3%).

## D9 — `stop_step` instead of shortening `max_steps`
**Decision.** Early exit does not change the learning-rate schedule.
**Why.** A run “stopped early” by lowering `max_steps` follows a different cosine decay and cannot be resumed
bit-identically; resume is verified bit-exact.
