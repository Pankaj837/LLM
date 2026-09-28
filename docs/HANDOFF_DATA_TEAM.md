# Handoff for the data team — real preference data for the reward model

Audience: whoever (person or AI agent) is sourcing/building preference data for `src/llm/reward/`.
**This is a separate task from `docs/HANDOFF_GPU.md`** — that doc is for training the base language model on a
GPU; this one is for the conditional reward model, which currently only has synthetic training data.

## 1. What exists today (read this first)

`src/llm/reward/preferences.py` generates **synthetic** preference data — 240 pairs, 20 topics, template-written
responses, no real annotators. It exists to prove the *mechanism* works (see `docs/REWARD_MODEL.md`), not to train
a model anyone should trust. It is also weak as evidence: a 5-line length/contraction rule scores 100% on its
validation set. **Your job is to replace it with real data**, not to add to it.

## 2. The exact schema to produce

```python
@dataclass
class PreferencePair:
    prompt: str        # the question/instruction
    chosen: str         # the response judged BETTER under `condition`
    rejected: str        # the response judged WORSE under `condition`
    condition: str       # which criterion this judgement is under, e.g. "helpful", "concise", "formal"
    conflicting: bool    # True if some OTHER condition would rank this same pair the other way
```
(`src/llm/reward/preferences.py:234`, `CONDITIONS = ["helpful", "concise", "formal"]` today — see §4 if you're
changing the set of conditions.)

**What makes this reward model different from an ordinary chosen/rejected pair:** the same two responses can
appear **twice**, once per condition, with opposite winners — that's the entire point of a *conditional* reward
model (`r = f(prompt, response, condition)` instead of `r = f(prompt, response)`). If your data doesn't have cases
where two criteria disagree on the same pair, the conditional part of the model has nothing to learn from.

**Minimum viable delivery: a flat file of these five fields — JSONL is easiest:**
```json
{"prompt": "...", "chosen": "...", "rejected": "...", "condition": "helpful", "conflicting": true}
```
`conflicting` can be `false`/unknown for every row if you don't have the cross-condition labels — it's used for a
diagnostic slice, not required for training (see `condition_blind_ceiling()` in `preferences.py` for how it's used
in this repo).

## 3. Text and tokenization — the part most likely to break silently

- **Send raw text**, not pre-tokenized ids. If you must pre-tokenize for your own storage, use the C++ tokenizer in
  `native/tokenizer/` (built and verified — `docs/CPP_TOKENIZER_PARITY.md`) or `src/llm/tokenizer.py`, and tell us
  which `vocab.tok` you used — its identity is hashed and checked at load time (`docs/DECISIONS.md` D5), so a
  mismatch fails loudly rather than silently corrupting ids. **Do not invent your own binary format** — two earlier
  deliveries (Earth & Environment's `.bin` files; the reward model's own tokenizer) used incompatible formats and
  had to be adapted after the fact (findings F-17, F-29 in `docs/REVIEW_FINDINGS.md`).
- Plain UTF-8 text, no control characters. `<|endoftext|>` in your text is treated as a literal token by our
  tokenizer if you tokenize it yourselves — don't rely on it as a delimiter in raw text you hand us.
- No length requirement, but very short (<3 tokens) or very long (>1000 words) responses are harder to pool
  correctly (see `docs/REWARD_MODEL.md`, "Pooling at the last real token").

## 4. If you're changing or extending the conditions

`CONDITIONS` is currently `["helpful", "concise", "formal"]`, chosen because they're easy to construct synthetically
and two of them (`helpful`/`concise`) are guaranteed to conflict. Real preference data usually has richer,
continuous attributes — e.g. NVIDIA's HelpSteer2 rates responses 0–4 on helpfulness, correctness, coherence,
complexity, and verbosity. If your source data has attributes like that:
- tell us the exact attribute names and scale (0–4 int, 1–5 Likert, binary, etc.) rather than us guessing,
- we'll adapt `ConditionalRewardModel`'s condition embedding to match (it's an `nn.Embedding(num_conditions, ...)` —
  a fixed small vocabulary of conditions, not free text, so a closed list of names is what we need either way).

## 5. Splitting

Split by **prompt/topic**, never by individual pair — putting the same prompt's pairs on both sides of a train/val
split measures memorisation, not generalisation (`preferences.py::split`, and this exact mistake is finding F-25
from an earlier dataset). If you hand us unsplit data we'll split it the same way; tell us if you've already split
it and by what field.

## 6. What we do with it (so you know it'll actually get used)

```
your data (prompt/chosen/rejected/condition, raw text)
    -> tokenize with the agreed vocab.tok (or we do it)
    -> src/llm/reward/train_reward.py  (Bradley-Terry loss, FiLM conditioning)
    -> evaluated for: overall accuracy, accuracy on conflicting pairs specifically,
       accuracy with condition shuffled (control — should collapse to chance),
       and a comparison against a trivial surface-feature baseline (so we don't fool ourselves again)
```
Full mechanism and current (synthetic-data) results: `docs/REWARD_MODEL.md`.

## 7. Send it back as

- A file (JSONL preferred, CSV/Parquet fine) with the 5 fields in §2, or a link to where it lives.
- The vocabulary file and its identity, if you tokenized it yourselves (§3).
- The condition list and what each one means, if different from `["helpful", "concise", "formal"]` (§4).
- Roughly how many pairs, how many distinct prompts, and how it was collected/labelled (template-generated,
  human-annotated, model-judged, etc.) — this goes directly into how much we trust the resulting model.

## Questions, not assumptions
If anything above is ambiguous for the data you actually have, ask rather than guess at the schema — a silent
mismatch here (wrong tokenizer, pairs leaked across train/val, conditions that don't actually conflict) produces a
reward model that looks fine in a table and is wrong in a way nobody notices until much later, which is exactly the
failure mode `docs/REVIEW_FINDINGS.md` exists to catalogue.
