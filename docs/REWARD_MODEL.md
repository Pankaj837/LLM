# Conditional Reward Model

Part of the from-scratch transformer project (positional encodings, Dynamic NTK/YaRN, GQA, structured pruning, CRM).

## What this is

A reward model scores a response so that a language model can be trained to
prefer better outputs. The standard formulation collapses "better" onto a
single axis:

```
r = f(prompt, response)
```

That is the wrong shape for human preference. The same two responses can be
ranked oppositely depending on what you are optimising for — a thorough
three-sentence answer beats a one-liner on *helpfulness* and loses to it on
*conciseness*. A single-scalar reward model has to pick one ordering and be
wrong about the other.

A conditional reward model takes the criterion as an explicit input:

```
r = f(prompt, response, condition)
```

One set of weights covers every criterion. At RL time you select or blend
conditions instead of training and maintaining a separate reward model per
axis.

## Result

Trained on 192 preference pairs, evaluated on 48 pairs from **held-out topics**
(the split is by prompt, so no validation prompt or response string appears in
training).

| Metric | FiLM | Concat | Condition ablated |
|---|---|---|---|
| Validation accuracy | **0.938** | 0.958 | 0.604 |
| Accuracy on conflicting pairs | 0.950 | 0.975 | 0.575 |
| Accuracy with condition shuffled at test | 0.542 | 0.583 | 0.604 |
| Condition-sensitivity gap | **+0.396** | +0.375 | +0.000 |
| Ranking flips on swap test | YES | YES | NO |

**Condition-blind ceiling: 0.667.** This is the exact upper bound on what *any*
reward model that ignores the condition can score on this validation set,
computed in `condition_blind_ceiling()`. A condition-blind model must commit to
one ranking per pair, so wherever two criteria disagree it is guaranteed wrong
on at least one. Both conditional models clear that ceiling by a wide margin;
the ablated model sits just below it, which is exactly where theory says it
should land.

### Condition-swap test

One pair, one backbone pass, scored under each condition:

```
helpful   detailed=+2.880  brief=-2.777  -> detailed
concise   detailed=-2.548  brief=+2.366  -> brief
formal    detailed=-2.565  brief=-2.500  -> brief
```

The ranking inverts between `helpful` and `concise`. This is the load-bearing
result — high overall accuracy can be achieved by a model that has quietly
learned one global notion of quality and ignores the condition input entirely.
The flip cannot.

The two controls close off that explanation. Shuffling conditions at test time
drops accuracy to chance (0.542), so the model is genuinely reading the
condition. Shuffling them during *training* produces a model with a
sensitivity gap of exactly zero and no flip — the failure mode, reproduced
deliberately.

## Design decisions

**FiLM conditioning.** The condition generates per-feature scale and shift
applied to the pooled representation: `h' = γ(c)·h + β(c)`. It is
multiplicative, so the condition can gate features rather than only adding a
bias the head can learn to ignore. Initialised at γ=1, β=0 so training starts
from the identity.

Concatenation was implemented as a baseline and slightly outperformed FiLM here
(0.958 vs 0.938). That is a real finding and it is reported as such: at this
scale, with three conditions defined by surface features, FiLM's extra
expressiveness is not needed. FiLM should be expected to pull ahead with more
conditions or subtler criteria, but that is a prediction, not something this
experiment shows.

**Late fusion.** The backbone never sees the condition, so one forward pass can
be scored under every condition. This is what makes `score_all_conditions()`
cheap and the swap test practical. A text-prefix approach (`<|helpful|>` as a
real token) would generalise better to unseen criteria phrased in natural
language, but costs context length and forces one backbone pass per condition.

**Pooling at the last real token.** The backbone is causal, so this is the only
position that has attended to the whole sequence. Mean-pooling would dilute it
with prefix positions that cannot see the response. Every example ends with an
EOS token so there is a consistent position to pool from.

**L2 penalty on rewards.** Bradley-Terry supervises only the *difference*
between scores, so the absolute scale is unidentified and drifts outward during
training, which destabilises any downstream RL stage. A small penalty pins the
scale without touching the ranking.

**Left truncation.** Sequences over `max_len` are cut from the front. The
response end is what gets pooled, so it is the part that cannot be lost.

## Tokenizer integration

The team's tokenizer is C++ with no Python bindings, but the reward model trains in PyTorch. The shared
`src/llm/tokenizer.py` is a line-for-line Python port of `pretokenize` and `bpe_encode_piece` that loads the same `vocab.tok`
file (one implementation for the whole stack; an earlier second copy had diverged on unknown-id decoding).
`tests/test_cpp_parity.py` asserts id-for-id equality with the C++ `encode_cli` on punctuation, contractions, multi-byte
UTF-8, digit/letter boundaries and repeated whitespace. **It needs the C++ binary and the real `vocab.tok`, which are not part
of this repository, so C++ parity is currently UNVERIFIED.**

## Limitations

**The data is synthetic.** Responses are generated with known attributes
(detail level, register) and labels are derived from those attributes rather
than collected from annotators. The model can therefore learn surface cues —
sentence count, contraction frequency — instead of anything deeper. This is a
real weakness and should be stated in any presentation.

It is also a deliberate trade. Real preference data gives you no ground truth
about *when two criteria should disagree*, so a ranking flip is
indistinguishable from noise. Here the disagreements are known by construction,
which is what makes the condition-blind ceiling computable and the swap test
interpretable. The architecture is unchanged if you swap in real data; only
`data/preferences.py` needs replacing.

**Scale.** 1.05M parameters, 240 pairs, 20 topics. Enough to demonstrate the
mechanism, not enough to say anything about how it behaves at scale.

**Unseen conditions.** The condition embedding is a lookup table, so this model
cannot handle a criterion it was not trained on. The text-prefix variant would
address that.

## How to read the results (review notes)

- **A 5-line rule scores 100%.** A hand-written rule using only character length and contraction count
  (`surface_heuristic_accuracy`) solves the validation set completely, so accuracy here shows the model can *read the
  condition*, not that it has learned anything about response quality. `train_reward` now prints this baseline.
- **Validation is 4 held-out topics / 48 pairs**, so one pair is 2.1 accuracy points. Across seeds 0-3 FiLM and concat
  trade places (FiLM 0.90/1.00/0.96/0.94 vs concat 0.90/0.98/1.00/1.00), i.e. their difference is noise. Report mean and
  spread over several seeds, not one run.
- The swap test now uses a **held-out** prompt and reports `n/a` where a criterion does not apply to the pair
  (e.g. `formal` on two casual answers) instead of an arbitrary winner.

## Interaction with the rest of the project

The reward head only touches the final hidden state, so a different backbone can be swapped in. `LMBackboneAdapter`
(in `reward_model.py`) wraps the language model via `forward(..., return_hidden=True)`; inputs must be right-padded. The LM and
the CRM may use different vocabularies (8000 vs ~1.7k ids), so hand off **text**, never token ids. `TransformerConfig` carries
`n_kv_head` (unused stub) and `rope_scale` (linear position interpolation; it is *not* Dynamic NTK or YaRN).

**Structured pruning needs coordination.** Pruning optimises for preserved language-modelling quality, but the reward model
depends on the *pooled* representation at the last token, which is not what pruning measures. Reward accuracy should be
evaluated before and after pruning as a separate check — specifically the condition-sensitivity gap, since pruning away the
low-magnitude features the FiLM layer modulates would collapse the model to condition-blind behaviour while overall accuracy
still looks fine. (No pruning code is part of this repository, so this interaction is untested.)

## Files

```
llm/reward/
├── transformer.py      standalone backbone (contract: forward(ids, mask) -> [B, T, d_model])
├── reward_model.py     the CRM: FiLM + concat conditioning, Bradley-Terry loss, LMBackboneAdapter
├── preferences.py      synthetic dataset, conflict labelling, blind ceiling, surface-heuristic baseline
├── train_reward.py     training, evaluation, swap test (held-out prompt)
└── train_bpe.py        stand-in vocabulary for the reward corpus (wrapper over llm.bpe_trainer)
```
Tokenizer: the shared `src/llm/tokenizer.py`. C++ parity: `tests/test_cpp_parity.py`.

## Running it

Run from the repository root.

```bash
python -m llm.reward.train_bpe --out data/reward_vocab.tok                                   # stand-in vocab, unless you have the team's vocab.tok
python -m llm.reward.train_reward --vocab data/reward_vocab.tok                              # FiLM (main result)
python -m llm.reward.train_reward --vocab data/reward_vocab.tok --conditioning concat        # baseline
python -m llm.reward.train_reward --vocab data/reward_vocab.tok --ablate-condition           # negative control
TOK_CLI=... TOK_VOCAB=... python -m pytest tests/test_cpp_parity.py                          # C++ parity
```

Roughly two minutes per run on CPU.
