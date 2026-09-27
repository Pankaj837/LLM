# Archive

Empty on purpose. This folder exists so deprecated/unused/uncertain files have somewhere to go without being
deleted — an audit of the repo at restructuring time found nothing inside it that qualified. Two related
decisions, made explicit rather than silently applied:

## `src/llm/yarn.py` was considered and kept in place, not archived
It is documented in its own docstring as a deprecated shim (`YaRNRotaryEmbedding` is now `llm.rope.RotaryEmbedding`).
By the letter of the restructuring instructions it looks like an archive candidate. It was **kept** because it is
not dead code: it is the only backward-compatibility path for any code (inside or outside this repo) still doing
`from llm.yarn import YaRNRotaryEmbedding`. Moving it out of the package would silently break that import instead
of just deprecating it. If nothing turns out to depend on it, it can be deleted outright in a later pass — that is
a smaller, more honest change than parking it here while still needing it importable.

## What is *not* inside this repository, and was left alone
This repo (`LLM_Squad1_Final/`) was built by consolidating several sibling directories that sit next to it, not
inside it:
- `../llm/` — the originally delivered, unpackaged codebase (pre-review). Fully superseded; kept where it was found.
- `../llm_review/` — an earlier, since-superseded review workspace (per-owner analysis, before the ask shifted to
  "component by component, not person by person"). Superseded by this repo's `docs/`.
- `../New/` — raw uploaded status reports and zips (RoPE/GQA/Dynamic-NTK reports, an early conditional-reward
  delivery) that were read and incorporated; the source files themselves were left in place.
- `../Biology Dataset.zip`, `../Earth and Environment Processed Dataset.zip` — real datasets (hundreds of MB);
  intentionally kept outside git, see `data/README.md`.

These were not pulled into this repo's `archive/` because doing so would import a large amount of unrelated,
already-superseded material into what is meant to be a clean deliverable. If you want them cleaned up, that is a
separate, explicit decision on the parent directory — flagging it here rather than acting on it.
