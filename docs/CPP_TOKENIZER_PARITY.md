# C++ tokenizer parity — verified

**This closes the last major open item from `docs/REVIEW_FINDINGS.md` (F-22 / C5): C++/Python tokenizer parity was
unverified because the C++ source, a compiled binary and a real `vocab.tok` were not available. All three now are,
and parity is confirmed by actually building and running the C++ code — not by inspection.**

## What was delivered

The tokenizer team sent the full C++ source (`native/tokenizer/` in this repo): a byte-level BPE trainer/encoder/
decoder, a dataset-compilation pipeline (`prepare_dataset`), inspection tools (`dump_vocab`, `encode_text`), and the
team's own test suite (`tests/test_tokenizer.cpp`) — CMake-based, C++17, no external dependencies. Not a pre-built
binary: source only, meant to be compiled locally.

## Build environment problem, found and worked around

The machine's installed MinGW-w64 g++ (MSYS2, `C:\msys64\mingw64`) has a **broken installation**: its C++ standard
library headers are missing core files (`bits/char_traits.h` and others), so nothing compiles with it — not specific
to this codebase. `cmake` is also not installed. Neither is fixable without administrative package management, which
was out of scope here.

**Workaround used:** this machine has Visual Studio 2019 Build Tools already installed
(`C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools`), with a working `cl.exe` (MSVC 19.29). All eight
library source files, all four example executables, the team's test suite, and one new file
(`native/tokenizer/tests/encode_cli.cpp` — a small batch-mode harness added so the parity test in this repo can
drive the tokenizer via stdin/stdout; not part of the original delivery) were compiled and linked directly with
`cl /std:c++17`, bypassing CMake entirely. Full command sequence in `native/README.md`.

## Results — actually executed, not simulated

**The team's own test suite**, compiled and run as delivered:
```
46/46 checks passed.
All tests passed.
```
Covers pretokenization, the byte vocabulary, merge-token ids, special tokens, save/load round-trip, BPE training,
encode/decode round-trip, UTF-8 sanitization, and the binary dataset format.

**A real vocabulary and dataset**, produced by the real `prepare_dataset` tool (not synthetic) on 110,920 documents
(15.6 MB) of real text — the cleaned corpus from the Earth & Environment delivery (`docs/EARTH_ENVIRONMENT_SMOKE_TEST.md`),
chosen simply because it was already on hand as ready-to-tokenize plain text; this run is independent of that
dataset's own (differently-formatted) `.bin` files:
```
Trained Vocabulary Size: 8001   (8000 merges + <|endoftext|>)
Train Tokens Exported: 3,743,862 -> train.bin
Val Tokens Exported:     197,716 -> val.bin
Header Magic Verified: 0x544f4b31
```

**This repo's own `.bin` reader (`src/llm/data.py`) opened the real files with zero changes needed:**
```
TokHeader(magic=1414482737, version=1, vocab_size=8001, eos_id=8000, total_tokens=3743862, split=0, reserved=0)  # train.bin
TokHeader(magic=1414482737, version=1, vocab_size=8001, eos_id=8000, total_tokens=197716,  split=1, reserved=0)  # val.bin
```
`1414482737 == 0x544F4B31` — the exact `TOK1` magic this repo's spec expects. This also resolves a previously open
question (`docs/COMPONENTS.md`, C10): the header's `split` field is `0` for train and `1` for val, confirming the
convention this repo already assumed (`SPLIT_TRAIN = 0`, `SPLIT_VAL = 1`) is the real one, not a guess.

**Tokenizer parity** (`tests/test_cpp_parity.py`, run against the real `encode_cli` binary and the real vocabulary above):
```
tests/test_cpp_parity.py::test_ids_match_the_cpp_encoder_exactly PASSED
tests/test_cpp_parity.py::test_roundtrip_is_lossless_on_the_real_vocabulary PASSED
2 passed in 4.41s
```
Every test case — plain text, multiple/trailing spaces, contractions (`I've`, `he'd`, `they're`), digits mixed with
letters, punctuation runs, a standalone apostrophe, multi-byte UTF-8 (café, résumé, 日本語), and an emoji — produced
**identical token ids** from the C++ encoder and the Python port (`src/llm/tokenizer.py`), and round-tripped losslessly
through the real vocabulary.

## A real bug found and fixed while running this

The first attempt at `test_ids_match_the_cpp_encoder_exactly` hung indefinitely. Cause: `subprocess.run(..., text=True)`
on Windows defaults to the console's codepage (cp1252), not UTF-8; writing the emoji/Japanese test case through that
pipe failed inside the subprocess machinery in a way that left the child process blocked on stdin forever instead of
raising cleanly. Fixed by passing `encoding="utf-8"` and a `timeout=30` explicitly. This is now how the test behaves
for anyone who runs it in the future — without this fix, the very first real run of this test on Windows would have
hung rather than failed loudly.

## What this does and doesn't prove
- **Proves:** the Python tokenizer port is a faithful re-implementation of the real C++ tokenizer — same ids, same
  round-trip behaviour, on real text and a realistically-sized (8001-token) trained vocabulary. The `.bin` format
  this repo assumed is exactly what the real tool produces.
- **Does not prove:** performance parity (C++ is presumably much faster; not measured here), or that every possible
  Unicode edge case matches (the test set is representative, not exhaustive — `tests/test_c5_tokenizer.py` already
  fuzzes the Python side extensively; extending the fuzz set through the C++ side too is a natural next step).
- The vocabulary and `.bin` files produced during this verification are not committed (consistent with this repo's
  policy of not committing generated data — see `data/README.md`); their SHA-256 hashes are recorded here for
  traceability: `vocab.tok` `71c7465d…79132e`, `train.bin` `505e2764…41fb849`, `val.bin` `fcae529d…e86f8494`.

## Reproduce
```bash
cd native/tokenizer
# build (see native/README.md for the exact compiler command on your platform)
./build/test_tokenizer                       # or test_tokenizer.exe on Windows
./build/prepare_dataset <a text corpus> vocab.tok 8000
cd ../..
TOK_CLI=native/tokenizer/build/encode_cli TOK_VOCAB=native/tokenizer/vocab.tok \
    python -m pytest tests/test_cpp_parity.py -v
```
