# Native (C++) tokenizer

`tokenizer/` is the tokenizer team's own C++ byte-level BPE implementation, as delivered — source unchanged
except for one addition: `tests/encode_cli.cpp`, a small batch harness added here (not by the tokenizer team)
so `tests/test_cpp_parity.py` in the Python package can drive it. Everything else — `include/`, `src/`,
`examples/`, `tests/test_tokenizer.cpp`, `CMakeLists.txt`, `README.md` — is exactly what was received.

This is a **separate, independent implementation** from `src/llm/tokenizer.py`. The Python port is a
line-for-line re-implementation of this C++ code (so the whole stack can train and run in PyTorch without
shelling out to a binary per token); they are proven to produce identical output — see
`docs/CPP_TOKENIZER_PARITY.md` for the real, executed verification and its results.

## Build

Needs a C++17 compiler. No external dependencies.

```bash
cd native/tokenizer
cmake -B build && cmake --build build
```
Or without CMake (any C++17 compiler):
```bash
# g++ / clang
g++ -std=c++17 -O2 -Iinclude -o build/prepare_dataset examples/prepare_dataset.cpp src/*.cpp
# MSVC (from a Developer Command Prompt / after vcvars64.bat)
cl /std:c++17 /EHsc /O2 /Iinclude examples\prepare_dataset.cpp src\*.cpp /Febuild\prepare_dataset.exe
```
Repeat for `examples/dump_vocab.cpp`, `examples/encode_text.cpp`, `examples/train.cpp`, `tests/test_tokenizer.cpp`,
and `tests/encode_cli.cpp` (the parity harness), linking each against the same `src/*.cpp` / object files.

## Verified working (see docs/CPP_TOKENIZER_PARITY.md)
- `test_tokenizer` (the team's own suite): 46/46 checks pass.
- `prepare_dataset` produces `.bin` files with the exact header this repo's `llm.data` module expects
  (magic `TOK1` / `0x544F4B31`, 36 bytes) — confirms `src/llm/data.py`'s format assumption against the real tool.
- `encode_cli` vs `src/llm/tokenizer.py`: identical token ids on every test case (punctuation, contractions,
  multi-byte UTF-8, digits, whitespace runs, emoji) and lossless round-trip, both on a vocabulary trained by
  the real `prepare_dataset` tool.
