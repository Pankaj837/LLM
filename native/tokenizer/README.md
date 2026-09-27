# Custom Byte-Level BPE Tokenizer & Pipeline

A high-performance C++ byte-level BPE tokenizer and binary dataset compilation pipeline for LLM pretraining.

---

## 1. Directory & Corpus Setup

Place your training data files in the project workspace (e.g. in the root directory or inside a `data/` folder).

### Supported Corpus Formats:
- **Plain Text (`.txt`)**: Raw text file where each line is treated as a document.
- **JSON Lines (`.jsonl`)**: JSONL file where each line is a JSON object. The dataset parser automatically extracts text from keys like `"text"`, `"content"`, `"document"`, or `"body"`.

*Example paths:*
- `data/my_corpus.txt`
- `drive-download-20260907T180325Z-1-001/qbio_cleaned.jsonl`
- `"Earth and Environment Processed Dataset/processed_dataset/cleaned_corpus.txt"`

---

## 2. Building the Project with CMake

### What does CMake do?
CMake configures the build system, scans source files in `src/` and `examples/`, links the core C++ `tokenizer_lib`, and generates Makefile targets for building executables inside the `build/` directory.

### Build Steps:
```bash
# 1. Configure the build system (creates build/ directory):
cmake -B build

# 2. Compile source code into C++ executables:
cmake --build build
```

This generates four primary executables inside `build/`:
- `prepare_dataset`: Trains BPE vocabulary and compiles binary datasets (`train.bin`, `val.bin`).
- `dump_vocab`: Displays the learned vocabulary table (ID <-> Token mapping).
- `encode_text`: Tokenizes custom input text and tests roundtrip decoding.
- `train_example`: Simple example to train a BPE model on custom text strings.

---

## 3. End-to-End Workflow

### Step A: Prepare Dataset & Train BPE Vocabulary (`prepare_dataset`)

Trains a byte-level BPE vocabulary of a specified target size and compiles `train.bin` / `val.bin` files for model training.

```bash
# Syntax:
./build/prepare_dataset <path_to_corpus> <vocab_output.tok> <target_vocab_size>

# Example (.jsonl file):
./build/prepare_dataset drive-download-20260907T180325Z-1-001/qbio_cleaned.jsonl vocab_qbio.tok 4000

# Example (.txt file with spaces in directory name):
./build/prepare_dataset "Earth and Environment Processed Dataset/processed_dataset/cleaned_corpus.txt" vocab_earth.tok 4000
```

### Step B: Inspect Learned Vocabulary Tokens (`dump_vocab`)

Inspect all token IDs, token types (Byte, Special, Merge), Base64 encodings, and decoded subword strings:

```bash
./build/dump_vocab vocab_qbio.tok
```

### Step C: Tokenize & Decode Custom Text (`encode_text`)

Encode any text string into a list of token IDs and verify roundtrip decoding:

```bash
./build/encode_text vocab_qbio.tok "The multisite phosphorylation cycle regulates cell signaling."
```

### Step D: Cleanup Generated Files (`clean_data`)

Remove all generated dataset binaries (`*.bin`) and vocabulary dictionaries (`*.tok`) without using `rm`:

```bash
cmake --build build --target clean_data
```


