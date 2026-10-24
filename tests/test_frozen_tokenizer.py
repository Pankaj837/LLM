"""The frozen tokenizer file must match the hash recorded in tokenizers/FROZEN.md, and keep the layout the pipeline relies on."""
import re
from pathlib import Path

from llm.tokenizer import BPETokenizer

ROOT = Path(__file__).resolve().parents[1]


def test_frozen_vocabulary_matches_its_recorded_hash_and_layout():
    text = (ROOT / "tokenizers" / "FROZEN.md").read_text(encoding="utf-8")
    sha = re.search(r"SHA-256 \| `([0-9a-f]{64})`", text).group(1)
    directory = re.search(r"Directory \| `(tokenizers/[^`]+)`", text).group(1)
    path = ROOT / directory / "vocab32k.tok"
    assert BPETokenizer.file_sha256(str(path)) == sha
    tok = BPETokenizer.from_file(str(path))
    assert tok.n_vocab == 32000 and tok.eos_id == 31936 and len(tok.special_to_id) == 64
    assert tok.id_to_special[tok.eos_id] == "<|endoftext|>"


def test_the_unfrozen_experiment_is_not_the_frozen_file():
    frozen = (ROOT / "tokenizers" / "vocab32k_v2" / "vocab32k.tok").read_bytes()
    other = (ROOT / "tokenizers" / "vocab32k_digits_v2" / "vocab32k.tok").read_bytes()
    assert frozen != other
