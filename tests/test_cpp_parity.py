"""C++ <-> Python tokenizer parity (component C5). SKIPPED unless the C++ side is provided.

    TOK_CLI=/path/to/encode_cli TOK_VOCAB=/path/to/vocab.tok python -m pytest tests/test_cpp_parity.py

``encode_cli`` contract (from the tokenizer team's harness): ``encode_cli <vocab.tok>`` reads text lines on
stdin and prints, per line, the space-separated token ids. Until this passes on the team's real vocabulary,
parity with the C++ tokenizer is UNVERIFIED (see docs/COMPONENTS.md, C5).
"""
import os
import subprocess

import pytest

from llm.tokenizer import BPETokenizer

CLI, VOCAB = os.environ.get("TOK_CLI"), os.environ.get("TOK_VOCAB")
pytestmark = pytest.mark.skipif(not (CLI and VOCAB and os.path.isfile(CLI) and os.path.isfile(VOCAB)),
                                reason="set TOK_CLI and TOK_VOCAB to run C++/Python tokenizer parity")

CASES = [
    "The quick brown fox jumps over the lazy dog.",
    "Hello,   world!!!  Multiple   spaces here",
    "I've got 42 apples and he'd say they're fine",
    "Grind the beans right before you brew.",
    "numbers 12345 mixed99with letters",
    "punctuation!?!?...---***",
    "'standalone quote and 'tis",
    "trailing spaces here   ",
    "cafe resume naive",
    "café résumé 日本語 \U0001f600",
    "a",
]


def test_ids_match_the_cpp_encoder_exactly():
    tok = BPETokenizer.from_file(VOCAB)
    out = subprocess.run([CLI, VOCAB], input="\n".join(CASES), capture_output=True, text=True, check=True)
    lines = out.stdout.strip("\n").split("\n")
    assert len(lines) == len(CASES)
    for case, cpp in zip(CASES, lines):
        assert " ".join(map(str, tok.encode(case, allowed_special=set()))) == cpp.strip(), case


def test_roundtrip_is_lossless_on_the_real_vocabulary():
    tok = BPETokenizer.from_file(VOCAB)
    for case in CASES + ["", "\ttab\nnewline"]:
        assert tok.decode(tok.encode(case, allowed_special=set())) == case
