"""Shared pytest fixtures.

Environment variables (all optional):
  LLM_CKPT               path to a checkpoint to inspect (default: none -> checkpoint-artefact tests skip)
  LLM_REQUIRE_TRAINED=1  make the checkpoint-artefact test FAIL if LLM_CKPT looks untrained (release gate)
  BIO_ZIP                path to 'Biology Dataset.zip' for the real-data smoke test (default: skip)
  TOK_CLI / TOK_VOCAB    C++ ``encode_cli`` binary and its vocab.tok for C++/Python tokenizer parity
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from helpers import write_standin_vocab

ROOT = Path(__file__).resolve().parent.parent


def pytest_report_header(config):
    return f"LLM Squad-1 suite | root={ROOT}"


@pytest.fixture(scope="session")
def ckpt_path() -> Path:
    p = os.environ.get("LLM_CKPT")
    if not p or not Path(p).is_file():
        pytest.skip("set LLM_CKPT to a checkpoint file to run checkpoint-artefact tests")
    return Path(p)


@pytest.fixture(scope="session")
def vocab_path(tmp_path_factory) -> str:
    """A stand-in .tok vocabulary (1722 tokens) trained on the reward-model corpus.
    (The team's own vocab.tok is produced by the C++ trainer, which is not part of this repo.)"""
    return write_standin_vocab(str(tmp_path_factory.mktemp("vocab") / "vocab.tok"))
