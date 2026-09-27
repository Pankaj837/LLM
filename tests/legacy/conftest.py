"""Fixtures so the team's original integration tests (written as a script with
hand-passed arguments) also run under pytest, unchanged."""
import pytest

from llm.model_config import ModelConfig
from llm.tokenizer import SimpleTokenizer


@pytest.fixture
def config():
    return ModelConfig()


@pytest.fixture
def tok():
    return SimpleTokenizer(vocab_size=8000)


@pytest.fixture
def token_ids(tok):
    return tok.encode("Hello, world! This is a test.")
