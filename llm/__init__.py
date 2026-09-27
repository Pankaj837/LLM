"""Squad-1 LLM stack: GQA transformer + RoPE / Dynamic NTK / YaRN + BPE tokenizer + pipeline + trainer.

Common entry points are re-exported here; everything else lives in its component module
(see docs/ARCHITECTURE.md).
"""
from .model import TransformerModel
from .model_config import ModelConfig
from .pipeline import LLMPipeline
from .tokenizer import BPETokenizer

__all__ = ["ModelConfig", "TransformerModel", "LLMPipeline", "BPETokenizer"]
__version__ = "0.1.0"
