"""
LLMPipeline — End-to-end text generation pipeline.

Connects:
    User Prompt → BPETokenizer → TransformerModel (with RoPE/YaRN & GQA) →
    KV-Cached Sampling → Detokenization → Generated Text
"""

from __future__ import annotations

import os
from typing import Optional

import torch

from .checkpoint import read_checkpoint
from .model_config import ModelConfig
from .model import TransformerModel
from .tokenizer import BPETokenizer


class LLMPipeline:
    """High-level pipeline connecting tokenizer, model, and generation."""

    def __init__(
        self,
        model: TransformerModel,
        tokenizer: BPETokenizer,
        config: Optional[ModelConfig] = None,
        device: torch.device = torch.device("cpu"),
    ):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.config = config or model.config
        self.device = device
        self.model.eval()

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: str,
        vocab_path: str,
        config: Optional[ModelConfig] = None,
        device: str = "cpu",
    ) -> LLMPipeline:
        """Loads model weights and tokenizer from files.

        Fails loudly: a missing checkpoint or vocab, a key/shape/config mismatch, a vocabulary
        file that differs from the one the checkpoint was trained with, or a tokenizer whose ids
        do not fit the embedding table all raise, instead of silently producing a randomly
        initialised model or a byte-level tokenizer. Untrained / legacy checkpoints emit a warning.
        """
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
        tokenizer = BPETokenizer.from_file(vocab_path)          # FileNotFoundError if missing

        if config is None:                                       # prefer the config stored in the checkpoint
            _, stored_cfg, _, _ = read_checkpoint(checkpoint_path)
            config = stored_cfg or ModelConfig()
        model = TransformerModel(config)
        model.load_pretrained(
            checkpoint_path, strict=True, expected_vocab_sha256=BPETokenizer.file_sha256(vocab_path)
        )
        if tokenizer.n_vocab > config.vocab_size:
            raise ValueError(
                f"tokenizer emits ids up to {tokenizer.n_vocab - 1} but the model embedding "
                f"has only {config.vocab_size} rows"
            )
        tokenizer.vocab_size = config.vocab_size
        return cls(model=model, tokenizer=tokenizer, config=config, device=torch.device(device))

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = 0.9,
        use_cache: bool = True,
    ) -> str:
        """Generates text continuation from a prompt string.

        Args:
            prompt: initial text prompt
            max_new_tokens: maximum number of new tokens to generate
            temperature: sampling temperature (0.0 for greedy)
            top_k: top-k filtering
            top_p: nucleus sampling threshold
            use_cache: whether to use KV cache for fast generation

        Returns:
            Generated text string (prompt + continuation)
        """
        self.model.eval()
        # allowed_special=set(): user text must never be able to inject control tokens (e.g. EOS).
        input_ids = self.tokenizer.encode(prompt, allowed_special=set())
        if not input_ids:
            raise ValueError("prompt encodes to zero tokens")
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)

        eos_id = self.tokenizer.eot_token_id

        output_tensor = self.model.generate(
            token_ids=input_tensor,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            eos_token_id=eos_id,
            use_cache=use_cache,
        )

        output_ids = output_tensor[0].tolist()
        return self.tokenizer.decode(output_ids)
