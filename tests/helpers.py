"""Shared test helpers (stand-in vocabulary, legacy checkpoint format, small model config, surface baseline)."""
from __future__ import annotations

import re


# ------------------------------------------------------------------ vocab
def write_standin_vocab(path: str, vocab_size: int = 1722) -> str:
    """Byte-level BPE trained on the reward-model corpus, written in the team's .tok format."""
    from llm.bpe_trainer import train_bpe, write_tok
    from llm.reward.preferences import corpus_for_tokenizer
    write_tok(train_bpe(corpus_for_tokenizer(), vocab_size, min_count=1), path)
    return path


# ------------------------------------------------------------------ checkpoints
def to_checkpoint_format(state_dict: dict) -> dict:
    """model.state_dict() -> the key naming used by pretrain_model.pt (W_q/W_k/W_v/W_o)."""
    out = {}
    for k, v in state_dict.items():
        for new, old in ((".attn.q_proj.", ".attn.W_q."), (".attn.k_proj.", ".attn.W_k."),
                         (".attn.v_proj.", ".attn.W_v."), (".attn.o_proj.", ".attn.W_o.")):
            k = k.replace(new, old)
        out[k] = v.detach().clone()
    return out


def small_lm_config(**kw):
    from llm.model_config import ModelConfig
    base = dict(vocab_size=1722, hidden_dim=64, num_query_heads=4, num_kv_heads=2, head_dim=16,
                intermediate_dim=128, num_layers=2, max_position_embeddings=256, yarn_original_max_position=256)
    base.update(kw)
    return ModelConfig(**base)


# ------------------------------------------------------------------ reward-data baseline
def surface_heuristic_accuracy(pairs) -> float:
    """A ~5-line rule (character length + contraction count) -- no learning at all."""
    def contractions(t: str) -> int:
        return len(re.findall(r"'(?:re|ve|ll|s|t|m|d)\b", t))

    def formality(t: str):
        w = t.split()
        return (-contractions(t), sum(map(len, w)) / max(len(w), 1))

    def ok(p) -> bool:
        if p.condition == "helpful":
            return len(p.chosen) > len(p.rejected)
        if p.condition == "concise":
            return len(p.chosen) < len(p.rejected)
        return formality(p.chosen) > formality(p.rejected)

    return sum(ok(p) for p in pairs) / max(len(pairs), 1)
