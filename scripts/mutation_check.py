"""Mutation check: deliberately break the code and confirm the test-suite notices.

    python scripts/mutation_check.py            # runs every mutant against its target tests (~2-4 min)

Each mutant is applied to a temporary COPY of the repository (the working tree is never modified).
A surviving mutant means a behaviour that nothing tests. ``EQUIVALENT`` mutants change the source but not the
behaviour (documented below) and are expected to survive.
"""
import os
import shutil
import subprocess
import sys
import tempfile

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

# (description, file, old, new, tests that must fail)
MUTANTS = [
    ("YaRN logit multiplier inverted", "llm/rope.py",
     "return (0.1 * math.log(self.cfg.yarn_scale_factor) + 1.0) ** 2",
     "return 1.0 / (0.1 * math.log(self.cfg.yarn_scale_factor) + 1.0) ** 0.5", ["tests/test_c3_rope_yarn.py"]),
    ("YaRN ramp linear in wavelength again", "llm/rope.py",
     "ramp = ((torch.arange(dim // 2, dtype=torch.float32) - low) / (high - low)).clamp(0.0, 1.0)",
     "wl = 2 * math.pi / inv_freq; ramp = ((wl - length / cfg.yarn_beta) / (length / cfg.yarn_alpha - length / cfg.yarn_beta)).clamp(0.0, 1.0)",
     ["tests/test_c3_rope_yarn.py"]),
    ("YaRN scale factor ignored", "llm/rope.py",
     "return (inv_freq / cfg.yarn_scale_factor) * (1.0 - extrapolation) + inv_freq * extrapolation", "return inv_freq",
     ["tests/test_c3_rope_yarn.py"]),
    ("Dynamic NTK wrong exponent", "llm/dynamic_ntk.py",
     "exponent = float(self.head_dim) / float(self.head_dim - 2)", "exponent = float(self.head_dim) / float(self.head_dim)",
     ["tests/test_dynamic_ntk.py"]),
    ("RopeCache key ignores theta", "llm/rope_cache.py",
     "key = (seq_len, head_dim, theta, None if inv_freq is None else freq_tag, device, dtype, duplicate)", "key = (seq_len,)",
     ["tests/test_dynamic_ntk.py"]),
    ("Dynamic NTK never activates in RotaryEmbedding", "llm/rope.py",
     "if self._ntk is not None and theta != self.cfg.theta_base:", "if False:", ["tests/test_dynamic_ntk.py"]),
    ("Model passes no shared cos/sin", "llm/model.py",
     "                rope_cos_sin=rope_cos_sin,\n            )\n", "                rope_cos_sin=None,\n            )\n", ["tests/test_dynamic_ntk.py"]),
    ("Planned context ignores KV-cache capacity", "llm/model.py",
     "context_len = kv_caches[0].max_seq_len if kv_caches is not None else seq_len", "context_len = seq_len",
     ["tests/test_dynamic_ntk.py"]),
    ("Uncached decode lets the context grow each step", "llm/model.py",
     "logits = self.forward(generated, use_causal_mask=True, rope_context_len=max_total_len)",
     "logits = self.forward(generated, use_causal_mask=True)", ["tests/test_dynamic_ntk.py"]),
    ("GQA groups tiled instead of interleaved", "llm/gqa.py",
     "return x.repeat_interleave(n_rep, dim=1)", "return x.repeat(1, n_rep, 1, 1)", ["tests/test_c4_gqa_kvcache.py"]),
    ("GQA causal mask leaks the future", "llm/gqa.py",
     "causal_mask = k_idx <= (total_seq_len - seq_len + q_idx)", "causal_mask = k_idx <= (total_seq_len - seq_len + q_idx + 1)",
     ["tests/test_c4_gqa_kvcache.py", "tests/legacy/test_gqa.py"]),
    ("Finished rows keep generating", "llm/model.py",
     "finished.unsqueeze(-1), torch.full_like(next_token, eos_token_id), next_token",
     "torch.zeros_like(finished).unsqueeze(-1), torch.full_like(next_token, eos_token_id), next_token", ["tests/test_c6_generation.py"]),
    ("(B,T) attention mask not normalised", "llm/model.py",
     "if attention_mask is not None and attention_mask.dim() == 2:", "if False:", ["tests/test_c11_integration.py"]),
    ("Weights initialised at N(0,1)", "llm/model.py",
     "nn.init.normal_(module.weight, mean=0.0, std=0.02)\n\n    def forward", "nn.init.normal_(module.weight, mean=0.0, std=1.0)\n\n    def forward",
     ["tests/test_c1_config_blocks.py", "tests/test_train.py"]),
    ("Reward pooling reads the wrong token", "llm/reward/reward_model.py",
     "last_idx = lengths - 1 ", "last_idx = lengths - 2 ", ["tests/test_c9_reward_model.py"]),
    ("Checkpoint config check removed", "llm/checkpoint.py",
     "if cfg is not None and model_cfg is not None and cfg != model_cfg:", "if False:", ["tests/test_checkpoint.py"]),
    ("Checkpoint metadata not weights_only-safe", "llm/checkpoint.py",
     '"torch_version": str(torch.__version__)', '"torch_version": torch.__version__', ["tests/test_checkpoint.py"]),
    ("Bin header 32 bytes", "llm/data.py", 'HEADER_FMT = "<IIIIQIQ"', 'HEADER_FMT = "<IIIIQI"', ["tests/test_data_bin.py"]),
    ("Special-token injection via documents", "llm/data.py",
     "out.extend(tokenizer.encode(d, allowed_special=set()))", "out.extend(tokenizer.encode(d))", ["tests/test_e2e.py"]),
    ("Near-duplicate filter sampling by position", "llm/data.py",
     "return {h for h in hs if h % self.stride == 0}", "return set(list(hs)[::self.stride])", ["tests/test_e2e.py"]),
    ("Weight decay on norm weights", "llm/train.py",
     "no_decay = [p for p in model.parameters() if p.requires_grad and p.ndim < 2]", "no_decay = []", ["tests/test_train.py"]),
    ("Resume ignores optimizer state", "llm/train.py",
     "optimizer.load_state_dict(raw[\"optimizer\"])", "pass", ["tests/test_train.py"]),
    ("Pipeline skips vocabulary-hash check", "llm/pipeline.py",
     "checkpoint_path, strict=True, expected_vocab_sha256=BPETokenizer.file_sha256(vocab_path)", "checkpoint_path, strict=True",
     ["tests/test_e2e.py"]),
    ("Prompt EOS injection allowed", "llm/pipeline.py",
     "self.tokenizer.encode(prompt, allowed_special=set())", "self.tokenizer.encode(prompt)", ["tests/test_c7_c8_checkpoint_pipeline.py"]),
    ("BPE trainer picks the least frequent pair", "llm/bpe_trainer.py",
     "return (-c, vocab[pair[0]], vocab[pair[1]], pair)", "return (c, vocab[pair[0]], vocab[pair[1]], pair)", ["tests/test_bpe_trainer.py"]),
]

# Source changes that do not change behaviour (kept for the record; expected to survive):
EQUIVALENT = [
    "DynamicNTK threshold '<=' -> '<': at L == L_orig the scale is exactly 1, so theta = base either way",
    "Cached decode step drops rope_context_len: the default is the KV-cache capacity, which equals the value passed",
]


def main() -> int:
    survivors = []
    for name, rel, old, new, tests in MUTANTS:
        tmp = tempfile.mkdtemp(prefix="mut_")
        dst = os.path.join(tmp, "repo")
        shutil.copytree(REPO, dst, ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache", "runs", "data", "*.pt", ".git"))
        path = os.path.join(dst, rel)
        src = open(path, encoding="utf-8", newline="").read()
        if old not in src:
            print(f"[BAD PATTERN] {name}")
            survivors.append(name + " (pattern not found: update this script)")
            shutil.rmtree(tmp, ignore_errors=True)
            continue
        open(path, "w", encoding="utf-8", newline="").write(src.replace(old, new, 1))
        r = subprocess.run([sys.executable, "-m", "pytest", *tests, "-x", "-q", "-o", "addopts=", "-p", "no:cacheprovider"],
                           cwd=dst, capture_output=True, text=True, env={**os.environ, "PYTHONIOENCODING": "utf-8"})
        killed = r.returncode != 0
        first = next((ln for ln in r.stdout.splitlines() if ln.startswith(("FAILED", "ERROR"))), "")
        print(f"[{'KILLED  ' if killed else 'SURVIVED'}] {name:52s} {first[:80]}", flush=True)
        if not killed:
            survivors.append(name)
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"\nmutants: {len(MUTANTS)}  killed: {len(MUTANTS) - len(survivors)}  survived: {len(survivors)}")
    for s in survivors:
        print("  SURVIVOR:", s)
    print("documented equivalent mutants (expected to survive):")
    for e in EQUIVALENT:
        print("  -", e)
    return 1 if survivors else 0


if __name__ == "__main__":
    sys.exit(main())
