"""scripts/analyze_checkpoint.py: loss summary, next-token metrics, repetition measures and the end-to-end run."""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import analyze_checkpoint as ac  # noqa: E402
from llm.checkpoint import save_checkpoint  # noqa: E402
from llm.data import write_bin  # noqa: E402
from llm.model import TransformerModel  # noqa: E402
from llm.model_config import ModelConfig  # noqa: E402
from llm.tokenizer import BPETokenizer  # noqa: E402

VOCAB = ROOT / "tokenizers" / "vocab32k_v2" / "vocab32k.tok"


def tiny_model():
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=32000, hidden_dim=64, num_query_heads=4, num_kv_heads=2, head_dim=16,
                      intermediate_dim=128, num_layers=2, max_position_embeddings=512, yarn_original_max_position=512)
    return TransformerModel(cfg).eval(), cfg


def write_metrics(path: Path):
    rows = []
    for s in range(10, 1010, 10):
        r = {"step": s, "loss": 4.0 - s / 500, "grad_norm": 9.0 if s == 500 else 0.3, "tokens_per_s": 100.0, "peak_mem_gb": 1.0}
        if s % 250 == 0:
            r["val_loss"] = 3.9 - s / 500
        rows.append(r)
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def test_loss_analysis(tmp_path):
    write_metrics(tmp_path / "m.jsonl")
    s, curve = ac.loss_analysis(str(tmp_path / "m.jsonl"), tokens_per_step=1000, tail_steps=500)
    assert s["val_points"] == 4 and s["val_best_step"] == 1000 and s["val_loss_best"] == pytest.approx(1.9)
    assert s["val_change_per_1k_steps_at_end"] == pytest.approx(-2.0)        # -1.0 over the last 500 steps
    assert s["tokens_seen"] == 1_000_000 and s["grad_norm_spikes"] == 1 and s["all_finite"]
    assert len(curve) == 100 and curve[-1][3] == pytest.approx(1.9)


def test_repetition_measures():
    assert ac.repeated_ngram_share([1, 2, 3, 4, 5, 6]) == 0.0
    assert ac.repeated_ngram_share([1, 2, 3, 4] * 5) == pytest.approx(1 - 4 / 17)
    assert ac.distinct([1, 2, 1, 2, 1], n=2) == pytest.approx(0.5)


def test_next_token_metrics_match_direct_computation(tmp_path):
    model, cfg = tiny_model()
    ids = np.random.default_rng(0).integers(0, 32000, size=2 * 64 + 1).astype(np.uint16)
    write_bin(str(tmp_path / "d.val.bin"), ids, 32000, 31936)
    r = ac.next_token_metrics(model, str(tmp_path / "d.val.bin"), 64, 5, "cpu")
    x = torch.tensor(ids[:128].astype(np.int64)).view(2, 64)
    y = torch.tensor(ids[1:129].astype(np.int64)).view(2, 64)
    with torch.no_grad():
        logp = F.log_softmax(model(x).float(), -1)
    ce = float(-logp.gather(-1, y.unsqueeze(-1)).mean())
    assert r["tokens"] == 128 and r["cross_entropy"] == pytest.approx(ce, rel=1e-5)
    assert r["perplexity"] == pytest.approx(math.exp(ce), rel=1e-5)
    assert 0 <= r["top1"] <= r["top5"] <= 1 and 0 <= r["calibration_error"] <= 1


def test_end_to_end(tmp_path):
    model, cfg = tiny_model()
    tok = BPETokenizer.from_file(str(VOCAB))
    vsha = BPETokenizer.file_sha256(str(VOCAB))
    write_metrics(tmp_path / "metrics.jsonl")
    ck_a, ck_b = tmp_path / "a.pt", tmp_path / "b.pt"
    save_checkpoint(str(ck_a), model, cfg, step=1000, vocab_sha256=vsha, untrained=True)
    with torch.no_grad():
        model.lm_head.weight.add_(0.01) if hasattr(model, "lm_head") else next(model.parameters()).add_(0.01)
    save_checkpoint(str(ck_b), model, cfg, step=750, vocab_sha256=vsha, untrained=True)
    text = "The beam carries a load. The stress is the force divided by the area." * 40
    ids = []
    for _ in range(4):
        ids += tok.encode(text, allowed_special=set()) + [tok.eos_id]
    write_bin(str(tmp_path / "d.val.bin"), np.array(ids, dtype=np.uint16), 32000, tok.eos_id)
    (tmp_path / "cfg.json").write_text(json.dumps({"train": {"seq_len": 64, "batch_size": 2, "grad_accum": 1}}))
    out = tmp_path / "res" / "an"
    args = ["--ckpt", str(ck_a), "--compare-ckpt", str(ck_b), "--metrics", str(tmp_path / "metrics.jsonl"),
            "--config", str(tmp_path / "cfg.json"), "--vocab", str(VOCAB), "--val", f"dom={tmp_path / 'd.val.bin'}",
            "--evals", str(ROOT / "evals"), "--out", str(out), "--windows", "2", "--per-domain", "2", "--new-tokens", "8",
            "--probe-tokens", "6", "--mcq", "4", "--device", "cpu"]
    assert ac.main(args) == 0
    res = json.loads(Path(str(out) + ".json").read_text())
    assert res["loss"]["val_best_step"] == 1000 and res["next_token"]["dom"]["tokens"] == 128
    g = res["generation"]
    assert g["probe_summary"]["greedy_identical_twice"] == len(g["probes"]) and g["order_consistency"]["items"] == 4
    assert {"rep4", "rep4_sampled", "loops", "loops_sampled"} <= set(g["domains"]["dom"])
    i = res["integrity"]
    assert i["checkpoints"]["a.pt"]["all_finite"] and i["checkpoints"]["a.pt"]["vocab_matches"]
    assert i["checkpoints"]["a.pt"]["val_loss_in_metrics"] == pytest.approx(1.9)
    assert i["pair"]["same_config"] and i["pair"]["relative_l2_difference"] > 0
    assert i["forward"]["repeat_max_abs_diff"] == 0.0
    assert Path(str(out) + "_curve.csv").read_text().startswith("step,train_loss")


def test_val_overlap_and_tag_count(tmp_path):
    import check_val_overlap as cvo
    tok = BPETokenizer.from_file(str(VOCAB))
    enc = lambda s: tok.encode(s, allowed_special=set())   # noqa: E731
    shared = "The beam carries a uniform load along its whole length and rests on two simple supports at the ends."
    only_val = "A violet amphibian sings opera beneath the frozen lakes of a distant imaginary moon every single winter night."
    tagged = "Some text here. Explanation: Benchmark source: data/CoT/math50k_camel.json and more words follow it now."
    eos = tok.eos_id
    val = enc(shared) + [eos] + enc(only_val) + [eos]
    train = enc(tagged) + [eos] + enc(shared) + [eos] + enc(tagged) + [eos]
    write_bin(str(tmp_path / "d.val.bin"), np.array(val, dtype=np.uint16), 32000, eos)
    write_bin(str(tmp_path / "d.train.bin"), np.array(train, dtype=np.uint16), 32000, eos)
    out = tmp_path / "o.json"
    # chunk 11: a document and a tag can straddle chunk edges
    assert cvo.main(["--bins", str(tmp_path), "--domains", "d", "--vocab", str(VOCAB), "--out", str(out), "--n", "8",
                     "--windows", "20", "--chunk", "11", "--tags", "Benchmark source: data/CoT"]) == 0
    res = json.loads(out.read_text())
    r = res["overlap"]["d"]
    assert r["val_docs"] == 2 and r["exact_docs"] == 1 and 0 < r["window_share"] < 1
    # the shared sentence occurs once in training, so every matched passage counts as rare (a copy), none as boilerplate
    assert r["window_share_rare"] == r["window_share"] and r["window_share_common"] == 0
    assert res["tags"]["Benchmark source: data/CoT"]["d"] == 2


def test_sampling_is_fixed_and_repeatable():
    """Same seed -> same tokens, whatever ran before; different seeds differ; settings follow SAMPLING."""
    model, cfg = tiny_model()
    prompt = [5, 6, 7, 8]
    a = ac.sample(model, prompt, 30, eos=-1, device="cpu", seed=1)
    torch.manual_seed(12345)
    torch.rand(1000)                  # the global random state must not matter
    b = ac.sample(model, prompt, 30, eos=-1, device="cpu", seed=1)
    c = ac.sample(model, prompt, 30, eos=-1, device="cpu", seed=2)
    assert a == b and len(a) == 30 and a != c
    assert ac.SAMPLING == {"temperature": 0.8, "top_k": 50, "top_p": 0.95, "seeds": [1, 2, 3]}
    # top-k 1 is greedy decoding
    g = ac.sample(model, prompt, 10, eos=-1, device="cpu", seed=3, top_k=1)
    x = torch.tensor([prompt])
    greedy = model.generate(x, max_new_tokens=10, temperature=0.0)[0, len(prompt):].tolist()
    assert g == greedy


def test_val_overlap_counts_boilerplate(tmp_path):
    """A passage repeated in many training documents counts as common, not as a copy of the validation text."""
    import check_val_overlap as cvo
    tok = BPETokenizer.from_file(str(VOCAB))
    enc = lambda s: tok.encode(s, allowed_special=set())   # noqa: E731
    header = "Licensed under the Apache License, Version 2.0 (the License); you may not use this file except in compliance."
    eos = tok.eos_id
    val = enc(header) + [eos]
    train = [i for k in range(12) for i in enc(header) + enc(f" file {k}") + [eos]]
    write_bin(str(tmp_path / "d.val.bin"), np.array(val, dtype=np.uint16), 32000, eos)
    write_bin(str(tmp_path / "d.train.bin"), np.array(train, dtype=np.uint16), 32000, eos)
    out = tmp_path / "o.json"
    assert cvo.main(["--bins", str(tmp_path), "--domains", "d", "--vocab", str(VOCAB), "--out", str(out), "--n", "8",
                     "--windows", "10", "--tags"]) == 0
    r = json.loads(out.read_text())["overlap"]["d"]
    assert r["window_share"] > 0 and r["window_share_common"] == r["window_share"] and r["window_share_rare"] == 0
