"""scripts/eval_science.py, scripts/check_leakage.py and the frozen question set in evals/."""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import build_eval_set as bes  # noqa: E402
import check_leakage as cl  # noqa: E402
import eval_science as es  # noqa: E402
from llm.checkpoint import save_checkpoint  # noqa: E402
from llm.data import write_bin  # noqa: E402
from llm.model import TransformerModel  # noqa: E402
from llm.model_config import ModelConfig  # noqa: E402
from llm.tokenizer import BPETokenizer  # noqa: E402

VOCAB = ROOT / "tokenizers" / "vocab32k_v2" / "vocab32k.tok"
EVALS = ROOT / "evals"


@pytest.fixture(scope="module")
def tok():
    return BPETokenizer.from_file(str(VOCAB))


def tiny_model():
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=32000, hidden_dim=64, num_query_heads=4, num_kv_heads=2, head_dim=16,
                      intermediate_dim=128, num_layers=2, max_position_embeddings=512, yarn_original_max_position=512)
    return TransformerModel(cfg).eval(), cfg


# ------------------------------------------------------------------ the frozen set itself
def test_question_set_is_well_formed_and_frozen():
    items = es.load_items(str(EVALS))
    assert len(items) > 2000
    for it in items:
        assert {"id", "domain", "level", "type", "question", "answer", "source", "licence"} <= set(it), it["id"]
        assert it["level"] in es.LEVELS, it["id"]
        if it["type"] == "mcq":
            assert 2 <= len(it["choices"]) and 0 <= it["answer"] < len(it["choices"]), it["id"]
            assert len(set(it["choices"])) == len(it["choices"]), it["id"]
            if "distractor_types" in it:
                assert it["distractor_types"][it["answer"]] == "correct", it["id"]
                assert it["distractor_types"].count("correct") == 1, it["id"]
        else:
            assert it["type"] == "numeric" and isinstance(it["answer"], (int, float)), it["id"]
            assert it.get("fewshot") in es.FEWSHOT_FILES, it["id"]
    assert es.manifest_problems(str(EVALS)) == [], "evals/ changed: re-run scripts/build_eval_set.py --manifest on purpose"
    shots = es.load_fewshot(str(EVALS))
    shot_qs = {s["question"] for v in shots.values() for s in v}
    assert not shot_qs & {it["question"] for it in items}          # worked examples are never test questions


def test_handwritten_right_answer_is_not_usually_the_longest_option():
    mcq = [it for it in es.load_items(str(EVALS)) if it["type"] == "mcq" and it["source"].startswith("hand-written")]
    longest = sum(max(range(len(it["choices"])), key=lambda i: len(it["choices"][i])) == it["answer"] for it in mcq)
    assert longest / len(mcq) <= 0.3, f"right answer is the longest option in {longest}/{len(mcq)} questions"


# ------------------------------------------------------------------ scoring pieces
def test_choice_logprobs_match_unpadded_computation(tok):
    model, cfg = tiny_model()
    ctx, choices = "Question: What is the chemical formula of water?\nAnswer:", ["H2O", "carbon dioxide gas", "NaCl"]
    got = es.choice_logprobs(model, tok, ctx, choices, "cpu", cfg.max_position_embeddings)
    c_ids = tok.encode(ctx, allowed_special=set())
    for g, ch in zip(got, choices):
        a_ids = tok.encode(" " + ch, allowed_special=set())
        logp = F.log_softmax(model(torch.tensor([c_ids + a_ids])).float(), -1)[0]
        ref = sum(float(logp.detach()[len(c_ids) - 1 + j, t]) for j, t in enumerate(a_ids))
        assert abs(g - ref) < 1e-4


class FavourToken(torch.nn.Module):
    """Stand-in model whose next-token distribution always prefers one token id."""

    def __init__(self, token_id, vocab=32000):
        super().__init__()
        self.t, self.v = token_id, vocab

    def forward(self, x):
        out = torch.zeros(x.shape[0], x.shape[1], self.v)
        out[..., self.t] = 10.0
        return out


def test_mcq_picks_the_choice_the_model_prefers_and_labels_the_error(tok):
    item = {"question": "q", "choices": ["alpha", "beta"], "answer": 0, "distractor_types": ["correct", "unit_error"]}
    alpha, beta = (tok.encode(" " + w, allowed_special=set())[0] for w in ("alpha", "beta"))
    assert alpha != beta
    r = es.score_mcq(FavourToken(beta), tok, item, "cpu", 64)
    assert r["pick_norm"] == 1 and not r["correct_norm"] and r["error"] == "unit_error" and r["chance"] == 0.5
    assert es.score_mcq(FavourToken(alpha), tok, item, "cpu", 64)["correct_norm"]


def test_parse_number_and_error_types():
    assert es.parse_number(" Natalia sold 48/2 = 24 clips. The answer is 1,234.\n\nQuestion: x", worked=True) == 1234
    assert es.parse_number(" 48 + 24 = 72 clips\n\nQuestion: 5", worked=True) is None       # strict: no 'The answer is'
    assert es.parse_number(" 20 m/s\n", worked=False) == 20
    assert es.parse_number(" 2.5e3 W", worked=False) == 2500
    assert es.parse_number(" the speed is high", worked=False) is None
    assert es.numeric_correct(7.86, 7.85, 0.02) and not es.numeric_correct(8.2, 7.85, 0.02)
    assert es.numeric_correct(72.0, 72, 0.0) and not es.numeric_correct(72.5, 72, 0.0)
    assert es.numeric_error(None, 5, 0.02, []) == "no_number"
    assert es.numeric_error(750.0, 0.75, 0.02, []) == "unit_error (off by 10^3)"
    assert es.numeric_error(-2.0, 2.0, 0.02, []) == "sign_error"
    assert es.numeric_error(3.0, 2.0, 0.02, list(range(30))) == "wrong_value"
    assert es.numeric_error(2.01, 2.0, 0.02, []) == "correct"


def test_prompts():
    shots = {"science_numeric": [{"question": "Q1?", "answer": 75}], "gsm8k": [{"question": "G?", "solution": "2+2=4.", "answer": 4.0}]}
    p = es.numeric_prompt({"question": "How far?", "fewshot": "science_numeric"}, shots)
    assert p == "Question: Q1?\nAnswer: 75\n\nQuestion: How far?\nAnswer:"
    p = es.numeric_prompt({"question": "How many?", "fewshot": "gsm8k"}, shots)
    assert p.startswith("Question: G?\nAnswer: 2+2=4. The answer is 4.\n\n") and p.endswith("Question: How many?\nAnswer:")
    lo, hi = es.wilson(50, 100)
    assert lo < 0.5 < hi and es.wilson(0, 0) == (0.0, 0.0)


# ------------------------------------------------------------------ end to end
def make_set(d: Path):
    (d / "handwritten").mkdir(parents=True)
    shutil.copy(EVALS / "handwritten" / "fewshot_science_numeric.json", d / "handwritten")
    rows = [{"id": "t-1", "domain": "physics", "level": "basic", "type": "mcq", "question": "Water boils at sea level at",
             "choices": ["100 °C", "50 °C", "0 °C"], "answer": 0, "distractor_types": ["correct", "calculation_error", "misconception"],
             "source": "test", "licence": "test", "skill": "concept"},
            {"id": "t-2", "domain": "engineering", "level": "intermediate", "type": "numeric", "question": "What is 6 times 7?",
             "answer": 42, "tolerance": 0.0, "fewshot": "science_numeric", "source": "test", "licence": "test"}]
    with open(d / "handwritten" / "t.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    bes.write_manifest(str(d))


def test_end_to_end_report_and_frozen_check(tmp_path):
    model, cfg = tiny_model()
    ck = tmp_path / "m.pt"
    save_checkpoint(str(ck), model, cfg, step=7, vocab_sha256=BPETokenizer.file_sha256(str(VOCAB)), untrained=True)
    d = tmp_path / "evals"
    make_set(d)
    out = tmp_path / "res" / "e"
    args = ["--ckpt", str(ck), "--vocab", str(VOCAB), "--evals", str(d), "--out", str(out), "--device", "cpu", "--new-tokens", "6"]
    with pytest.warns(UserWarning, match="UNTRAINED"):
        assert es.main(args) == 0
    res = json.loads((tmp_path / "res" / "e.json").read_text())
    assert res["step"] == 7 and res["eval_set"]["items"] == 2 and res["repeat_check"]["identical"]
    assert set(res["summary"]["domain"]) == {"physics", "engineering"}
    assert res["summary"]["domain"]["physics"]["chance"] == pytest.approx(1 / 3)
    md = (tmp_path / "res" / "e.md").read_text()
    assert "## By domain" in md and "## Error types" in md and "t-1" in md
    # leaked ids are left out and listed
    (tmp_path / "leak.json").write_text(json.dumps({"flagged_ids": ["t-1"]}))
    with pytest.warns(UserWarning):
        assert es.main(args + ["--exclude", str(tmp_path / "leak.json")]) == 0
    res = json.loads((tmp_path / "res" / "e.json").read_text())
    assert res["excluded"] == ["t-1"] and res["eval_set"]["items"] == 1
    # a changed question file is refused until the manifest is rewritten on purpose
    with open(d / "handwritten" / "t.jsonl", "a", encoding="utf-8") as f:
        f.write("\n")
    assert es.main(args) == 2


def test_leakage_finds_a_copied_question_across_chunk_edges(tmp_path, tok):
    d = tmp_path / "evals"
    (d / "x").mkdir(parents=True)
    leaked = "A steel rod with a modulus of two hundred gigapascals is two metres long and carries ten kilonewtons of tension."
    clean = "Which unusual violet amphibian sings opera beneath the frozen lakes of a distant imaginary moon every winter?"
    stock = "Which of the following statements is true?"     # one window long: a shared stock phrase is not a leak
    rows = [{"id": "q-leak", "domain": "engineering", "question": leaked, "source": "test (x)"},
            {"id": "q-clean", "domain": "engineering", "question": clean, "source": "test (x)"},
            {"id": "q-short", "domain": "math", "question": "2 + 2?", "source": "test (x)"},
            {"id": "q-stock", "domain": "math", "question": stock, "source": "test (x)"}]
    with open(d / "x" / "q.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    filler = tok.encode("The quick brown fox jumps over the lazy dog. " * 40, allowed_special=set())
    ids = filler + tok.encode(" " + leaked, allowed_special=set())[1:] + filler + tok.encode(leaked, allowed_special=set()) + filler + tok.encode(stock, allowed_special=set()) + filler
    write_bin(str(tmp_path / "a.train.bin"), np.array(ids, dtype=np.uint16), 32000, tok.eos_id)
    out = tmp_path / "leak"
    assert cl.main(["--evals", str(d), "--vocab", str(VOCAB), "--bins", str(tmp_path / "*.train.bin"),
                    "--out", str(out), "--chunk", "37"]) == 0
    res = json.loads(Path(str(out) + ".json").read_text())
    assert res["items"]["q-leak"]["status"] == "leaked" and res["items"]["q-leak"]["fraction"] == 1.0
    assert res["items"]["q-clean"]["status"] == "clean"
    assert res["items"]["q-short"]["status"] == "too_short"
    assert res["items"]["q-stock"]["status"] == "partial" and res["items"]["q-stock"]["matched"] == 1
    assert res["flagged_ids"] == ["q-leak"]
    assert "steel rod" in Path(str(out) + ".md").read_text()
