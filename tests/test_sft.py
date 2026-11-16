"""SFT: chat format and loss mask, LoRA, training, checkpoints, data preparation, generation scoring (see src/llm/sft.py)."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from helpers import small_lm_config
from llm import data, sft
from llm.bpe_trainer import train_bpe, write_tok
from llm.checkpoint import load_checkpoint, read_checkpoint, save_checkpoint
from llm.model import TransformerModel
from llm.tokenizer import BPETokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import sft_check_run  # noqa: E402
import sft_generate  # noqa: E402
import sft_make_verified  # noqa: E402
import sft_prepare  # noqa: E402
import sft_verify  # noqa: E402

SPECIALS = ("<|endoftext|>", "<|pad|>", "<|bos|>", "<|system|>", "<|user|>", "<|assistant|>", "<|end_of_turn|>")
TEXT = ["what is the derivative of x squared? the answer is 2x. water is H2O. the speed of light is about 3e8 m/s. "
        "def add(a, b): return a + b. newton's second law states that force equals mass times acceleration. " * 4] * 8


@pytest.fixture(scope="module")
def tok_path(tmp_path_factory):
    p = str(tmp_path_factory.mktemp("sft_vocab") / "chat.tok")
    write_tok(train_bpe(TEXT, 330, min_count=1, reserved_specials=len(SPECIALS)), p, specials=SPECIALS)
    return p


@pytest.fixture(scope="module")
def tok(tok_path):
    return BPETokenizer.from_file(tok_path)


def tiny_cfg(tok, **kw):
    return small_lm_config(vocab_size=tok.n_vocab, **kw)


def make_base(tmp_path, tok, tok_path, seed=0, name="base.pt"):
    torch.manual_seed(seed)
    cfg = tiny_cfg(tok)
    m = TransformerModel(cfg)
    p = str(tmp_path / name)
    save_checkpoint(p, m, cfg, step=100, train_loss=3.0, val_loss=3.5, vocab_sha256=BPETokenizer.file_sha256(tok_path))
    return p, m, cfg


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return str(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ------------------------------------------------------------------ format and mask
def test_chat_format_and_response_only_mask(tok):
    sp = sft.special_ids(tok)
    e = sft.encode_example(tok, {"prompt": "what is 2x?", "response": "the answer is 2x."}, 128)
    x, y = e["x"], e["y"]
    assert x[0] == sp["<|user|>"] and sp["<|assistant|>"] in x and sp["<|system|>"] not in x
    sup = [t for t in y if t != sft.IGNORE]
    assert sup[-1] == sp["<|endoftext|>"] and len(sup) == e["n_response"]
    assert tok.decode(sup[:-1]) == "the answer is 2x."                          # exactly the response, then the end token
    k = x.index(sp["<|assistant|>"])
    assert all(t == sft.IGNORE for t in y[:k])                                   # nothing before the assistant marker is supervised
    assert y[k] != sft.IGNORE                                                    # the first response token is predicted from <|assistant|>
    with_sys = sft.encode_example(tok, {"system": "be exact", "prompt": "q", "response": "a"}, 128)
    assert with_sys["x"][0] == sp["<|system|>"]


def test_special_token_text_is_plain_text_and_long_examples_are_dropped_not_cut(tok):
    sp = sft.special_ids(tok)
    ids = sft.encode_plain(tok, "hello <|user|> <|endoftext|>")
    assert sp["<|user|>"] not in ids and sp["<|endoftext|>"] not in ids         # cannot forge a boundary
    assert sft.encode_example(tok, {"prompt": "q " * 100, "response": "a " * 100}, 64) is None
    d = sft.SFTData([{"prompt": "q", "response": "a"}, {"prompt": "q " * 100, "response": "a"}], tok, 64)
    assert len(d) == 1 and d.dropped_too_long == 1 and d.supervised_tokens == 2


def test_sum_loss_counts_only_supervised_tokens(tok):
    torch.manual_seed(0)
    model = TransformerModel(tiny_cfg(tok)).eval()
    items = [(e["x"], e["y"]) for e in (sft.encode_example(tok, {"prompt": f"q{i}", "response": "the answer is 2x"}, 64) for i in range(3))]
    x, y = sft.collate(items, 0, torch.device("cpu"))
    manual = 0.0
    with torch.no_grad():
        logits = model(x).float()
        for b in range(x.shape[0]):
            for t in range(x.shape[1]):
                if y[b, t] != sft.IGNORE:
                    manual += torch.nn.functional.cross_entropy(logits[b, t][None], y[b, t][None]).item()
        assert sft.sum_loss(model, x, y).item() == pytest.approx(manual, rel=1e-5)


# ------------------------------------------------------------------ LoRA
def test_lora_starts_unchanged_trains_only_adapters_and_merges_exactly(tok):
    torch.manual_seed(0)
    cfg = tiny_cfg(tok)
    base = TransformerModel(cfg).eval()
    model = TransformerModel(cfg)
    model.load_state_dict(base.state_dict())
    info = sft.apply_lora(model, r=4, alpha=8)
    model.eval()
    ids = torch.randint(0, cfg.vocab_size, (2, 12))
    with torch.no_grad():
        assert torch.allclose(model(ids), base(ids), atol=1e-6)                  # B = 0: identical at the start
    assert info["trainable"] < info["total"] and info["wrapped_layers"] == cfg.num_layers * 7
    assert all(p.requires_grad == n.endswith(("lora_A", "lora_B")) for n, p in model.named_parameters())
    for m in model.modules():
        if isinstance(m, sft.LoRALinear):
            torch.nn.init.normal_(m.lora_B, std=0.05)
    merged = sft.merged_state_dict(model)
    assert set(merged) == set(base.state_dict())                                 # original key names, no adapter keys
    plain = TransformerModel(cfg).eval()
    plain.load_state_dict(merged)
    with torch.no_grad():
        assert torch.allclose(plain(ids), model(ids), atol=1e-4)


# ------------------------------------------------------------------ training
def make_cfg(tmp_path, tok_path, base, train, val, **kw):
    d = dict(base=base, vocab=tok_path, train=train, val=val, out_dir=str(tmp_path / "run"), device="cpu", dtype="fp32", batch_size=4,
             grad_accum=2, max_len=64, eval_interval=10, log_interval=10, lr=3e-3, epochs=1.0, seed=0)
    d.update(kw)
    return sft.SFTConfig(**d)


def fixed_rows(n=16):
    return [{"prompt": "what is the derivative of x squared?", "response": "the answer is 2x."} for _ in range(n)]


def test_full_training_learns_saves_loadable_checkpoints_and_never_touches_the_base(tmp_path, tok, tok_path):
    base, _, cfg = make_base(tmp_path, tok, tok_path)
    before = sha(base)
    tr, va = write_jsonl(tmp_path / "train.jsonl", fixed_rows()), write_jsonl(tmp_path / "val.jsonl", fixed_rows(4))
    rep = sft.sft_train(make_cfg(tmp_path, tok_path, base, tr, va, max_steps=40, eval_interval=10, save_interval=20))
    assert rep["after"]["val"]["loss"] < rep["before"]["val"]["loss"] * 0.5
    assert sha(base) == before                                                   # the base checkpoint is read only
    out = Path(rep["config"]["out_dir"])
    assert {p.name for p in out.glob("*.pt")} >= {"ckpt_best.pt", "ckpt_last.pt", "ckpt_step_000020.pt"}
    fresh = TransformerModel(cfg)
    meta = load_checkpoint(str(out / "ckpt_last.pt"), fresh, expected_vocab_sha256=BPETokenizer.file_sha256(tok_path))
    assert meta["step"] == 40 and meta["extra"]["sft"]["base_sha256"] == before and meta["extra"]["sft"]["mode"] == "full"
    assert "optimizer" not in read_checkpoint(str(out / "ckpt_last.pt"))[3]
    assert (out / "sft_report.json").exists() and (out / "sft_metrics.jsonl").exists()
    # after training the model answers the trained prompt with the trained response and ends it itself
    fresh.eval()
    eos = tok.special_to_id["<|endoftext|>"]
    pids = sft.prompt_ids(tok, "what is the derivative of x squared?")
    got, finished = sft.generate_ids(fresh, pids, max_new_tokens=30, eos_id=eos, device=torch.device("cpu"))
    assert tok.decode(got) == "the answer is 2x." and finished


def test_step_does_not_depend_on_how_it_is_split_into_micro_batches(tmp_path, tok, tok_path):
    base, _, cfg = make_base(tmp_path, tok, tok_path)
    rows = [{"prompt": f"question {i} " * (1 + i % 3), "response": "answer " * (1 + i % 4)} for i in range(8)]
    tr = write_jsonl(tmp_path / "train.jsonl", rows)
    recs = []
    for bs, ga in ((4, 1), (2, 2), (1, 4)):
        rep = sft.sft_train(make_cfg(tmp_path, tok_path, base, tr, None, batch_size=bs, grad_accum=ga, max_steps=1, out_dir=str(tmp_path / f"r{bs}"),
                                     warmup_ratio=0.0))
        recs.append(json.loads((Path(rep["config"]["out_dir"]) / "sft_metrics.jsonl").read_text(encoding="utf-8").splitlines()[0]))
    for other in recs[1:]:                      # same token-mean loss and same gradient, whatever the micro-batch size
        assert other["loss"] == pytest.approx(recs[0]["loss"], rel=1e-5)
        assert other["grad_norm"] == pytest.approx(recs[0]["grad_norm"], rel=1e-4)


def test_lora_training_gives_a_plain_checkpoint_and_leaves_frozen_weights_alone(tmp_path, tok, tok_path):
    base, base_model, cfg = make_base(tmp_path, tok, tok_path)
    tr, va = write_jsonl(tmp_path / "train.jsonl", fixed_rows()), write_jsonl(tmp_path / "val.jsonl", fixed_rows(4))
    rep = sft.sft_train(make_cfg(tmp_path, tok_path, base, tr, va, mode="lora", lora_r=8, lora_alpha=16, lr=1e-2, max_steps=40))
    assert rep["trainable_parameters"] < 0.5 * sum(p.numel() for p in base_model.parameters())
    assert rep["after"]["val"]["loss"] < rep["before"]["val"]["loss"]
    state = read_checkpoint(rep["final_checkpoint"])[0]
    assert set(state) == set(base_model.state_dict())
    marker_rows = [tok.special_to_id[m] for m in ("<|system|>", "<|user|>", "<|assistant|>")]
    keep = [i for i in range(cfg.vocab_size) if i not in marker_rows]
    assert torch.equal(state["tok_embeddings.weight"][keep], base_model.state_dict()["tok_embeddings.weight"][keep])   # frozen except the marker rows
    assert not torch.equal(state["tok_embeddings.weight"][marker_rows], base_model.state_dict()["tok_embeddings.weight"][marker_rows])
    assert not torch.equal(state["layers.0.attn.q_proj.weight"], base_model.state_dict()["layers.0.attn.q_proj.weight"])   # merged adapter
    load_checkpoint(rep["final_checkpoint"], TransformerModel(cfg))


def test_refuses_wrong_hash_wrong_vocabulary_and_empty_data(tmp_path, tok, tok_path):
    base, _, _ = make_base(tmp_path, tok, tok_path)
    tr = write_jsonl(tmp_path / "train.jsonl", fixed_rows())
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        sft.sft_train(make_cfg(tmp_path, tok_path, base, tr, None, base_sha256="0" * 64, max_steps=1))
    other = str(tmp_path / "other.tok")
    write_tok(train_bpe(TEXT, 300, min_count=1, reserved_specials=len(SPECIALS)), other, specials=SPECIALS)
    with pytest.raises(ValueError, match="different vocabulary"):
        sft.sft_train(make_cfg(tmp_path, other, base, tr, None, max_steps=1))
    empty = write_jsonl(tmp_path / "long.jsonl", [{"prompt": "q " * 200, "response": "a"}])
    with pytest.raises(ValueError, match="no usable training examples"):
        sft.sft_train(make_cfg(tmp_path, tok_path, base, empty, None, max_steps=1))


def test_retention_check_reports_pretraining_loss_before_and_after(tmp_path, tok, tok_path):
    base, _, cfg = make_base(tmp_path, tok, tok_path)
    ids = np.tile(np.arange(1, 20), 200)
    binp = str(tmp_path / "pre.val.bin")
    data.write_bin(binp, ids, vocab_size=cfg.vocab_size, eos_id=tok.special_to_id["<|endoftext|>"])
    tr = write_jsonl(tmp_path / "train.jsonl", fixed_rows())
    rep = sft.sft_train(make_cfg(tmp_path, tok_path, base, tr, None, max_steps=3, retain_val=[f"pre={binp}"], retain_batches=2, retain_seq_len=32))
    assert set(rep["before"]["retain"]) == {"pre"} and set(rep["after"]["retain"]) == {"pre"}


# ------------------------------------------------------------------ generation
def test_decoding_helpers(tok):
    assert sft._banned_by_ngram([1, 2, 3, 1, 2], 3) == {3}
    assert sft._banned_by_ngram([1, 2, 3], 3) == set()
    torch.manual_seed(0)
    model = TransformerModel(tiny_cfg(tok)).eval()
    dev, eos = torch.device("cpu"), tok.special_to_id["<|endoftext|>"]
    p = sft.prompt_ids(tok, "hello")
    a, _ = sft.generate_ids(model, p, max_new_tokens=12, eos_id=eos, device=dev, temperature=0.8, top_k=20, top_p=0.9, seed=3)
    b, _ = sft.generate_ids(model, p, max_new_tokens=12, eos_id=eos, device=dev, temperature=0.8, top_k=20, top_p=0.9, seed=3)
    assert a == b                                                                # same seed, same sample
    g, fin = sft.generate_ids(model, p, max_new_tokens=6, eos_id=eos, device=dev, repetition_penalty=1.3, no_repeat_ngram=2)
    assert len(g) <= 6 and not (fin and len(g) == 6)
    pairs = list(zip(g, g[1:]))
    assert len(pairs) == len(set(pairs))                                         # no repeated bigram
    with pytest.raises(ValueError, match="exceeds the context"):
        sft.generate_ids(model, p, max_new_tokens=10_000, eos_id=eos, device=dev)


# ------------------------------------------------------------------ verify helpers
def test_answer_and_code_checks():
    assert sft_verify.final_number("17 * 20 = 340, 17 * 4 = 68. The answer is 408.") == 408
    assert sft_verify.final_number("Answer: 1,234.5") == 1234.5
    assert sft_verify.final_number("no digits") is None
    assert sft_verify.answer_matches("so the answer is 4.08e2", 408) and not sft_verify.answer_matches("the answer is 409", 408)
    assert sft_verify.answer_matches("It is H2O.", "h2o") and not sft_verify.answer_matches("It is CH4.", "H2O")
    assert sft_verify.extract_code_block("text\n```python\nx = 1\n```\nmore") == "x = 1"
    assert sft_verify.run_python_tests("def add(a, b):\n    return a + b", "assert add(1, 2) == 3")[0]
    ok, msg = sft_verify.run_python_tests("def add(a, b):\n    return a - b", "assert add(1, 2) == 3")
    assert not ok and "AssertionError" in msg
    assert not sft_verify.run_python_tests("while True:\n    pass", "", timeout=1.0)[0]


# ------------------------------------------------------------------ prepare
def test_prepare_filters_splits_and_decontaminates(tmp_path, tok):
    rows = [{"id": f"r{i}", "prompt": f"unique question number {i} about physics", "response": f"unique answer {i}", "domain": "physics"} for i in range(60)]
    rows += [
        {"prompt": "Dup  question", "response": "same"}, {"prompt": "dup question", "response": "SAME"},                       # duplicate
        {"prompt": "dup question", "response": "another answer"},                                                            # duplicate_prompt
        {"prompt": "what is 2+2", "response": "it is 4", "answer": 4}, {"prompt": "what is 2+2", "response": "five", "answer": 5},   # contradictory x2
        {"prompt": "bad math", "response": "the answer is 7", "answer": 8},                                                  # answer_mismatch
        {"prompt": "has <|user|> inside", "response": "x"},                                                                  # special_text
        {"prompt": "q " * 400, "response": "a"},                                                                             # too_long
        {"id": "leak", "prompt": "which planet in our solar system has the most confirmed moons orbiting it today", "response": "saturn"},
    ]
    items = [{"id": "E1", "question": "which planet in our solar system has the most confirmed moons orbiting it today"}]
    idx = sft_prepare.EvalIndex(items, tok)
    out = sft_prepare.prepare(rows, tok, max_len=160, val_frac=0.2, test_frac=0.2, seed=1, eval_index=idx)
    r = out["reasons"]
    assert r["duplicate"] == 1 and r["duplicate_prompt"] == 1 and r["contradictory"] == 2 and r["answer_mismatch"] == 1
    assert r["special_text"] == 1 and r["too_long"] == 1 and r["contaminated"] == 1
    assert out["contaminated"] == {"leak": ["E1"]}
    kept = [ex for s in out["splits"].values() for ex in s]
    prompts = [sft_prepare.norm(ex["prompt"]) for ex in kept]
    assert len(prompts) == len(set(prompts))                                     # no prompt twice, so none in two splits
    assert {len(s) > 0 for s in out["splits"].values()} == {True}
    again = sft_prepare.prepare(rows, tok, max_len=160, val_frac=0.2, test_frac=0.2, seed=1, eval_index=idx)
    assert [ex["id"] for ex in again["splits"]["test"]] == [ex["id"] for ex in out["splits"]["test"]]       # deterministic


def test_prepare_verifies_code_only_when_asked(tmp_path, tok):
    good = {"prompt": "write add", "response": "```python\ndef add(a, b):\n    return a + b\n```", "tests": "assert add(1, 2) == 3"}
    bad = {"prompt": "write sub", "response": "```python\ndef sub(a, b):\n    return a + b\n```", "tests": "assert sub(3, 1) == 2"}
    off = sft_prepare.prepare([dict(good), dict(bad)], tok, max_len=96, val_frac=0, test_frac=0)
    assert len(off["splits"]["train"]) == 2
    on = sft_prepare.prepare([dict(good), dict(bad)], tok, max_len=96, val_frac=0, test_frac=0, verify_code=True)
    assert [e["prompt"] for e in on["splits"]["train"]] == ["write add"] and on["reasons"]["code_test_failed"] == 1


def test_prepare_command_line_writes_files_and_manifest(tmp_path, tok_path):
    raw = write_jsonl(tmp_path / "raw.jsonl", [{"instruction": f"question {i}", "output": f"answer {i}", "domain": "d%d" % (i % 2)} for i in range(40)])
    assert sft_prepare.main(["--input", raw, "--vocab", tok_path, "--out-dir", str(tmp_path / "ds"), "--max-len", "64", "--val-frac", "0.1",
                             "--test-frac", "0.1"]) == 0
    man = json.loads((tmp_path / "ds" / "manifest.json").read_text(encoding="utf-8"))
    assert man["kept"] == 40 and sum(s["examples"] for s in man["splits"].values()) == 40
    for name, s in man["splits"].items():
        assert s["sha256"] == sha(tmp_path / "ds" / s["file"]) and s["supervised_tokens"] > 0
    assert man["vocab_sha256"] == BPETokenizer.file_sha256(tok_path) and man["inputs"][0]["sha256"] == sha(raw)
    sft.load_jsonl(str(tmp_path / "ds" / "train.jsonl"))                          # what the trainer reads


def test_generate_script_compares_checkpoints_on_identical_prompts(tmp_path, tok, tok_path):
    a, _, _ = make_base(tmp_path, tok, tok_path, seed=0, name="a.pt")
    b, _, _ = make_base(tmp_path, tok, tok_path, seed=1, name="b.pt")
    prompts = write_jsonl(tmp_path / "test.jsonl", [{"prompt": "what is 2+3?", "response": "5", "answer": 5, "domain": "math"}, {"prompt": "name a gas", "response": "neon", "domain": "chem"}])
    out = str(tmp_path / "cmp")
    assert sft_generate.main(["--ckpt", f"A={a}", f"B={b}", "--vocab", tok_path, "--prompts", prompts, "--out", out, "--max-new-tokens", "8",
                              "--device", "cpu", "--dtype", "fp32"]) == 0
    res = json.loads(Path(out + ".json").read_text(encoding="utf-8"))
    assert [r["name"] for r in res["runs"]] == ["A", "B"]
    assert [x["prompt"] for x in res["runs"][0]["results"]] == [x["prompt"] for x in res["runs"][1]["results"]]
    s = res["runs"][0]["summary"]
    assert s["examples"] == 2 and s["scored"] == 1 and 0 <= s["finished_rate"] <= 1
    md = Path(out + ".md").read_text(encoding="utf-8")
    assert "what is 2+3?" in md and "Every prompt and every response" in md
    assert sft_generate.wilson(0, 0) == [0.0, 1.0] and sft_generate.wilson(5, 10)[0] < 0.5 < sft_generate.wilson(5, 10)[1]


# ------------------------------------------------------------------ synthetic verified examples
def _poly(text):
    import re
    e = re.sub(r"(\d)x", r"\g<1>*x", text.replace("^", "**").replace(" ", ""))
    return lambda x: eval(e, {"x": x})                                            # test-only: strings generated by sft_make_verified


def test_synthetic_examples_are_correct_by_construction_and_pass_the_pipeline(tok):
    import re
    rows = sft_make_verified.make(40, 0, list(sft_make_verified.KINDS))
    assert rows == sft_make_verified.make(40, 0, list(sft_make_verified.KINDS))   # the seed fixes the file
    assert len({r["prompt"] for r in rows}) == len(rows) == 160
    for r in rows:
        if r["source"] == "synthetic:arith":                                       # recompute from the prompt, independently
            a, b = map(int, re.findall(r"\d+", r["prompt"]))
            op = "*" if ("*" in r["prompt"] or "times" in r["prompt"] or "product" in r["prompt"]) else "+" if ("+" in r["prompt"] or "plus" in r["prompt"]) else "-"
            assert r["answer"] == {"*": a * b, "+": a + b, "-": a - b}[op]
        if r["source"] == "synthetic:poly":
            f = _poly(r["prompt"].split("f(x) = ")[1].split(" with respect")[0])
            g = _poly(r["answer"])
            assert g(1.3) == pytest.approx((f(1.3 + 1e-5) - f(1.3 - 1e-5)) / 2e-5, rel=1e-4)
    out = sft_prepare.prepare(rows, tok, max_len=700, val_frac=0.1, test_frac=0.1, verify_code=True)
    assert sum(out["reasons"].values()) == 0                                       # answers re-checked and every code test executed
    assert sum(len(s) for s in out["splits"].values()) == 160


# ------------------------------------------------------------------ marker initialisation and run check
def test_role_markers_start_from_their_word_embeddings_and_nothing_else_changes(tok):
    torch.manual_seed(0)
    model = TransformerModel(tiny_cfg(tok))
    before = model.tok_embeddings.weight.detach().clone()
    used = sft.init_marker_embeddings(model, tok)
    after = model.tok_embeddings.weight.detach()
    rows = [tok.special_to_id[m] for m in ("<|system|>", "<|user|>", "<|assistant|>")]
    for marker, word in sft.MARKER_WORDS.items():
        ids = sft.encode_plain(tok, word)
        assert used[marker] == ids
        assert torch.allclose(after[tok.special_to_id[marker]], before[ids].mean(0))
    others = [i for i in range(before.shape[0]) if i not in rows]
    assert torch.equal(after[others], before[others])                              # no other token is touched
    assert not torch.equal(after[rows], before[rows])


def test_check_run_passes_a_good_run_and_flags_each_kind_of_damage(tmp_path, tok, tok_path):
    base, _, _ = make_base(tmp_path, tok, tok_path)
    sha_base = sha(base)
    tr, va = write_jsonl(tmp_path / "train.jsonl", fixed_rows()), write_jsonl(tmp_path / "val.jsonl", fixed_rows(4))
    rep = sft.sft_train(make_cfg(tmp_path, tok_path, base, tr, va, max_steps=20, eval_interval=10, base_sha256=sha_base))
    run = rep["config"]["out_dir"]
    res = sft_check_run.check_run(run, base, sha_base, hash_base=True, require_improvement=True)
    assert all(ok for ok, _ in res), [m for ok, m in res if not ok]
    assert sft_check_run.main(["--run", run, "--base", base, "--base-sha256", sha_base, "--require-improvement"]) == 0
    wrong = sft_check_run.check_run(run, base, "0" * 64)
    assert not dict((m, ok) for ok, m in wrong)["report's base SHA-256 equals the expected one"]
    with open(Path(run) / "sft_metrics.jsonl", "a", encoding="utf-8") as f:        # a NaN loss appears in the log
        f.write(json.dumps({"step": 21, "loss": float("nan"), "grad_norm": 1.0}) + "\n")
    assert not all(ok for ok, _ in sft_check_run.check_run(run, base, sha_base))
    assert not all(ok for ok, _ in sft_check_run.check_run(str(tmp_path / "missing"), base))       # nothing there
    state, cfg, meta, raw = read_checkpoint(str(Path(run) / "ckpt_last.pt"))
    raw["optimizer"] = {"state": {}, "param_groups": []}                              # an optimizer sneaks into the checkpoint
    torch.save(raw, Path(run) / "ckpt_last.pt")
    assert "ckpt_last.pt has no optimizer state" in [m for ok, m in sft_check_run.check_run(run, base, sha_base) if not ok]
