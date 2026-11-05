"""scripts/compare_eval.py: reproduction check and checkpoint comparison on eval_science result files."""
import copy
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import compare_eval as ce  # noqa: E402


def _result(n=200, vocab="v", correct=lambda i: i % 3 == 0):
    items = []
    for i in range(n):
        items.append({"id": f"q{i}", "domain": "math" if i % 2 else "physics", "type": "mcq", "correct_norm": correct(i),
                      "pick_norm": i % 4, "logprobs": [-1.0, -2.0, -3.0, -4.0]})
    return {"vocab_sha256": vocab, "items": items}


def test_identical_runs_reproduce():
    c = ce.compare(_result(), _result())
    assert c["items_changed"] == 0 and c["ref_overall"] == c["new_overall"]
    assert ce.verdict(c, 0.5, 0.02)[0]


def test_a_few_flipped_near_ties_still_count_as_reproduced():
    ref, new = _result(), _result()
    for i in (3, 10, 21):                              # 3 of 200 = 1.5%, overall moves 0.5 points at most
        new["items"][i]["correct_norm"] = not new["items"][i]["correct_norm"]
    c = ce.compare(ref, new)
    assert c["items_changed"] == 3
    assert ce.verdict(c, tol_overall=1.0, tol_items=0.02)[0]


def test_a_different_checkpoint_is_not_a_reproduction():
    c = ce.compare(_result(), _result(correct=lambda i: i % 3 != 0))
    ok, problems = ce.verdict(c, 0.5, 0.02)
    assert not ok and len(problems) == 2


def test_different_tokenizer_or_question_set_cannot_be_compared():
    with pytest.raises(ValueError, match="tokenizer"):
        ce.compare(_result(vocab="a"), _result(vocab="b"))
    short = _result()
    short["items"] = short["items"][:-1]
    with pytest.raises(ValueError, match="question sets"):
        ce.compare(_result(), short)


def test_per_domain_table_and_cli_exit_codes(tmp_path, capsys):
    ref, new = _result(), copy.deepcopy(_result())
    new["items"][0]["correct_norm"] = not new["items"][0]["correct_norm"]
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.write_text(json.dumps(ref))
    b.write_text(json.dumps(new))
    assert ce.main(["--ref", str(a), "--new", str(b), "--reproduce"]) == 0
    out = capsys.readouterr().out
    assert "REPRODUCED" in out and "| physics |" in out and "| math |" in out
    b.write_text(json.dumps(_result(correct=lambda i: False)))
    assert ce.main(["--ref", str(a), "--new", str(b), "--reproduce"]) == 1
    assert ce.main(["--ref", str(a), "--new", str(b)]) == 0                     # plain comparison never fails
    b.write_text(json.dumps(_result(vocab="x")))
    assert ce.main(["--ref", str(a), "--new", str(b)]) == 2
