"""scripts/train_vocab.py: JSONL samples -> a valid .tok that round-trips arbitrary text."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import train_vocab as tv  # noqa: E402
from llm.tokenizer import BPETokenizer  # noqa: E402


def test_trains_a_valid_vocab_from_jsonl(tmp_path):
    docs = ["def add(a, b): return a + b", "The mitochondria produce ATP.", "Let $x^2 \ge 0$ for real x."] * 40
    (tmp_path / "a.jsonl").write_text("\n".join(json.dumps({"t": d}) for d in docs), encoding="utf-8")
    out = tmp_path / "v.tok"
    assert tv.main([str(tmp_path / "*.jsonl"), "--vocab-size", "300", "--out", str(out)]) == 0
    tok = BPETokenizer.from_file(str(out))
    assert tok.n_vocab == 300 and tok.eos_id == 299
    for s in docs[:3] + ["héllo 日本語 \U0001f600", ""]:
        assert tok.decode(tok.encode(s, allowed_special=set())) == s
