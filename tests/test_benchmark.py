"""scripts/benchmark.py: runs real optimizer steps on synthetic tokens, reports cost, survives out-of-memory."""
import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import benchmark as bm  # noqa: E402


def test_cpu_benchmark_reports_throughput_and_cost(tmp_path, capsys):
    out = tmp_path / "b.json"
    assert bm.main(["--presets", "tiny", "--batch-sizes", "1", "2", "--seq-len", "64", "--steps", "4", "--warmup", "1",
                    "--device", "cpu", "--dtype", "fp32", "--vocab", "200", "--out", str(out)]) == 0
    rows = json.loads(out.read_text())["rows"]
    assert [r["status"] for r in rows] == ["ok", "ok"] and all(r["tokens_per_s"] > 0 for r in rows)
    r = rows[1]
    assert r["hours_per_1B_tokens"] == pytest.approx(1e9 / r["tokens_per_s"] / 3600, abs=0.01)
    assert r["rupees_per_1B_tokens"] == round(r["hours_per_1B_tokens"] * 330) or abs(r["rupees_per_1B_tokens"] - r["hours_per_1B_tokens"] * 330) < 2
    assert "tokens/s" in capsys.readouterr().out


def test_out_of_memory_is_reported_and_stops_larger_batches(tmp_path, monkeypatch):
    calls = []

    def fake_one(preset, batch, *a, **k):
        calls.append(batch)
        if batch >= 4:
            return {"preset": preset, "batch": batch, "status": "OOM"}
        return {"preset": preset, "batch": batch, "status": "ok", "params_M": 1.0, "tokens_per_s": 1000, "step_s": 0.1,
                "peak_mem_GB": None, "flops_per_token": 1e9}

    monkeypatch.setattr(bm, "bench_one", fake_one)
    out = tmp_path / "o.json"
    assert bm.main(["--presets", "tiny", "--batch-sizes", "1", "4", "8", "16", "--device", "cpu", "--out", str(out)]) == 0
    assert calls == [1, 4]                                       # 8 and 16 never tried after the OOM at 4
    assert [r["status"] for r in json.loads(out.read_text())["rows"]] == ["ok", "OOM"]
    assert bm.is_oom(RuntimeError("CUDA out of memory. Tried to allocate")) and not bm.is_oom(RuntimeError("shape mismatch"))
    assert bm.flops_per_token(10, type("C", (), {"num_layers": 2, "hidden_dim": 4})(), 8) == 60 + 12 * 2 * 8 * 4


def test_sequence_longer_than_the_preset_context_is_refused_not_crashed():
    r = bm.bench_one("tiny", 1, 100000, 3, 1, torch.device("cpu"), None, 200)
    assert r["status"].startswith("seq_len")
