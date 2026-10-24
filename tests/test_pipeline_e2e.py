"""The whole data pipeline end to end on synthetic data:
dedup -> tokenize (Parquet in row-group chunks, and text) -> per-domain merge -> integrity check -> weighted training
-> per-domain evaluation. Guards the hand-offs between the scripts, which no single-script test covers."""
import sys
from pathlib import Path

import numpy as np
import pytest

pa = pytest.importorskip("pyarrow")
import pyarrow.parquet as pq  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import check_bins as cb  # noqa: E402
import dedup_parquet as dp  # noqa: E402
import dedup_text as dt  # noqa: E402
import eval_domains as ed  # noqa: E402
import merge_bins as mb  # noqa: E402
import tokenize_parquet as tp  # noqa: E402
import tokenize_text as tt  # noqa: E402
from llm.bpe_trainer import train_bpe, write_tok  # noqa: E402
from llm.data import read_header  # noqa: E402
from llm.train import TrainConfig, train  # noqa: E402


def test_pipeline_from_raw_files_to_a_trained_and_evaluated_model(tmp_path):
    bio = [f"Protein {i} binds receptor {i % 7} and regulates the cell cycle in tissue {i % 5}." for i in range(240)]
    code = [f"def f{i}(x):\n    return x * {i} + {i % 9}  # helper number {i}" for i in range(200)]
    math_lines = [f"Let n = {i}. Then n squared equals {i * i} and n plus one equals {i + 1}." for i in range(300)]
    vocab = tmp_path / "v.tok"
    write_tok(train_bpe((bio + code + math_lines) * 3, 400), str(vocab))

    # raw inputs: biology with duplicates (Parquet, several row groups), code (Parquet v4.0 layout), math (text)
    pq.write_table(pa.table({"doc_id": [f"b{i}" for i in range(len(bio) + 20)], "domain": ["bio"] * (len(bio) + 20),
                             "text": bio + bio[:20]}), tmp_path / "bio.parquet", row_group_size=60)
    import json
    pq.write_table(pa.table({"entity_id": [f"c{i}" for i in range(len(code))], "domain": ["code"] * len(code),
                             "description": ["d"] * len(code),
                             "properties": [json.dumps({"code_content": c}) for c in code]}), tmp_path / "code.parquet", row_group_size=50)
    (tmp_path / "math.txt").write_text("\n".join(math_lines + math_lines[:30]) + "\n", encoding="utf-8")

    # 1. exact dedup
    r = dp.clean_shards([str(tmp_path / "bio.parquet")], str(tmp_path / "clean"))
    assert r["total"]["duplicate"] == 20
    assert dt.main([str(tmp_path / "math.txt"), "--out", str(tmp_path / "clean"), "--min-chars", "10"]) == 0

    # 2. tokenize: Parquet split into one job per row group; text in one go
    bins = tmp_path / "bins"
    assert tp.main([str(tmp_path / "clean" / "bio.parquet"), str(tmp_path / "code.parquet"), "--vocab", str(vocab),
                    "--out", str(bins), "--workers", "2", "--val-frac", "0.1", "--no-merge", "--chunk-mb", "0.0001"]) == 0
    assert len(list(bins.glob("code.rg*.train.bin"))) > 1
    assert tt.main([f"math={tmp_path / 'clean' / 'math.txt'}", "--vocab", str(vocab), "--out", str(bins), "--workers", "2",
                    "--val-frac", "0.1", "--no-merge"]) == 0

    # 3. one train and one validation file per domain
    dom = tmp_path / "dom"
    for d in ("bio", "code", "math"):
        for split in ("train", "val"):
            assert mb.main([str(bins / f"{d}*.{split}.bin"), "--out", str(dom / f"{d}.{split}.bin"), "--split", split]) == 0

    # 4. integrity gate
    assert cb.main([str(dom / "*.bin"), "--vocab", str(vocab)]) == 0

    # 5. weighted training, then per-domain evaluation (wildcard pools a domain's validation files)
    cfg = TrainConfig(train_mix=[f"{dom / f'{d}.train.bin'}={w}" for d, w in (("bio", 0.4), ("code", 0.3), ("math", 0.3))],
                      out_dir=str(tmp_path / "run"), preset="tiny", seq_len=32, batch_size=8, max_steps=30, warmup_steps=3,
                      eval_interval=1000, save_interval=1000, log_interval=10, lr=3e-3, device="cpu", dtype="fp32")
    res = train(cfg)
    assert res["history"][-1]["loss"] < res["history"][0]["loss"]
    ev = ed.evaluate_domains(res["checkpoint"], {d: str(dom / f"{d}.val.bin") for d in ("bio", "code", "math")},
                             32, 4, 10, device="cpu", dtype="fp32", vocab=str(vocab))
    assert ev["ok"] and all(v["val_loss"] is not None for v in ev["domains"].values())
    # token accounting: what was tokenized is exactly what the merged files hold
    total = sum(read_header(str(dom / f"{d}.{s}.bin")).total_tokens for d in ("bio", "code", "math") for s in ("train", "val"))
    texts = bio + code + math_lines
    tok = tp.BPETokenizer.from_file(str(vocab))
    assert total == sum(len(tok.encode(t, allowed_special=set())) + 1 for t in texts)
    assert np.isfinite(ev["mean_val_loss"])
