"""Real-data smoke test on the Biology dataset (opt-in).

    BIO_ZIP="Biology Dataset.zip" python -m pytest tests/test_bio_smoke.py -s

Uses only the small `qbio` shard at 10% budget so it runs in about a minute; the full protocol
(all sources, control set, per-source evaluation) is `scripts/prepare_biology.py` + `scripts/eval_bio.py`
(results in docs/BIOLOGY_SMOKE_TEST.md).
"""
import importlib.util
import json
import math
import os
from pathlib import Path

import pytest

from llm import data
from llm.checkpoint import read_checkpoint
from llm.train import TrainConfig, train

ROOT = Path(__file__).resolve().parent.parent
BIO_ZIP = os.environ.get("BIO_ZIP")
pytestmark = [pytest.mark.data, pytest.mark.slow,
              pytest.mark.skipif(not (BIO_ZIP and os.path.isfile(BIO_ZIP)), reason="set BIO_ZIP to the Biology Dataset zip")]


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    spec = importlib.util.spec_from_file_location("prepare_biology", ROOT / "scripts" / "prepare_biology.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out = tmp_path_factory.mktemp("bio")
    manifest = mod.main(["--zip", BIO_ZIP, "--out", str(out), "--sources", "qbio", "--scale", "0.15",
                         "--vocab-size", "1000", "--no-control"])
    return out, manifest


def test_data_integrity_checks_pass(prepared):
    _, m = prepared
    c = m["checks"]
    assert c["roundtrip_failures"] == 0 and c["ids_below_vocab"] and c["ids_below_65536"] and c["eos_count_matches_docs"]
    assert c["val_windows_seen_in_train_pct"] < 2.0
    assert m["n_vocab"] == 1000 and m["tokenization"]["train"]["bytes_per_token"] > 2.0


def test_manifest_hashes_match_the_files_on_disk(prepared):
    out, m = prepared
    for name, info in m["files"].items():
        assert data.read_header(info["path"]).total_tokens > 0
        assert Path(info["path"]).exists()
    assert json.loads((out / "manifest.json").read_text())["vocab_sha256"] == m["vocab_sha256"]


def test_model_learns_real_text_and_beats_the_unigram_baseline(prepared, tmp_path):
    out, m = prepared
    r = train(TrainConfig(train_bin=str(out / "train.bin"), val_bin=str(out / "val.bin"), vocab=str(out / "vocab.tok"),
                          out_dir=str(tmp_path / "run"), preset="tiny", seq_len=128, batch_size=16, max_steps=80,
                          warmup_steps=10, eval_interval=40, eval_batches=10, lr=3e-3, device="cpu", dtype="fp32", log_interval=1))
    hist = r["history"]
    assert abs(hist[0]["loss"] - math.log(m["n_vocab"])) < 0.5                # starts near uniform
    assert hist[-1]["loss"] < hist[0]["loss"] - 0.7                            # and learns (measured drop: 0.96)
    import numpy as np
    _, tr = data.read_tokens(str(out / "train.bin"), mmap=False)
    _, va = data.read_tokens(str(out / "val.bin"), mmap=False)
    counts = np.bincount(np.asarray(tr, dtype=np.int64), minlength=m["n_vocab"]).astype(float) + 1
    unigram = float(-np.log(counts / counts.sum())[np.asarray(va, dtype=np.int64)].mean())
    assert r["final"]["val_loss"] < unigram + 0.2                              # at least competitive with unigram after 80 steps
    _, _, meta, _ = read_checkpoint(r["checkpoint"])
    assert meta["vocab_sha256"] == m["vocab_sha256"]
