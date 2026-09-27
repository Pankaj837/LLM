"""`.bin` token files in the tokenizer team's format (36-byte LE header + uint16 ids)."""
import struct

import numpy as np
import pytest
import torch

from llm import data


@pytest.fixture()
def bin_file(tmp_path):
    ids = np.arange(1000) % 500
    p = tmp_path / "train.bin"
    data.write_bin(str(p), ids, vocab_size=500, eos_id=499)
    return str(p), ids


def test_header_is_36_bytes_not_32():
    assert data.HEADER_SIZE == 36 and struct.calcsize("<IIIIQIQ") == 36


def test_file_layout_matches_the_report_exactly(bin_file):
    """The exact snippet from the tokenizer report must read our file."""
    path, ids = bin_file
    with open(path, "rb") as f:
        header = f.read(36)
        magic, version, vocab_size, eos_id, total_tokens, split, reserved = struct.unpack("<IIIIQIQ", header)
        assert magic == 0x544F4B31
        token_ids = np.fromfile(f, dtype="<u2", count=total_tokens)
    assert (vocab_size, eos_id, total_tokens) == (500, 499, 1000)
    assert np.array_equal(token_ids, ids)
    assert len(open(path, "rb").read()) == 36 + 2 * 1000


def test_roundtrip_mmap_and_in_memory(bin_file):
    path, ids = bin_file
    for mm in (True, False):
        h, toks = data.read_tokens(path, mmap=mm)
        assert h.total_tokens == 1000 and np.array_equal(np.asarray(toks), ids)


def test_bad_magic_rejected(tmp_path):
    p = tmp_path / "x.bin"
    p.write_bytes(struct.pack("<IIIIQIQ", 0xDEADBEEF, 1, 10, 9, 0, 0, 0))
    with pytest.raises(ValueError, match="magic"):
        data.read_header(str(p))


def test_truncated_file_rejected(bin_file, tmp_path):
    path, _ = bin_file
    raw = open(path, "rb").read()
    q = tmp_path / "trunc.bin"
    q.write_bytes(raw[:-10])
    with pytest.raises(ValueError, match="truncated|size"):
        data.read_header(str(q))
    q.write_bytes(raw[:20])
    with pytest.raises(ValueError, match="shorter"):
        data.read_header(str(q))


def test_ids_above_uint16_rejected(tmp_path):
    with pytest.raises(ValueError, match="uint16|65535"):
        data.write_bin(str(tmp_path / "a.bin"), [70000], vocab_size=80000, eos_id=1)


def test_id_outside_vocab_and_bad_eos_rejected(tmp_path):
    with pytest.raises(ValueError):
        data.write_bin(str(tmp_path / "a.bin"), [10], vocab_size=10, eos_id=1)
    with pytest.raises(ValueError):
        data.write_bin(str(tmp_path / "b.bin"), [1], vocab_size=10, eos_id=10)


def test_empty_file_is_valid(tmp_path):
    p = str(tmp_path / "e.bin")
    data.write_bin(p, [], vocab_size=10, eos_id=1)
    h, t = data.read_tokens(p)
    assert h.total_tokens == 0 and len(t) == 0


def test_vocab_compatibility_check(bin_file):
    path, _ = bin_file
    h = data.read_header(path)
    data.check_compatible(h, 500)
    with pytest.raises(ValueError, match="vocab"):
        data.check_compatible(h, 8000)


def test_dataset_targets_are_inputs_shifted_by_one(bin_file):
    ds = data.TokenDataset(bin_file[0], seq_len=16, model_vocab_size=500)
    x, y = ds.window(37)
    assert np.array_equal(y[:-1], x[1:]) and x[0] == (37 % 500)


def test_get_batch_is_reproducible_with_a_generator_and_correct_shape(bin_file):
    ds = data.TokenDataset(bin_file[0], seq_len=16)
    g1, g2 = torch.Generator().manual_seed(5), torch.Generator().manual_seed(5)
    x1, y1 = ds.get_batch(4, g1)
    x2, y2 = ds.get_batch(4, g2)
    assert x1.shape == (4, 16) and x1.dtype == torch.int64 and torch.equal(x1, x2) and torch.equal(y1, y2)


def test_get_batch_never_reads_past_the_end(tmp_path):
    p = str(tmp_path / "s.bin")
    data.write_bin(p, np.arange(20) % 10, vocab_size=10, eos_id=9)
    ds = data.TokenDataset(p, seq_len=19)          # exactly one window
    g = torch.Generator().manual_seed(0)
    for _ in range(20):
        x, y = ds.get_batch(2, g)
        assert x.shape == (2, 19)


def test_sequential_batches_are_disjoint_and_deterministic(bin_file):
    ds = data.TokenDataset(bin_file[0], seq_len=16)
    a = [x for x, _ in ds.sequential_batches(4)]
    b = [x for x, _ in ds.sequential_batches(4)]
    assert all(torch.equal(u, v) for u, v in zip(a, b))
    starts = [int(row[0]) for batch in a for row in batch]
    assert len(starts) == len(set(range(0, len(ds), 16)))


def test_too_few_tokens_rejected(tmp_path):
    p = str(tmp_path / "t.bin")
    data.write_bin(p, [1, 2, 3], vocab_size=10, eos_id=9)
    with pytest.raises(ValueError, match="too few"):
        data.TokenDataset(p, seq_len=8)
