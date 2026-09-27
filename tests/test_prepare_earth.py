"""Earth & Environment data adapter: the delivered dataset uses a different binary header (magic b"GNRP",
24 bytes) than this repo's own format (magic b"TOK1", 36 bytes) -- see scripts/prepare_earth.py."""
import importlib.util
import os
import struct
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("prepare_earth", ROOT / "scripts" / "prepare_earth.py")
prepare_earth = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare_earth)

EARTH_ZIP = os.environ.get("EARTH_ZIP")


def make_source_bin(ids, vocab_size=100, eos_id=0, version=1) -> bytes:
    header = struct.pack(prepare_earth.SRC_HEADER_FMT, int.from_bytes(prepare_earth.SRC_MAGIC, "little"),
                         version, vocab_size, eos_id, len(ids))
    return header + np.asarray(ids, dtype="<u2").tobytes()


def test_header_is_24_bytes_with_gnrp_magic():
    assert prepare_earth.SRC_HEADER_SIZE == 24
    assert prepare_earth.SRC_MAGIC == b"GNRP"


def test_reads_a_well_formed_file():
    raw = make_source_bin([1, 2, 3, 0, 4, 5], vocab_size=100, eos_id=0)
    hdr, ids = prepare_earth.read_source_bin(raw)
    assert hdr == {"version": 1, "vocab_size": 100, "eos_id": 0, "total_tokens": 6}
    assert ids.tolist() == [1, 2, 3, 0, 4, 5]


def test_this_repos_own_header_is_not_readable_as_the_earth_format():
    """Documents the incompatibility directly: our TOK1 header does not parse as GNRP and vice versa."""
    from llm import data
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        path = f.name
    try:
        data.write_bin(path, [1, 2, 3], vocab_size=10, eos_id=0)
        raw = open(path, "rb").read()
        with pytest.raises(ValueError, match="magic"):
            prepare_earth.read_source_bin(raw)
    finally:
        os.unlink(path)


def test_rejects_wrong_magic():
    raw = make_source_bin([1, 2])
    raw = b"XXXX" + raw[4:]
    with pytest.raises(ValueError, match="magic"):
        prepare_earth.read_source_bin(raw)


def test_rejects_size_mismatch():
    raw = make_source_bin([1, 2, 3])
    with pytest.raises(ValueError, match="size"):
        prepare_earth.read_source_bin(raw[:-2])          # truncated by one token


def test_empty_file_is_valid():
    hdr, ids = prepare_earth.read_source_bin(make_source_bin([]))
    assert hdr["total_tokens"] == 0 and len(ids) == 0


def test_window_hashes_are_order_sensitive_and_reproducible():
    a = prepare_earth.window_hashes(np.arange(40), 32, 1)
    b = prepare_earth.window_hashes(np.arange(40), 32, 1)
    c = prepare_earth.window_hashes(np.arange(40)[::-1], 32, 1)
    assert np.array_equal(a, b) and not np.array_equal(a, c) and len(a) == 40 - 32 + 1


def test_window_hashes_short_array_returns_empty():
    assert len(prepare_earth.window_hashes(np.arange(5), 32, 1)) == 0


# ------------------------------------------------------------------ opt-in, real dataset
pytestmark_real = pytest.mark.skipif(not (EARTH_ZIP and os.path.isfile(EARTH_ZIP)),
                                     reason="set EARTH_ZIP to the Earth & Environment dataset zip")


@pytestmark_real
def test_prepare_matches_the_delivered_summary_numbers(tmp_path):
    # main() parses sys.argv; call it the same way the script would from the command line
    import sys
    old_argv = sys.argv
    try:
        sys.argv = ["prepare_earth.py", "--zip", EARTH_ZIP, "--out", str(tmp_path)]
        manifest = prepare_earth.main()
    finally:
        sys.argv = old_argv
    assert manifest["files"]["train"]["tokens"] == 3_544_700
    assert manifest["files"]["train"]["documents"] == 99_828
    assert manifest["files"]["val"]["tokens"] == 394_189
    assert manifest["vocab_size"] == 8192 and manifest["eos_id"] == 0
    for split in ("train", "val"):
        c = manifest["checks"][split]
        assert c["file_size_matches_header"] and c["ids_below_vocab"] and c["ids_below_65536"]
    assert manifest["decode_check"]["ok"]
    assert manifest["val_windows_seen_in_train_pct"] < 5.0
