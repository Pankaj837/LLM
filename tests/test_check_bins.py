"""scripts/check_bins.py: catches truncated files, out-of-range ids, mid-document cuts and vocabulary mismatches."""
import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import check_bins as cb  # noqa: E402
from llm import data  # noqa: E402
from llm.bpe_trainer import train_bpe, write_tok  # noqa: E402
from llm.tokenizer import BPETokenizer  # noqa: E402


def _bin(path, ids, vocab, eos):
    data.write_bin(str(path), np.asarray(ids), vocab_size=vocab, eos_id=eos)
    return str(path)


def test_good_files_pass_and_each_defect_is_reported(tmp_path, capsys):
    vocab = tmp_path / "v.tok"
    write_tok(train_bpe(["hello world " * 50], 300), str(vocab))
    t = BPETokenizer.from_file(str(vocab))
    V, E = t.n_vocab, t.eos_id
    good = _bin(tmp_path / "good.bin", [5, 6, E, 7, E], V, E)
    assert cb.main([good, "--vocab", str(vocab)]) == 0
    assert "RESULT: PASS" in capsys.readouterr().out

    no_eos = _bin(tmp_path / "noeos.bin", [5, 6, E, 7], V, E)
    big_id = _bin(tmp_path / "bigid.bin", [5, V + 100, E], V + 200, E)       # the writer refuses ids >= vocab ...
    raw = bytearray(Path(big_id).read_bytes())
    raw[8:12] = struct.pack("<I", V)                                         # ... so shrink the declared vocabulary
    Path(big_id).write_bytes(bytes(raw))
    other_vocab = _bin(tmp_path / "othervocab.bin", [5, 6, 127], 128, 127)
    for p, text in ((no_eos, "does not end with EOS"), (big_id, ">= vocab_size"), (other_vocab, "frozen tokenizer")):
        assert cb.main([p, "--vocab", str(vocab)]) == 1
        assert text in capsys.readouterr().out

    trunc = Path(good).read_bytes()[:-2]
    (tmp_path / "trunc.bin").write_bytes(trunc)
    assert cb.main([str(tmp_path / "trunc.bin")]) == 1
    assert "truncated or corrupt" in capsys.readouterr().out

    bad_magic = bytearray(Path(good).read_bytes())
    bad_magic[:4] = struct.pack("<I", 0xDEADBEEF)
    (tmp_path / "magic.bin").write_bytes(bytes(bad_magic))
    assert cb.main([str(tmp_path / "magic.bin")]) == 1
    assert cb.main([good, other_vocab]) == 1                       # mixed vocabularies in one set


def test_a_file_tokenized_with_the_other_32k_candidate_is_caught(tmp_path, capsys):
    """Gokul's bio / earth bins (25 Oct): same size, same EOS, same file name 'vocab32k.tok', but the digits candidate."""
    frozen = ROOT / "tokenizers" / "vocab32k_v2" / "vocab32k.tok"
    digits = ROOT / "tokenizers" / "vocab32k_digits_v2" / "digits_NOT_FROZEN.tok"
    tf, td = BPETokenizer.from_file(str(frozen)), BPETokenizer.from_file(str(digits))
    assert (tf.n_vocab, tf.eos_id) == (td.n_vocab, td.eos_id)                # the header cannot tell them apart
    text = [" ".join("Protein %d binds receptor %d in tissue %d (n = %d); the κ constant is 3.14 and β-sheets form." % (i, j % 7, i % 5, i * j)
                     for j in range(60)) for i in range(60)]
    ids = []
    for s in text:
        ids += td.encode(s, allowed_special=set()) + [td.eos_id]
    bad = _bin(tmp_path / "wrong.bin", ids, td.n_vocab, td.eos_id)
    assert cb.main([bad, "--vocab", str(digits)]) == 0                         # right vocabulary: fine
    capsys.readouterr()
    assert cb.main([bad, "--vocab", str(frozen)]) == 1                         # wrong one: caught
    assert "DIFFERENT vocabulary" in capsys.readouterr().out
    ids = []
    for s in text:
        ids += tf.encode(s, allowed_special=set()) + [tf.eos_id]
    good = _bin(tmp_path / "right.bin", ids, tf.n_vocab, tf.eos_id)
    assert cb.main([good, "--vocab", str(frozen)]) == 0


def test_merge_bins_script_concatenates_and_refuses_mixed_vocabularies(tmp_path):
    import merge_bins as mb
    from llm.data import read_header, read_tokens
    a = _bin(tmp_path / "code.rg00000.train.bin", [5, 6, 299], 300, 299)
    b = _bin(tmp_path / "code.rg00001.train.bin", [7, 299], 300, 299)
    out = tmp_path / "code.train.bin"
    assert mb.main([str(tmp_path / "code.rg*.train.bin"), "--out", str(out)]) == 0
    assert read_tokens(str(out), mmap=False)[1].tolist() == [5, 6, 299, 7, 299] and read_header(str(out)).total_tokens == 5
    assert mb.main([str(tmp_path / "code*.train.bin"), "--out", str(out)]) == 0          # re-running never merges the output into itself
    assert read_header(str(out)).total_tokens == 5
    bad = _bin(tmp_path / "other.train.bin", [3, 127], 128, 127)
    import pytest
    with pytest.raises(SystemExit):
        mb.main([a, bad, "--out", str(tmp_path / "bad.bin")])
