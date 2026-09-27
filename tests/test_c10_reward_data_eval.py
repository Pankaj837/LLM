"""C10 Preference data + evaluation protocol (synthetic pairs, split, ceiling, training entry-point)."""
import subprocess
import sys
from pathlib import Path
from collections import defaultdict

import pytest

from llm.reward.preferences import (CONDITIONS, TOPICS, Response, build_dataset, build_responses,
                                            condition_blind_ceiling, prefers, split)
from helpers import surface_heuristic_accuracy

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def data():
    pairs = build_dataset(seed=0)
    tr, va = split(pairs, 0.2, 0)
    return pairs, tr, va


def test_dataset_size_and_balance(data):
    pairs, tr, va = data
    assert len(TOPICS) == 20 and len(pairs) == 240 and (len(tr), len(va)) == (192, 48)
    assert {c: sum(p.condition == c for p in pairs) for c in CONDITIONS} == {"helpful": 80, "concise": 80, "formal": 80}


def test_dataset_is_deterministic_per_seed():
    a, b = build_dataset(0), build_dataset(0)
    assert [(p.prompt, p.chosen, p.condition) for p in a] == [(p.prompt, p.chosen, p.condition) for p in b]
    assert [p.chosen for p in a] != [p.chosen for p in build_dataset(1)]


def test_split_has_no_prompt_or_string_leakage(data):
    _, tr, va = data
    strings = lambda ps: {s for p in ps for s in (p.prompt, p.chosen, p.rejected)}
    assert not (strings(tr) & strings(va))


def test_split_is_by_topic_and_covers_all_pairs(data):
    pairs, tr, va = data
    assert not ({p.prompt for p in tr} & {p.prompt for p in va}) and len(tr) + len(va) == len(pairs)


def test_prefers_is_antisymmetric():
    r = build_responses(TOPICS[0][1])
    for a in r:
        for b in r:
            for c in CONDITIONS:
                x, y = prefers(c, a, b), prefers(c, b, a)
                assert (x is None and y is None) or (x == 0 and y == 1) or (x == 1 and y == 0)


def test_helpful_and_concise_always_disagree_when_detail_differs():
    a, b = Response("x", 3, 0), Response("y", 1, 0)
    assert prefers("helpful", a, b) != prefers("concise", a, b)


def test_conflicting_flag_matches_definition(data):
    pairs, *_ = data
    by_pair = defaultdict(set)
    for p in pairs:
        by_pair[(p.prompt, frozenset([p.chosen, p.rejected]))].add(p.chosen)
    for p in pairs:
        assert p.conflicting == (len(by_pair[(p.prompt, frozenset([p.chosen, p.rejected]))]) > 1)


def test_condition_blind_ceiling_is_two_thirds_and_matches_brute_force(data):
    """Independent brute force: a condition-blind model has one ranking per unordered pair."""
    _, _, va = data
    groups = defaultdict(lambda: defaultdict(int))
    for p in va:
        groups[(p.prompt, frozenset([p.chosen, p.rejected]))][p.chosen] += 1
    brute = sum(max(v.values()) for v in groups.values()) / len(va)
    assert condition_blind_ceiling(va) == pytest.approx(brute) == pytest.approx(2 / 3, abs=1e-9)


def test_synthetic_data_is_solvable_by_surface_features(data):
    """INFORMATIONAL (passes): a 5-line length/contraction rule scores ~100% on validation.
    So model accuracy on this set cannot be read as evidence of learned preference quality.
    Any reported accuracy must be shown next to this baseline (finding F-19)."""
    _, _, va = data
    assert surface_heuristic_accuracy(va) >= 0.99


def test_validation_set_is_too_small_to_rank_conditioning_methods(data):
    """INFORMATIONAL (passes): 4 held-out topics / 48 pairs -> one pair = 2.1 accuracy points, so the
    README's FiLM 0.938 vs concat 0.958 (one pair) is inside the noise."""
    _, _, va = data
    assert len({p.prompt for p in va}) == 4 and 100 / len(va) > 2.0


# ------------------------------------------------------------------ the training entry-point
@pytest.fixture(scope="module")
def train_run(vocab_path, tmp_path_factory):
    out = tmp_path_factory.mktemp("run") / "crm.pt"
    return subprocess.run(
        [sys.executable, "-m", "llm.reward.train_reward", "--epochs", "1", "--vocab", vocab_path, "--out", str(out)],
        cwd=str(ROOT), capture_output=True, text=True, timeout=300)


@pytest.mark.slow
@pytest.mark.finding("F-18")
def test_training_entrypoint_runs_as_a_module(train_run):
    assert train_run.returncode == 0, train_run.stderr[-400:]
    assert "condition-blind ceiling" in train_run.stdout


@pytest.mark.slow
@pytest.mark.finding("F-19")
def test_report_shows_surface_baseline_and_uses_held_out_swap_prompt(train_run):
    assert "surface-heuristic baseline" in train_run.stdout
    assert "held-out prompt" in train_run.stdout
