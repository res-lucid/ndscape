"""Tests for split_analyzer.

The tokenizer is the fingerprint of the embedding. If it changes, all
cache_nd/*.joblib artifacts are stale and must be rebuilt (~hours).
"""
import hashlib
import json
import pytest
from pathlib import Path
from joblib import load
from core import split_analyzer

FIXTURES = Path(__file__).parent / "fixtures"


def _load_trees(C, fname):
    cache = Path(__file__).parent.parent / "cache_nd" / fname
    if not cache.exists():
        pytest.skip(f"{fname} not found")
    return load(cache)["trees"]


# ---- structural properties (all trees) ----

@pytest.mark.parametrize("C, fname", [
    (6,  "art_1_C6_N945_s0_d2.joblib"),
    (7,  "art_1_C7_N10395_s0_d2.joblib"),
    (8,  "art_1_C8_N15000_s0_d2.joblib"),
    (9,  "art_1_C9_N15000_s0_d2.joblib"),
    (10, "art_1_C10_N15000_s0_d2.joblib"),
    (19, "art_1_C19_N15000_s0_d2.joblib"),
])
def test_returns_nonempty_list(C, fname):
    trees = _load_trees(C, fname)
    analyze = split_analyzer(C)
    for t in trees:
        tokens = analyze(str(t))
        assert isinstance(tokens, list)
        assert len(tokens) > 0


@pytest.mark.parametrize("C, fname", [
    (6,  "art_1_C6_N945_s0_d2.joblib"),
    (7,  "art_1_C7_N10395_s0_d2.joblib"),
    (10, "art_1_C10_N15000_s0_d2.joblib"),
])
def test_tokens_are_strings(C, fname):
    trees = _load_trees(C, fname)
    analyze = split_analyzer(C)
    for t in trees:
        for tok in analyze(str(t)):
            assert isinstance(tok, str)


@pytest.mark.parametrize("C, fname", [
    (6,  "art_1_C6_N945_s0_d2.joblib"),
    (7,  "art_1_C7_N10395_s0_d2.joblib"),
    (10, "art_1_C10_N15000_s0_d2.joblib"),
])
def test_accepts_tuple_input(C, fname):
    """Analyzer should work when passed a tuple directly, not just a string."""
    trees = _load_trees(C, fname)
    analyze = split_analyzer(C)
    for t in trees:
        tokens_str = analyze(str(t))
        tokens_tuple = analyze(t)
        assert tokens_str == tokens_tuple


# ---- golden master snapshots (hash of all token lists) ----

@pytest.mark.parametrize("C, fname", [
    (6,  "art_1_C6_N945_s0_d2.joblib"),
    (7,  "art_1_C7_N10395_s0_d2.joblib"),
    (8,  "art_1_C8_N15000_s0_d2.joblib"),
    (9,  "art_1_C9_N15000_s0_d2.joblib"),
    (10, "art_1_C10_N15000_s0_d2.joblib"),
    (19, "art_1_C19_N15000_s0_d2.joblib"),
])
def test_snapshot_all_trees(C, fname):
    """
    Hash token lists for all trees at each class count.
    If this fails after refactoring, all cache_nd artifacts need regeneration.
    """
    trees = _load_trees(C, fname)
    analyze = split_analyzer(C)
    digest = hashlib.md5(json.dumps([analyze(str(t)) for t in trees]).encode()).hexdigest()
    fp = FIXTURES / f"split_analyzer_hash_C{C}.json"
    if not fp.exists():
        fp.write_text(json.dumps(digest))
        pytest.skip(f"fixture created: {fp.name} — re-run to validate")
    assert digest == json.loads(fp.read_text())

