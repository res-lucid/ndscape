"""Tests for core.all_trees. Count must match (2n-3)!! and canonical form must be stable."""
import json
import pytest
from pathlib import Path
from joblib import load
import core as mh

FIXTURES = Path(__file__).parent / "fixtures"

TOTAL_TREES = {3: 3, 4: 15, 5: 105, 6: 945, 7: 10395}


@pytest.mark.parametrize("n, expected", TOTAL_TREES.items())
def test_count(n, expected):
    assert len(mh.all_trees(n)) == expected


@pytest.mark.parametrize("n, expected", TOTAL_TREES.items())
def test_uniqueness(n, expected):
    trees = mh.all_trees(n)
    assert len(set(map(str, trees))) == expected


@pytest.mark.parametrize("n", TOTAL_TREES.keys())
def test_all_classes_present(n):
    expected_classes = frozenset(range(n))
    for t in mh.all_trees(n):
        classes = frozenset(c for L, R in t for c in list(L) + list(R))
        assert classes == expected_classes


@pytest.mark.parametrize("n", TOTAL_TREES.keys())
def test_splits_are_bipartitions(n):
    for t in mh.all_trees(n):
        for L, R in t:
            assert set(L) & set(R) == set()
            assert len(L) >= 1 and len(R) >= 1


@pytest.mark.parametrize("n", TOTAL_TREES.keys())
def test_split_count_per_tree(n):
    """Every n-class ND tree has exactly n-1 splits."""
    for t in mh.all_trees(n):
        assert len(t) == n - 1


def test_deterministic():
    """Calling twice returns the same list in the same order."""
    a = mh.all_trees(5)
    b = mh.all_trees(5)
    assert a == b


# ---- golden master: snapshot the sorted tree list for each small n ----

@pytest.mark.parametrize("n", [3, 4, 5])
def test_snapshot_small(n):
    """Snapshot the full tree list for n=3,4,5. Fast enough to commit."""
    fp = FIXTURES / f"trees_n{n}.json"
    trees = [str(t) for t in mh.all_trees(n)]
    if not fp.exists():
        fp.write_text(json.dumps(trees))
        pytest.skip(f"fixture created: {fp.name} — re-run to validate")
    expected = json.loads(fp.read_text())
    assert trees == expected


# ---- cross-check against existing cache_nd artifacts ----

@pytest.mark.parametrize("C, N, fname", [
    (6, 945,   "art_1_C6_N945_s0_d2.joblib"),
    (7, 10395, "art_1_C7_N10395_s0_d2.joblib"),
    (8, 135135, "art_1_C8_N15000_s0_d2.joblib"),  # N=15000 is a sample, not full
    (9, None,  "art_1_C9_N15000_s0_d2.joblib"),
    (10, None, "art_1_C10_N15000_s0_d2.joblib"),
])
def test_cross_check_cache(C, N, fname):
    """
    For small class counts (6, 7) the cache holds all trees — check exact set match.
    For larger counts (8+) the cache holds a sample — check that all cached trees
    are valid members of all_trees (i.e., no invented trees crept in).
    """
    cache = Path(__file__).parent.parent / "cache_nd" / fname
    if not cache.exists():
        pytest.skip(f"{fname} not found")

    art = load(cache)
    cached_set = set(str(t) for t in art["trees"])

    if C <= 7:
        # full enumeration: exact set match
        generated = set(str(t) for t in mh.all_trees(C))
        assert cached_set == generated, f"tree sets differ for C={C}"
    else:
        # sample: every cached tree must be structurally valid
        for t in art["trees"]:
            classes = frozenset(c for L, R in t for c in list(L) + list(R))
            assert classes == frozenset(range(C)), f"wrong classes in cached tree: {classes}"
            for L, R in t:
                assert set(L) & set(R) == set(), f"overlap in split {L}|{R}"
            assert len(t) == C - 1, f"wrong split count: {len(t)} for C={C}"
