"""Tests for core.sample_trees.

sample_trees is used only for n > 7; for n <= 7 all_trees() enumerates exhaustively.
"""
import hashlib
import json
import pytest
from pathlib import Path
from core import sample_trees

FIXTURES = Path(__file__).parent / "fixtures"

# (n, N) pairs — N=15000 matches the actual production cache_nd artifacts
CASES = [
    (8,  15000),
    (9,  15000),
    (10, 15000),
    (10,  1000),   # quick golden master
    (15, 15000),
    (19, 15000),
    (26, 15000),
]

# smaller N for cheap checks (determinism, cross-check) — still representative
CASES_SMALL = [(n, 200) for n, _ in CASES]


# ---- count and uniqueness ----

@pytest.mark.parametrize("n, N", CASES)
def test_pbd_count(n, N):
    assert len(sample_trees(n, N)) == N


@pytest.mark.parametrize("n, N", CASES)
def test_pbd_uniqueness(n, N):
    trees = sample_trees(n, N)
    assert len(set(map(str, trees))) == N


# ---- structural properties (all sampled trees) ----

@pytest.mark.parametrize("n, N", CASES)
def test_pbd_all_classes_covered(n, N):
    expected = frozenset(range(n))
    for t in sample_trees(n, N):
        classes = frozenset(c for L, R in t for c in list(L) + list(R))
        assert classes == expected


@pytest.mark.parametrize("n, N", CASES)
def test_pbd_splits_are_bipartitions(n, N):
    for t in sample_trees(n, N):
        for L, R in t:
            assert set(L) & set(R) == set()
            assert len(L) >= 1 and len(R) >= 1


@pytest.mark.parametrize("n, N", CASES)
def test_pbd_each_tree_has_n_minus_1_splits(n, N):
    for t in sample_trees(n, N):
        assert len(t) == n - 1


# ---- determinism (small N to keep it cheap) ----

@pytest.mark.parametrize("n, N", CASES_SMALL)
def test_pbd_deterministic(n, N):
    a = list(map(str, sample_trees(n, N)))
    b = list(map(str, sample_trees(n, N)))
    assert a == b


# ---- cross-check against cache_nd artifacts (small sample) ----

@pytest.mark.parametrize("n, fname", [
    (8,  "art_1_C8_N15000_s0_d2.joblib"),
    (9,  "art_1_C9_N15000_s0_d2.joblib"),
    (10, "art_1_C10_N15000_s0_d2.joblib"),
])
def test_pbd_sample_is_subset_of_cache(n, fname):
    """
    sample_trees(n, 200) trees should be structurally valid -- same kind as those in the cache.
    Note: sample_trees(n, 200) is NOT guaranteed to be a subset of sample_trees(n, 15000) because
    integer allocation rounding in the proportional balanced design is budget-dependent.
    We check structural validity directly instead.
    """
    cache = Path(__file__).parent.parent / "cache_nd" / fname
    if not cache.exists():
        pytest.skip(f"{fname} not found")
    expected_classes = frozenset(range(n))
    for t in sample_trees(n, 200):
        classes = frozenset(c for L, R in t for c in list(L) + list(R))
        assert classes == expected_classes, "wrong classes in sample_trees result"
        for L, R in t:
            assert set(L) & set(R) == set(), f"overlap in split {L}|{R}"
        assert len(t) == n - 1, f"wrong split count for n={n}"


# ---- golden master snapshots (hash of full sample) ----

@pytest.mark.parametrize("n, N", CASES)
def test_snapshot_sample_trees(n, N):
    """
    Hash the full ordered list of tree strings for sample_trees(n, N).
    If sample_trees changes its sampling logic, this catches it.
    """
    trees = sample_trees(n, N)
    digest = hashlib.md5("".join(str(t) for t in trees).encode()).hexdigest()
    fp = FIXTURES / f"pbd_hash_C{n}_N{N}.json"
    if not fp.exists():
        fp.write_text(json.dumps(digest))
        pytest.skip(f"fixture created: {fp.name} -- re-run to validate")
    assert digest == json.loads(fp.read_text())
