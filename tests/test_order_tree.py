"""Tests for order_tree: reorders tree splits into BFS order (root first, left-to-right)."""
import hashlib
import json
import pytest
from pathlib import Path
from joblib import load
from helpers import order_tree
import core as mh

FIXTURES = Path(__file__).parent / "fixtures"


def _all_classes(tree):
    return frozenset(c for L, R in tree for c in list(L) + list(R))


def _is_bfs_order(tree):
    """Check that every split's children appear after the parent in the list."""
    by_union = {frozenset(L + R): i for i, (L, R) in enumerate(tree)}
    for i, (L, R) in enumerate(tree):
        for child in (L, R):
            if len(child) > 1:
                key = frozenset(child)
                assert key in by_union, f"child {child} not found as a split"
                assert by_union[key] > i, f"child {child} appears before parent at index {i}"
    return True


# ---- structural tests ----

@pytest.mark.parametrize("n", [3, 4, 5, 6])
def test_order_tree_is_bfs(n):
    """order_tree output should be in BFS order."""
    for t in mh.all_trees(n):
        ordered = order_tree(t, n)
        assert _is_bfs_order(ordered)


@pytest.mark.parametrize("n", [3, 4, 5, 6])
def test_order_tree_preserves_classes(n):
    for t in mh.all_trees(n):
        assert _all_classes(order_tree(t, n)) == _all_classes(t)


@pytest.mark.parametrize("n", [3, 4, 5, 6])
def test_order_tree_preserves_split_count(n):
    for t in mh.all_trees(n):
        assert len(order_tree(t, n)) == len(t)


@pytest.mark.parametrize("n", [3, 4, 5, 6])
def test_order_tree_preserves_split_set(n):
    """The set of splits is unchanged — just reordered."""
    for t in mh.all_trees(n):
        original_splits = set(str(s) for s in t)
        ordered_splits = set(str(s) for s in order_tree(t, n))
        assert original_splits == ordered_splits


@pytest.mark.parametrize("n", [3, 4, 5, 6])
def test_order_tree_idempotent(n):
    """Calling order_tree twice gives the same result."""
    for t in mh.all_trees(n):
        once = order_tree(t, n)
        twice = order_tree(once, n)
        assert once == twice


# ---- cross-check with real cache trees ----

@pytest.mark.parametrize("C, fname", [
    (6, "art_1_C6_N945_s0_d2.joblib"),
    (7, "art_1_C7_N10395_s0_d2.joblib"),
])
def test_order_tree_matches_cache(C, fname):
    """
    Trees in cache_nd come from _gen which uses DFS ordering, not BFS.
    Applying order_tree should be idempotent — but we only verify structure.
    """
    cache = Path(__file__).parent.parent / "cache_nd" / fname
    if not cache.exists():
        pytest.skip(f"{fname} not found")
    art = load(cache)
    for t in art["trees"]:
        classes = frozenset(c for L, R in t for c in list(L) + list(R))
        assert classes == frozenset(range(C)), "wrong classes in tree"
        assert len(t) == C - 1, "wrong split count"


# ---- golden master snapshots ----

@pytest.mark.parametrize("n", [3, 4, 5, 6])
def test_snapshot_all_trees(n):
    """Hash order_tree output for every tree at each class count."""
    trees = mh.all_trees(n)
    digest = hashlib.md5("".join(str(order_tree(t, n)) for t in trees).encode()).hexdigest()
    fp = FIXTURES / f"order_tree_hash_n{n}.json"
    if not fp.exists():
        fp.write_text(json.dumps(digest))
        pytest.ski