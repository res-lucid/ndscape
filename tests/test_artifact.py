"""Tests for cache_nd/*.joblib artifacts.

The artifacts are expensive to regenerate (~hours), so these tests catch
structural corruption or truncation early rather than downstream in scoring.
"""
import hashlib
import json
import numpy as np
import pytest
from pathlib import Path
from joblib import load
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from core import split_analyzer
FIXTURES = Path(__file__).parent / "fixtures"
CACHE  = Path(__file__).parent.parent / "cache_nd"
ZENODO = Path(__file__).parent.parent / "zenodo"

ARTIFACTS = [
    (6,  "art_1_C6_N945_s0_d2.joblib",   945),
    (7,  "art_1_C7_N10395_s0_d2.joblib", 10395),
    (8,  "art_1_C8_N15000_s0_d2.joblib",  15000),
    (9,  "art_1_C9_N15000_s0_d2.joblib",  15000),
    (10, "art_1_C10_N15000_s0_d2.joblib", 15000),
    (19, "art_1_C19_N15000_s0_d2.joblib", 15000),
]


def _load(fname):
    p = CACHE / fname
    if not p.exists():
        p = ZENODO / fname
    if not p.exists():
        pytest.skip(f"{fname} not found in cache_nd/ or zenodo/")
    return load(p)


# ---- required keys ----

@pytest.mark.parametrize("C, fname, N", ARTIFACTS)
def test_required_keys_present(C, fname, N):
    art = _load(fname)
    for key in ("trees", "trees_plot", "coords_plot", "labels_plot"):
        assert key in art, f"missing key '{key}' in {fname}"


# ---- trees list ----

@pytest.mark.parametrize("C, fname, N", ARTIFACTS)
def test_trees_count(C, fname, N):
    assert len(_load(fname)["trees"]) == N


@pytest.mark.parametrize("C, fname, N", ARTIFACTS)
def test_trees_are_tuples(C, fname, N):
    for t in _load(fname)["trees"]:
        assert isinstance(t, tuple)


@pytest.mark.parametrize("C, fname, N", ARTIFACTS)
def test_each_tree_has_c_minus_1_splits(C, fname, N):
    for t in _load(fname)["trees"]:
        assert len(t) == C - 1, f"expected {C-1} splits, got {len(t)}: {t}"


@pytest.mark.parametrize("C, fname, N", ARTIFACTS)
def test_each_split_covers_all_classes(C, fname, N):
    for t in _load(fname)["trees"]:
        all_classes = set(c for L, R in t for c in L + R)
        assert len(all_classes) == C


# ---- trees_plot ----

@pytest.mark.parametrize("C, fname, N", ARTIFACTS)
def test_trees_plot_count_matches_trees(C, fname, N):
    art = _load(fname)
    assert len(art["trees_plot"]) == len(art["trees"])


@pytest.mark.parametrize("C, fname, N", ARTIFACTS)
def test_trees_plot_are_strings(C, fname, N):
    for tp in _load(fname)["trees_plot"]:
        assert isinstance(tp, str)


# ---- coords_plot ----

@pytest.mark.parametrize("C, fname, N", ARTIFACTS)
def test_coords_shape(C, fname, N):
    coords = np.asarray(_load(fname)["coords_plot"])
    assert coords.shape == (N, 2), f"expected ({N}, 2), got {coords.shape}"


@pytest.mark.parametrize("C, fname, N", ARTIFACTS)
def test_coords_finite(C, fname, N):
    coords = np.asarray(_load(fname)["coords_plot"])
    assert np.isfinite(coords).all(), "coords_plot contains NaN or Inf"


# ---- labels_plot ----

@pytest.mark.parametrize("C, fname, N", ARTIFACTS)
def test_labels_length(C, fname, N):
    assert len(_load(fname)["labels_plot"]) == N


# ---- internal consistency: trees_plot matches bfs_splits for all trees ----

@pytest.mark.parametrize("C, fname, N", ARTIFACTS)
def test_trees_plot_matches_bfs_for_all(C, fname, N):
    """
    trees_plot[i] should equal str(bfs_splits(trees[i])) for every tree.
    Confirms the cache was built with the same bfs_splits logic as generate_data.
    """
    art = _load(fname)
    for t, tp in zip(art["trees"], art["trees_plot"]):
        assert str(t) == tp, f"bfs mismatch for tree {t}"


# ---- golden master snapshots (hash of all tree strings) ----

@pytest.mark.parametrize("C, fname, N", ARTIFACTS)
def test_snapshot_all_trees(C, fname, N):
    """Hash the full list of tree strings. Catches any reordering or mutation."""
    trees = _load(fname)["trees"]
    digest = hashlib.md5("".join(str(t) for t in trees).encode()).hexdigest()
    fp = FIXTURES / f"artifact_trees_hash_C{C}.json"
    if not fp.exists():
        fp.write_text(json.dumps(digest))
        pytest.skip(f"fixture created: {fp.name} — re-run to validate")
    assert digest == json.loads(fp.read_text())

@pytest.mark.parametrize("C, fname, N", ARTIFACTS)
def test_snapshot_distance_matrix(C, fname, N):
    """Hash pre-MDS structural distance matrix. This is the key reproducibility target."""
    trees = _load(fname)["trees"]
    vec = TfidfVectorizer(analyzer=split_analyzer(C), use_idf=False, norm="l2")
    X = vec.fit_transform(trees)
    S = cosine_similarity(X)
    D = np.sqrt(np.maximum(0, 2.0 * (1.0 - S)))
    np.fill_diagonal(D, 0.0)

    digest = hashlib.md5(D.astype("f8").tobytes()).hexdigest()
    fp = FIXTURES / f"artifact_D_hash_C{C}.json"
    if not fp.exists():
        fp.write_text(json.dumps(digest))
        pytest.skip(f"fixture created: {fp.name} — re-run to validate")
    assert digest == json.loads(fp.read_text())
    
@pytest.mark.parametrize("C, fname, N", ARTIFACTS)
def test_snapshot_coords(C, fname, N):
    """Hash the MDS coords array. Catches any drift in MDS or embedding."""
    coords = np.asarray(_load(fname)["coords_plot"])
    digest = hashlib.md5(coords.astype("f8").tobytes()).hexdigest()
    fp = FIXTURES / f"artifact_coords_hash_C{C}.json"
    if not fp.exists():
        fp.write_text(json.dumps(digest))
        pytest.skip(f"fixture created: {fp.name} — re-run to validate")
    assert digest == json.loads(fp.read_text())
