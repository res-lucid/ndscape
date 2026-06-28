"""Smoke tests for the artifact build pipeline.

C=4, C=5 always rebuild from scratch (tiny, seconds).
C=6 loads from cache if available, otherwise rebuilds (~2 min).
No pre-existing cache files required for C=4 and C=5.

Run with: pytest -m integration
"""
import hashlib
import json
import numpy as np
import pytest
from pathlib import Path

from core import build_artifact, get_trees_and_artifact, split_analyzer

pytestmark = pytest.mark.integration

CACHE    = Path(__file__).parent.parent.parent / "cache_nd"
FIXTURES = Path(__file__).parent / "fixtures"

REQUIRED_KEYS = {
    "trees", "trees_plot", "coords_plot", "labels_plot",
    "k", "idx_score_in_plot", "trustworthiness", "metadata",
}
REQUIRED_META = {
    "C", "N_sampled", "exhaustive", "seed", "dim",
    "tokenizer", "traversal",
    "mds_raw_stress", "mds_stress1", "trustworthiness",
    "n_clusters", "silhouette_score", "packages",
}

# C=4 (15 trees) and C=5 (105 trees) always rebuild; C=6 (945 trees) loads from cache
BUILD_CASES = [
    (4, 15),
    (5, 105),
]


# ---- build from scratch: C=4, C=5 ----

@pytest.mark.parametrize("C, n_trees", BUILD_CASES)
def test_build_artifact_keys(C, n_trees):
    art = build_artifact(range(C), seed=0)
    assert REQUIRED_KEYS <= set(art.keys())

@pytest.mark.parametrize("C, n_trees", BUILD_CASES)
def test_build_artifact_tree_count(C, n_trees):
    assert len(build_artifact(range(C), seed=0)["trees"]) == n_trees

@pytest.mark.parametrize("C, n_trees", BUILD_CASES)
def test_build_artifact_coords_shape(C, n_trees):
    coords = np.asarray(build_artifact(range(C), seed=0)["coords_plot"])
    assert coords.shape == (n_trees, 2)
    assert np.isfinite(coords).all()

@pytest.mark.parametrize("C, n_trees", BUILD_CASES)
def test_build_artifact_metadata(C, n_trees):
    art  = build_artifact(range(C), seed=0)
    meta = art["metadata"]
    assert REQUIRED_META <= set(meta.keys())
    assert meta["C"] == C
    assert meta["N_sampled"] == n_trees
    assert meta["exhaustive"] is True
    assert meta["n_clusters"] == art["k"]

@pytest.mark.parametrize("C, n_trees", BUILD_CASES)
def test_build_artifact_labels_shape(C, n_trees):
    assert len(build_artifact(range(C), seed=0)["labels_plot"]) == n_trees

def test_build_artifact_k_is_n_clusters():
    art = build_artifact(range(5), seed=0)
    assert art["k"] == art["metadata"]["n_clusters"]


# ---- C=6 cache load (load-or-build, then validate) ----

def test_c6_artifact_structure():
    """C=6 has 945 exhaustive trees. Load from cache if present, else build."""
    art = get_trees_and_artifact(range(6), N=945, seed=0, cache_dir=str(CACHE))
    assert REQUIRED_KEYS <= set(art.keys())
    assert len(art["trees"]) == 945
    assert art["metadata"]["exhaustive"] is True
    assert art["metadata"]["C"] == 6
    coords = np.asarray(art["coords_plot"])
    assert coords.shape == (945, 2)
    assert np.isfinite(coords).all()

def test_c6_artifact_trees_hash():
    """Golden master: C=6 tree list must not change between runs."""
    art    = get_trees_and_artifact(range(6), N=945, seed=0, cache_dir=str(CACHE))
    digest = hashlib.md5("".join(str(t) for t in art["trees"]).encode()).hexdigest()
    fp     = FIXTURES / "artifact_trees_hash_C6.json"
    if not fp.exists():
        fp.write_text(json.dumps(digest))
        pytest.skip(f"fixture created: {fp.name} -- re-run to validate")
    assert digest == json.loads(fp.read_text()), \
        "C=6 tree list changed -- all downstream results are invalidated"

def test_c6_split_analyzer_hash():
    """Golden master: TF-IDF tokenisation of C=6 trees must not change."""
    art    = get_trees_and_artifact(range(6), N=945, seed=0, cache_dir=str(CACHE))
    analyze = split_analyzer(6)
    digest  = hashlib.md5(
        json.dumps([analyze(str(t)) for t in art["trees"]]).encode()
    ).hexdigest()
    fp = FIXTURES / "split_analyzer_hash_C6.json"
    if not fp.exists():
        fp.write_text(json.dumps(digest))
        pytest.skip(f"fixture created: {fp.name} -- re-run to validate")
    assert digest == json.loads(fp.read_text()), \
        "split_analyzer output changed -- MDS embedding is no longer meaningful"
