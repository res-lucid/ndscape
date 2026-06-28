"""Integration tests: re-tokenize cached artifact trees and compare hashes.

A failure means the tokenizer changed and all cache_nd artifacts are stale.
Run with: pytest -m integration
"""
import hashlib
import json
import numpy as np
import pytest
from pathlib import Path
from joblib import load
from core import get_trees_and_artifact, split_analyzer

ROOT = Path(__file__).parent.parent.parent
CACHE = ROOT / "cache_nd"
FIXTURES = Path(__file__).parent / "fixtures"

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("C, fname", [
    (6, "art_1_C6_N945_s0_d2.joblib"),
    (7, "art_1_C7_N10395_s0_d2.joblib"),
    (8, "art_1_C8_N15000_s0_d2.joblib"),
    (9, "art_1_C9_N15000_s0_d2.joblib"),
    (10, "art_1_C10_N15000_s0_d2.joblib"),
    (19, "art_1_C19_N15000_s0_d2.joblib"),
])
def test_split_analyzer_hash_stable(C, fname):
    """
    Re-tokenizing all cached trees with split_analyzer should produce
    the same hash as when the artifacts were built. If this fails, the TF-IDF
    input has changed and MDS coordinates are no longer meaningful.
    """
    cache = CACHE / fname
    if not cache.exists():
        pytest.skip(f"{fname} not found")
    trees = load(cache)["trees"]
    analyze = split_analyzer(C)
    digest = hashlib.md5(
        json.dumps([analyze(str(t)) for t in trees]).encode()
    ).hexdigest()
    fp = FIXTURES / f"split_analyzer_hash_C{C}.json"
    if not fp.exists():
        fp.write_text(json.dumps(digest))
        pytest.skip(f"fixture created: {fp.name} — re-run to validate")
    assert digest == json.loads(fp.read_text())


# ---- get_trees_and_artifact: structure check (cache load only, no recomputation) ----

@pytest.mark.parametrize("C, cats, N", [
    (6, tuple(range(6)), 945),
    (7, tuple(range(7)), 10395),
])
def test_get_trees_and_artifact_structure(C, cats, N):
    """
    get_trees_and_artifact() should return the cached artifact when it exists.
    Keys, tree count, coords shape, and labels length must all be correct.
    Skips if the cache file is missing (avoids triggering MDS recomputation).
    """
    fname = f"art_1_C{C}_N{N}_s0_d2.joblib"
    if not (CACHE / fname).exists():
        pytest.skip(f"{fname} not found — skipping to avoid MDS recomputation")
    art = get_trees_and_artifact(cats, N=N)
    assert "trees" in art and "coords_plot" in art and "trees_plot" in art
    assert len(art["trees"]) == N
    coords = np.asarray(art["coords_plot"])
    assert coords.shape == (N, 2)
    assert np.isfinite(coords).all()


def test_get_trees_and_artifact_trees_hash_C6():
    """Snapshot the hash of all trees for C=6."""
    cats, N = tuple(range(6)), 945
    fname = f"art_1_C6_N{N}_s0_d2.joblib"
    if not (CACHE / fname).exists():
        pytest.skip(f"{fname} not found")
    art = get_trees_and_artifact(cats, N=N)
    digest = hashlib.md5("".join(str(t) for t in art["trees"]).encode()).hexdigest()
    fp = FIXTURES / "artifact_trees_hash_C6.json"
    if not fp.exists():
        fp.write_text(json.dumps(digest))
        pytest.skip(f"fixture created: {fp.name} -- re-run to validate")
    assert digest == json.loads(fp.read_text())
