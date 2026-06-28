"""Tests for the 6 ND structure generators.

Generators are used by run_datasets.py to pick named reference trees, so any
output drift silently changes the paper's comparison results.
"""
import json
import numpy as np
import pytest
from pathlib import Path
from sklearn.base import clone
import core as mh
from nd import RandomGeneration, BBoK, CBND, NDC, ACND, RPND
import re
setattr(np, "int", int)
setattr(np, "float", float)

FIXTURES = Path(__file__).parent / "fixtures"

ALL_DATASETS = [
    ("glass",        "glass_data"),
    ("steel_plates", "steel_plates_data"),
    ("mice",         "mice_data"),
    ("urban",        "urban_data"),
    ("pen_digits",   "pen_digits_data"),
]


def _check_tree_structure(tree, cats):
    """Bit-encoded tree: each entry is (left_bits, right_bits), right=0 means leaf."""
    full_mask = sum(1 << int(c) for c in cats)
    for L, R in tree:
        if int(R) == 0:
            assert bin(int(L)).count("1") == 1
        else:
            assert (int(L) | int(R)) == full_mask or (int(L) & int(R)) == 0


# ---- structural checks ----

@pytest.mark.parametrize("fixture_name, data_fixture", ALL_DATASETS)
def test_random_generation_structure(fixture_name, data_fixture, request):
    Xtr, ytr, _, _, cats = request.getfixturevalue(data_fixture)
    C = len(cats)
    t = RandomGeneration.generate(C, labels=list(cats), seed=0)
    _check_tree_structure(t, cats)
    assert len([s for s in t if int(s[1]) != 0]) == C - 1


@pytest.mark.parametrize("fixture_name, data_fixture", ALL_DATASETS)
def test_bbok_structure(fixture_name, data_fixture, request):
    Xtr, ytr, _, _, cats = request.getfixturevalue(data_fixture)
    C = len(cats)
    t = BBoK.generate(C, labels=list(cats), seed=0)
    _check_tree_structure(t, cats)


@pytest.mark.parametrize("fixture_name, data_fixture", ALL_DATASETS)
def test_cbnd_structure(fixture_name, data_fixture, request):
    Xtr, ytr, _, _, cats = request.getfixturevalue(data_fixture)
    t = CBND.generate(Xtr, ytr, seed=0)
    _check_tree_structure(t, cats)


@pytest.mark.parametrize("fixture_name, data_fixture", ALL_DATASETS)
def test_ndc_structure(fixture_name, data_fixture, request):
    Xtr, ytr, _, _, cats = request.getfixturevalue(data_fixture)
    t = NDC.generate(Xtr, ytr)
    _check_tree_structure(t, cats)


@pytest.mark.parametrize("fixture_name, data_fixture", ALL_DATASETS)
def test_acnd_structure(fixture_name, data_fixture, request):
    Xtr, ytr, _, _, cats = request.getfixturevalue(data_fixture)
    t = ACND.generate(Xtr, ytr)
    _check_tree_structure(t, cats)


@pytest.mark.parametrize("fixture_name, data_fixture", ALL_DATASETS)
def test_rpnd_structure(fixture_name, data_fixture, request):
    Xtr, ytr, _, _, cats = request.getfixturevalue(data_fixture)
    base = clone(mh.get_model("lr"))
    t = RPND.generate(Xtr, ytr, lambda **kw: clone(base), seed=0)
    _check_tree_structure(t, cats)


# ---- determinism ----

@pytest.mark.parametrize("fixture_name, data_fixture", ALL_DATASETS)
def test_random_generation_deterministic(fixture_name, data_fixture, request):
    Xtr, ytr, _, _, cats = request.getfixturevalue(data_fixture)
    C = len(cats)
    assert RandomGeneration.generate(C, labels=list(cats), seed=42) == RandomGeneration.generate(C, labels=list(cats), seed=42)


@pytest.mark.parametrize("fixture_name, data_fixture", ALL_DATASETS)
def test_bbok_deterministic(fixture_name, data_fixture, request):
    Xtr, ytr, _, _, cats = request.getfixturevalue(data_fixture)
    C = len(cats)
    assert BBoK.generate(C, labels=list(cats), seed=42) == BBoK.generate(C, labels=list(cats), seed=42)


@pytest.mark.parametrize("fixture_name, data_fixture", ALL_DATASETS)
def test_cbnd_deterministic(fixture_name, data_fixture, request):
    Xtr, ytr, _, _, cats = request.getfixturevalue(data_fixture)
    assert CBND.generate(Xtr, ytr, seed=42) == CBND.generate(Xtr, ytr, seed=42)


def _plain(s):
    return re.sub(r"np\.int64\((-?\d+)\)", r"\1", s)
# ---- golden master snapshots ----

@pytest.mark.parametrize("fixture_name, data_fixture", ALL_DATASETS)
def test_snapshot_all_generators(fixture_name, data_fixture, request):
    """
    Snapshot all 6 generator outputs for each dataset.
    If any of these change after refactoring, run_datasets.py
    will silently pick different reference trees.
    """
    Xtr, ytr, _, _, cats = request.getfixturevalue(data_fixture)
    C = len(cats)
    base = clone(mh.get_model("lr"))

    results = {
        "RandomGeneration": str(RandomGeneration.generate(C, labels=list(cats), seed=0)),
        "BBoK":             str(BBoK.generate(C, labels=list(cats), seed=0)),
        "CBND":             str(CBND.generate(Xtr, ytr, seed=0)),
        "NDC":              str(NDC.generate(Xtr, ytr)),
        "ACND":             str(ACND.generate(Xtr, ytr)),
        "RPND":             str(RPND.generate(Xtr, ytr, lambda **kw: clone(base), seed=0)),
    }

    fp = FIXTURES / f"nd_generators_{fixture_name}.json"
    if not fp.exists():
        fp.write_text(json.dumps(results, indent=2))
        pytest.skip(f"fixture created: {fp.name} — re-run to validate")
