"""Tests for core.nd_predict_proba.

nd_predict_proba reads config.X / config.y for on-demand model training.
Stale config state produces silently wrong probabilities, not a crash.
"""
import json
import numpy as np
import pytest
from pathlib import Path
import config
import core as mh
from core import sample_trees

FIXTURES = Path(__file__).parent / "fixtures"


def _set_config(Xtr, ytr):
    config.X = Xtr
    config.y = ytr
    config.model_cache = {}


# ---- shape and probability constraints ----

ALL_DATASETS = [
    "glass_data",
    "steel_plates_data",
    "mice_data",
    "urban_data",
    "pen_digits_data",
]


@pytest.mark.parametrize("data_fixture", ALL_DATASETS)
def test_output_shape(data_fixture, request):
    Xtr, ytr, Xte, yte, cats = request.getfixturevalue(data_fixture)
    C = len(cats)
    _set_config(Xtr, ytr)
    T = mh.all_trees(C)[0] if C <= 7 else sample_trees(C, 1)[0]
    proba, classes = mh.nd_predict_proba(Xte, T, list(cats))
    assert proba.shape == (Xte.shape[0], C)


@pytest.mark.parametrize("data_fixture", ALL_DATASETS)
def test_rows_sum_to_one(data_fixture, request):
    Xtr, ytr, Xte, yte, cats = request.getfixturevalue(data_fixture)
    C = len(cats)
    _set_config(Xtr, ytr)
    T = mh.all_trees(C)[0] if C <= 7 else sample_trees(C, 1)[0]
    proba, _ = mh.nd_predict_proba(Xte, T, list(cats))
    np.testing.assert_allclose(proba.sum(axis=1), np.ones(Xte.shape[0]), atol=1e-6)


@pytest.mark.parametrize("data_fixture", ALL_DATASETS)
def test_probabilities_non_negative(data_fixture, request):
    Xtr, ytr, Xte, yte, cats = request.getfixturevalue(data_fixture)
    C = len(cats)
    _set_config(Xtr, ytr)
    T = mh.all_trees(C)[0] if C <= 7 else sample_trees(C, 1)[0]
    proba, _ = mh.nd_predict_proba(Xte, T, list(cats))
    assert (proba >= 0).all()


@pytest.mark.parametrize("data_fixture", ALL_DATASETS)
def test_classes_returned_match_input(data_fixture, request):
    Xtr, ytr, Xte, yte, cats = request.getfixturevalue(data_fixture)
    C = len(cats)
    _set_config(Xtr, ytr)
    T = mh.all_trees(C)[0] if C <= 7 else sample_trees(C, 1)[0]
    _, classes = mh.nd_predict_proba(Xte, T, list(cats))
    assert list(classes) == list(cats)


# ---- model_cache is populated ----

def test_model_cache_populated(glass_data):
    Xtr, ytr, Xte, yte, cats = glass_data
    _set_config(Xtr, ytr)
    T = mh.all_trees(len(cats))[0]
    assert len(config.model_cache) == 0
    mh.nd_predict_proba(Xte, T, list(cats))
    assert len(config.model_cache) > 0


def test_model_cache_reused(glass_data):
    """Second call should not retrain — cache size stays the same."""
    Xtr, ytr, Xte, yte, cats = glass_data
    _set_config(Xtr, ytr)
    T = mh.all_trees(len(cats))[0]
    mh.nd_predict_proba(Xte, T, list(cats))
    size_after_first = len(config.model_cache)
    mh.nd_predict_proba(Xte, T, list(cats))
    assert len(config.model_cache) == size_after_first


# ---- different trees give different outputs ----

def test_different_trees_differ(glass_data):
    Xtr, ytr, Xte, yte, cats = glass_data
    trees = mh.all_trees(len(cats))
    _set_config(Xtr, ytr)
    p0, _ = mh.nd_predict_proba(Xte, trees[0], list(cats))
    _set_config(Xtr, ytr)
    p1, _ = mh.nd_predict_proba(Xte, trees[1], list(cats))
    assert not np.allclose(p0, p1)


# ---- golden master snapshot ----

def test_snapshot_all_predictions(glass_data):
    """
    Snapshot the predicted class (argmax) for every test sample.
    Uses the first tree for 6 classes (glass). If this changes,
    get_or_train_split_model or nd_predict_proba logic has drifted.
    """
    Xtr, ytr, Xte, yte, cats = glass_data
    _set_config(Xtr, ytr)
    T = mh.all_trees(len(cats))[0]
    proba, classes = mh.nd_predict_proba(Xte, T, list(cats))
    predicted = [int(classes[i]) for i in np.argmax(proba, axis=1)]

    fp = FIXTURES / "nd_predict_glass_all.json"
    if not fp.exists():
        fp.write_text(json.dumps(predicted))
        pytest.skip(f"fixture created: {fp.name} — re-run to validate")
    assert predicted == json.loads(fp.read_text())


