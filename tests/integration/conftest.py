"""
Integration test configuration.

Integration tests are skipped by default (pytest.ini sets -m "not integration").
Run them explicitly with:

    pytest -m integration
    pytest -m integration -v                   # verbose
    pytest -m integration tests/integration/   # only integration tests
"""
import sys
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

import config

DATA = ROOT / "zenodo" / "data"
CACHE_ND = ROOT / "cache_nd"
REAL_DIR = ROOT / "real-data-bias-analysis"
FIXTURES = Path(__file__).parent / "fixtures"
FIXTURES.mkdir(exist_ok=True)


@pytest.fixture(autouse=True)
def reset_config():
    config.model_cache = {}
    config.X = np.array([]).reshape(0, 1)
    config.y = np.array([])
    config.X_test = np.array([]).reshape(0, 1)
    config.y_test = np.array([])
    yield
    config.model_cache = {}


def _load_csv(fname):
    df = pd.read_csv(DATA / fname)
    X = df.drop("y", axis=1).to_numpy(float)
    y = df["y"].to_numpy(int)
    cats = tuple(sorted(set(y.tolist())))
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
    return Xtr, ytr, Xte, yte, cats


def _load_urban():
    """Urban land cover has a predefined train/test split column."""
    df = pd.read_csv(DATA / "urban_land_cover.csv")
    tr_idx = df.index[df["split"] == "train"].to_numpy()
    te_idx = df.index[df["split"] == "test"].to_numpy()
    df = df.drop("split", axis=1)
    y = df["y"].to_numpy(int)
    X = df.drop("y", axis=1).to_numpy(float)
    cats = tuple(sorted(set(y.tolist())))
    return X[tr_idx], y[tr_idx], X[te_idx], y[te_idx], cats


@pytest.fixture(scope="module")
def glass():
    return _load_csv("glass_identification.csv")


@pytest.fixture(scope="module")
def steel_plates():
    return _load_csv("steel_plates_faults.csv")


@pytest.fixture(scope="module")
def mice():
    return _load_csv("mice_protein.csv")


@pytest.fixture(scope="module")
def urban():
    return _load_urban()


@pytest.fixture(scope="module")
def pen_digits():
    return _load_csv("pen_based_recognition_of_handwritten_digits_81.csv")


@pytest.fixture(scope="module")
def soybean():
    return _load_csv("soybean_large_122.csv")
