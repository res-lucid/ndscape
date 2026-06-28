import sys
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from sklearn.model_selection import train_test_split

# make the simulation root importable
ROOT = Path(__file__).parent.parent
TESTS = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TESTS))

import config

FIXTURES = Path(__file__).parent / "fixtures"
FIXTURES.mkdir(exist_ok=True)

CACHE_ND = ROOT / "cache_nd"
DATA = ROOT / "zenodo" / "data"
REAL_RESULTS = ROOT / "real-data-bias-analysis"
BOOT_RESULTS = ROOT / "bias-analysis" / "raw_simulation_results"


# ---- config state reset ----

@pytest.fixture(autouse=True)
def reset_config():
    config.model_cache = {}
    config.X = np.array([]).reshape(0, 1)
    config.y = np.array([])
    config.X_test = np.array([]).reshape(0, 1)
    config.y_test = np.array([])
    yield
    config.model_cache = {}


# ---- shared data fixtures ----

@pytest.fixture(scope="session")
def glass_data():
    df = pd.read_csv(DATA / "glass_identification.csv")
    X = df.drop("y", axis=1).to_numpy(float)
    y = df["y"].to_numpy(int)
    cats = tuple(sorted(np.unique(y).tolist()))
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
    return Xtr, ytr, Xte, yte, cats


@pytest.fixture(scope="session")
def steel_plates_data():
    df = pd.read_csv(DATA / "steel_plates_faults.csv")
    X = df.drop("y", axis=1).to_numpy(float)
    y = df["y"].to_numpy(int)
    cats = tuple(sorted(np.unique(y).tolist()))
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
    return Xtr, ytr, Xte, yte, cats


@pytest.fixture(scope="session")
def mice_data():
    df = pd.read_csv(DATA / "mice_protein.csv")
    X = df.drop("y", axis=1).to_numpy(float)
    y = df["y"].to_numpy(int)
    cats = tuple(sorted(np.unique(y).tolist()))
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
    return Xtr, ytr, Xte, yte, cats


@pytest.fixture(scope="session")
def urban_data():  # 9 classes, 675 rows, 148 features
    df = pd.read_csv(DATA / "urban_land_cover.csv")
    if "split" in df.columns:
        tr = df.index[df["split"] == "train"].to_numpy()
        te = df.index[df["split"] == "test"].to_numpy()
        df = df.drop("split", axis=1)
        y_full = df["y"].to_numpy(int)
        X_full = df.drop("y", axis=1).to_numpy(float)
        cats = tuple(sorted(np.unique(y_full).tolist()))
        Xtr_raw = X_full[tr].astype(float, copy=True)
        Xte_raw = X_full[te].astype(float, copy=True)
        ytr, yte = y_full[tr], y_full[te]
        Xtr, Xte = Xtr_raw.copy(), Xte_raw.copy()
    else:
        X = df.drop("y", axis=1).to_numpy(float)
        y = df["y"].to_numpy(int)
        cats = tuple(sorted(np.unique(y).tolist()))
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
    return Xtr, ytr, Xte, yte, cats


@pytest.fixture(scope="session")
def pen_digits_data():  # 10 classes, 10992 rows, 16 features
    df = pd.read_csv(DATA / "pen_based_recognition_of_handwritten_digits_81.csv")
    X = df.drop("y", axis=1).to_numpy(float)
    y = df["y"].to_numpy(int)
    cats = tuple(sorted(np.unique(y).tolist()))
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
    return Xtr, ytr, Xte, yte, cats


# ---- snapshot helpers ----


