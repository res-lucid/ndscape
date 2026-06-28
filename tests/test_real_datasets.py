"""Tests against golden-master fixtures for real dataset scoring.

Fixtures in tests/fixtures/dataset_scores_<dataset>.json hold MD5 hashes of
the score and logloss columns (logloss rounded to 8dp to absorb float noise).
modelvar stats use rtol=1e-3 due to ~3e-6 cross-machine spread in accumulated ops.
A hash failure means scoring or tree-encoding logic changed.
"""
import ast
import hashlib
import json
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

FIXTURES = Path(__file__).parent / "fixtures"
ROOT     = Path(__file__).parent.parent
BOOT_DIR = ROOT / "bias-analysis" / "raw_simulation_results"
REAL_DIR = ROOT / "real-data-bias-analysis"

DATASETS = [
    ("glass_identification",                           6,    945),
    ("steel_plates_faults",                            7,  10395),
    ("mice_protein",                                   8,  15000),
    ("urban_land_cover",                               9,  15000),
    ("pen_based_recognition_of_handwritten_digits_81", 10, 15000),
    ("soybean_large_122",                              19, 15000),
]


def _load_boot_csv():
    csvs = sorted(BOOT_DIR.glob("boot_results_*_runs_.csv"))
    if not csvs:
        pytest.skip("no boot_results CSVs found")
    return pd.read_csv(csvs[0])


def _load_tree_metrics(dataset, model="lr"):
    xlsx = REAL_DIR / dataset / f"v_{dataset}_{model}_cluster_tables.xlsx"
    if not xlsx.exists():
        pytest.skip(f"{xlsx.name} not found — run run_datasets.py first")
    return pd.read_excel(xlsx, sheet_name="tree_metrics")


def _raw_hash(series):
    return hashlib.md5(series.to_numpy().astype("f8").tobytes()).hexdigest()


def _rounded_hash(series, dp):
    return hashlib.md5(series.round(dp).to_numpy().astype("f8").tobytes()).hexdigest()


def _load_fixture(dataset):
    fp = FIXTURES / f"dataset_scores_{dataset}.json"
    if not fp.exists():
        pytest.skip(f"fixture {fp.name} not found")
    return json.loads(fp.read_text())


# ========== boot_results CSV tests ==========

def test_boot_csv_required_columns():
    df = _load_boot_csv()
    for col in ("simulation_id", "experiment_name", "n_samples",
                "tree_idx", "tree_text", "bias2", "var", "mse"):
        assert col in df.columns


def test_boot_csv_no_nulls_in_key_columns():
    df = _load_boot_csv()
    for col in ("tree_text", "bias2", "var", "mse"):
        assert df[col].isna().sum() == 0, f"nulls found in {col}"


def test_boot_csv_mse_equals_bias2_plus_var():
    """mse should equal bias2 + var (bias-variance decomposition)."""
    df = _load_boot_csv()
    np.testing.assert_allclose(
        df["mse"].values, (df["bias2"] + df["var"]).values, rtol=1e-5
    )


def test_boot_csv_tree_text_parses():
    df = _load_boot_csv()
    for text in df["tree_text"]:
        assert isinstance(ast.literal_eval(text), tuple)


def test_boot_csv_all_csvs_have_same_columns():
    csvs = sorted(BOOT_DIR.glob("boot_results_*_runs_.csv"))
    if len(csvs) < 2:
        pytest.skip("need at least 2 CSVs")
    ref_cols = pd.read_csv(csvs[0], nrows=0).columns.tolist()
    for csv in csvs[1:]:
        cols = pd.read_csv(csv, nrows=0).columns.tolist()
        assert cols == ref_cols, f"{csv.name} has unexpected columns"


def test_boot_csv_count():
    csvs = list(BOOT_DIR.glob("boot_results_*_runs_.csv"))
    if not csvs:
        pytest.skip("no boot_results CSVs found — run run_simulations.py first")
    fp = FIXTURES / "boot_csv_count.json"
    if not fp.exists():
        fp.write_text(json.dumps(len(csvs)))
        pytest.skip(f"fixture created: {fp.name} — re-run to validate")
    assert len(csvs) == json.loads(fp.read_text())


# ========== structural checks ==========

@pytest.mark.parametrize("dataset, C, N", DATASETS)
def test_tree_metrics_row_count(dataset, C, N):
    """One row per tree in the artifact."""
    assert len(_load_tree_metrics(dataset)) == N


@pytest.mark.parametrize("dataset, C, N", DATASETS)
def test_score_in_unit_interval(dataset, C, N):
    df = _load_tree_metrics(dataset)
    assert df["score"].between(0, 1).all()
    assert df["score"].min() > 0.05


@pytest.mark.parametrize("dataset, C, N", DATASETS)
def test_no_null_scores(dataset, C, N):
    assert _load_tree_metrics(dataset)["score"].isna().sum() == 0


@pytest.mark.parametrize("dataset, C, N", DATASETS)
def test_error_rate_complement_of_score(dataset, C, N):
    df = _load_tree_metrics(dataset)
    np.testing.assert_allclose(
        df["error_rate"].values, 1 - df["score"].values, atol=1e-6
    )


@pytest.mark.parametrize("dataset, C, N", DATASETS)
def test_model_var_non_negative(dataset, C, N):
    assert (_load_tree_metrics(dataset)["model_var"] >= 0).all()


@pytest.mark.parametrize("dataset, C, N", DATASETS)
def test_trees_cover_correct_class_count(dataset, C, N):
    """Every tree in the xlsx covers exactly C classes."""
    df = _load_tree_metrics(dataset)
    for tree_str in df["tree"]:
        parsed = ast.literal_eval(str(tree_str))
        classes = set(c for L, R in parsed for c in L + R)
        assert len(classes) == C


# ========== golden-master score snapshots ==========
#
# All three tests sort rows by tree string before hashing so they pass even
# if run_datasets.py writes trees in a different order than the reference xlsx.

@pytest.mark.parametrize("dataset, C, N", DATASETS)
def test_snapshot_score_sorted(dataset, C, N):
    """Score column matches golden master (bit-exact after sorting by tree).
    Score = n_correct/n_test is an exact ratio; any change here means the
    model or test split changed."""
    df  = _load_tree_metrics(dataset).sort_values("tree")
    ref = _load_fixture(dataset)
    assert _raw_hash(df["score"]) == ref["score_sorted_hash"], (
        f"{dataset}: score hash mismatch — scoring or tree ordering changed"
    )


@pytest.mark.parametrize("dataset, C, N", DATASETS)
def test_snapshot_logloss_sorted(dataset, C, N):
    """Log-loss matches golden master to 8 decimal places (sorted by tree).
    Hashed at 8dp to absorb machine-epsilon differences (~1e-15)."""
    df  = _load_tree_metrics(dataset).sort_values("tree")
    ref = _load_fixture(dataset)
    assert _rounded_hash(df["logloss"], 8) == ref["logloss_sorted_hash"], (
        f"{dataset}: logloss hash mismatch"
    )


@pytest.mark.parametrize("dataset, C, N", DATASETS)
def test_snapshot_modelvar_summary(dataset, C, N):
    """Bootstrap model-variance summary stats match golden master within 0.1%.
    A hash is not used here because accumulated float ops give ~3e-6 relative
    spread across machines; summary stats at rtol=1e-3 give a meaningful
    regression guard without false positives."""
    df  = _load_tree_metrics(dataset)
    ref = _load_fixture(dataset)
    mv  = df["model_var"].to_numpy()
    np.testing.assert_allclose(mv.mean(), ref["modelvar_mean"], rtol=1e-3,
        err_msg=f"{dataset}: model_var mean mismatch")
    np.testing.assert_allclose(mv.min(),  ref["modelvar_min"],  rtol=1e-3,
        err_msg=f"{dataset}: model_var min mismatch")
    np.testing.assert_allclose(mv.max(),  ref["modelvar_max"],  rtol=1e-3,
        err_msg=f"{dataset}: model_var max mismatch")
