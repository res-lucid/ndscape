"""Integration smoke tests: decision-tree generator simulation pipeline.

Tests the full run_simulation() call with generator_type='decisiontree',
using a minimal config (C=3, R=2 reps, n_test=200, n_train=100) to verify
the DT path produces valid bias-variance CSV output without running the
expensive full simulation.

Runtime: seconds.
Run with: pytest -m integration
"""
import ast
import numpy as np
import pandas as pd
import pytest
from core import all_trees
from run_simulations import run_simulation
from scripts.csv_to_configs import _parse_row

pytestmark = pytest.mark.integration


# Minimal C=3 DT config

_DT_SPECS = [
    {"depth": 1, "feat": [0], "thr": [0.0], "leafp": [0.2, 0.8]},
    {"depth": 1, "feat": [0], "thr": [0.0], "leafp": [0.3, 0.7]},
]

_DT_CFG = {
    "simulation_id":    "dt_smoke_1",
    "experiment":       "3_class_2_features_dense_sparsity",
    "seed":             42,
    "n_classes":        3,
    "n_features":       2,
    "generator_type":   "decisiontree",
    "nd_params":        {
        str(((0, 1), (2,))): _DT_SPECS[0],
        str(((0,),   (1,))): _DT_SPECS[1],
    },
    "distribution_types": ["Col1: normal(0.0, 1.0)", "Col2: normal(0.0, 1.0)"],
}


@pytest.fixture(scope="module")
def dt_smoke_output(tmp_path_factory):
    """Run a minimal DT simulation and return the output DataFrame."""
    out_dir = tmp_path_factory.mktemp("dt_smoke")
    run_simulation(
        _DT_CFG,
        output_dir=str(out_dir),
        r=2,
        n_test=200,
        sample_sizes=[100],
    )
    csvs = list(out_dir.glob("boot_results_dt_smoke_1_*.csv"))
    assert len(csvs) == 1, f"expected 1 output CSV, got {csvs}"
    return pd.read_csv(csvs[0])


# Output structure

def test_output_has_expected_columns(dt_smoke_output):
    df = dt_smoke_output
    for col in ("simulation_id", "n_samples", "tree_idx", "tree_text", "bias2", "var", "mse"):
        assert col in df.columns, f"missing column '{col}'"


def test_output_row_count_matches_tree_count(dt_smoke_output):
    """C=3 has exactly 3 ND trees."""
    n_trees = len(all_trees(3))
    assert len(dt_smoke_output) == n_trees


def test_output_bias2_non_negative(dt_smoke_output):
    assert (dt_smoke_output["bias2"] >= 0).all()


def test_output_var_non_negative(dt_smoke_output):
    assert (dt_smoke_output["var"] >= 0).all()


def test_output_mse_equals_bias2_plus_var(dt_smoke_output):
    df = dt_smoke_output
    np.testing.assert_allclose(df["mse"], df["bias2"] + df["var"], atol=1e-9)


def test_output_values_finite(dt_smoke_output):
    for col in ("bias2", "var", "mse"):
        assert np.isfinite(dt_smoke_output[col]).all(), f"{col} has non-finite values"


def test_output_tree_text_parseable(dt_smoke_output):
    """tree_text must be parseable as a tuple of ND splits."""
    for text in dt_smoke_output["tree_text"]:
        tree = ast.literal_eval(text)
        assert isinstance(tree, tuple)
        assert len(tree) == 2  # C-1 = 3-1 = 2 splits


def test_output_simulation_id_matches(dt_smoke_output):
    assert (dt_smoke_output["simulation_id"] == "dt_smoke_1").all()


# csv_to_configs round-trip: CSV row → config dict → run_simulation

def test_dt_csv_row_parses_and_runs(tmp_path):
    """Write a minimal DT CSV row, parse it, and verify run_simulation accepts it."""
    generated_nd = str([((0, 1), (2,)), ((0,), (1,))])
    node_spec    = str(_DT_SPECS)

    row = {
        "simulation_id":    "dt_csv_round_trip",
        "experiment":       "3_class_2_features_dense_sparsity",
        "seed":             "99",
        "n_classes":        "3",
        "n_features":       "2",
        "generator_type":   "decisiontree",
        "generated_nd":     generated_nd,
        "node_spec":        node_spec,
        "distribution_types": "['Col1: normal(0.0, 1.0)', 'Col2: normal(0.0, 1.0)']",
    }

    cfg = _parse_row(row)
    assert cfg["generator_type"] == "decisiontree"

    out_dir = tmp_path / "csv_run"
    out_dir.mkdir()
    run_simulation(cfg, output_dir=str(out_dir), r=2, n_test=100, sample_sizes=[50])

    csvs = list(out_dir.glob("boot_results_dt_csv_round_trip_*.csv"))
    assert len(csvs) == 1
    df = pd.read_csv(csvs[0])
    assert len(df) == 3     # 3 ND trees for C=3
    assert (df["bias2"] >= 0).all()
