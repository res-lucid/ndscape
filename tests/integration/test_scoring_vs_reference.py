"""Integration test: compare run_datasets.py output against simulation-repo reference.

Ground truth is in tests/integration/fixtures/reference_<dataset>.parquet.
Missing reference parquet → test fails (fixture is committed, so this means it was deleted).
Missing ndscape xlsx → test skips (run run_datasets.py to generate it).
"""
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

pytestmark = pytest.mark.integration

FIXTURES = Path(__file__).parent / "fixtures"
ROOT     = Path(__file__).parent.parent.parent
REAL_DIR = ROOT / "real-data-bias-analysis"

DATASETS = [
    ("glass_identification",                           6,    945),
    ("steel_plates_faults",                            7,  10395),
    ("mice_protein",                                   8,  15000),
    ("urban_land_cover",                               9,  15000),
    ("pen_based_recognition_of_handwritten_digits_81", 10, 15000),
    ("soybean_large_122",                              19, 15000),
]


def _load_reference(dataset):
    fp = FIXTURES / f"reference_{dataset}.parquet"
    if not fp.exists():
        pytest.fail(
            f"Reference fixture {fp.name} not found in tests/integration/fixtures/. "
            "This file should be committed to the repo."
        )
    return pd.read_parquet(fp).sort_values("tree").reset_index(drop=True)


def _load_output(dataset):
    xlsx = REAL_DIR / dataset / f"v_{dataset}_lr_cluster_tables.xlsx"
    if not xlsx.exists():
        pytest.skip(
            f"{xlsx.name} not found. Run: python run_datasets.py --datasets {dataset}"
        )
    df = pd.read_excel(xlsx, sheet_name="tree_metrics")
    return df.sort_values("tree").reset_index(drop=True)


@pytest.mark.parametrize("dataset, C, N", DATASETS)
def test_tree_set_matches_reference(dataset, C, N):
    """Both outputs must contain exactly the same set of trees."""
    ref = _load_reference(dataset)
    out = _load_output(dataset)
    ref_trees = set(ref["tree"].astype(str))
    out_trees = set(out["tree"].astype(str))
    only_ref = ref_trees - out_trees
    only_out = out_trees - ref_trees
    assert not only_ref and not only_out, (
        f"{dataset}: tree sets differ. "
        f"Only in reference: {len(only_ref)}. Only in output: {len(only_out)}."
    )


@pytest.mark.parametrize("dataset, C, N", DATASETS)
def test_score_matches_reference(dataset, C, N):
    """Per-tree accuracy must match the simulation-repo reference exactly.
    Score is n_correct/n_test, an exact integer ratio."""
    ref = _load_reference(dataset)
    out = _load_output(dataset)
    np.testing.assert_array_equal(
        out["score"].to_numpy(),
        ref["score"].to_numpy(),
        err_msg=f"{dataset}: score values differ from simulation-repo reference",
    )


@pytest.mark.parametrize("dataset, C, N", DATASETS)
def test_logloss_matches_reference(dataset, C, N):
    """Per-tree log-loss must match the reference to 8 decimal places."""
    ref = _load_reference(dataset)
    out = _load_output(dataset)
    np.testing.assert_allclose(
        out["logloss"].to_numpy(),
        ref["logloss"].to_numpy(),
        atol=1e-8,
        err_msg=f"{dataset}: logloss values differ from simulation-repo reference",
    )


@pytest.mark.parametrize("dataset, C, N", DATASETS)
def test_model_var_matches_reference(dataset, C, N):
    """Bootstrap model-variance must match the reference within 0.1%.
    Small differences (~3e-6 relative) are expected from accumulated float ops."""
    ref = _load_reference(dataset)
    out = _load_output(dataset)
    np.testing.assert_allclose(
        out["model_var"].to_numpy(),
        ref["model_var"].to_numpy(),
        rtol=1e-3,
        err_msg=f"{dataset}: model_var values differ from simulation-repo reference",
    )
