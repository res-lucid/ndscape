from pathlib import Path

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

ROOT = Path(__file__).resolve().parents[2]
GOLDEN = ROOT / "tests" / "integration" / "fixtures" / "moran_scores"
SUMMARY = ROOT / "zenodo" / "simulation_scores"

GOLDEN_FILES = [
    "6_class_10_features_dense_sparsity_simulation_scores.csv",
    "6_class_1000_features_dense_sparsity_simulation_scores.csv",
    "6_class_1000_features_sparse_sparsity_simulation_scores.csv",
]


def _read_scores(path):
    assert path.exists(), f"Missing file: {path}"

    df = pd.read_csv(path, dtype={"simulation_id": str})

    # The statistics should be identical across machines, but paths may use
    # Windows or POSIX separators. Compare only the result filename.
    if "result_file" in df.columns:
        df["result_file"] = (
            df["result_file"].astype(str)
            .str.replace("\\", "/", regex=False)
            .str.rsplit("/", n=1).str[-1]
        )

    sort_cols = [c for c in ["experiment", "simulation_id", "n_samples"] if c in df.columns]
    df = df.sort_values(sort_cols).reset_index(drop=True)

    return df.reindex(sorted(df.columns), axis=1)


@pytest.mark.parametrize("fname", GOLDEN_FILES)
def test_moran_golden_file_is_complete(fname):
    df = _read_scores(GOLDEN / fname)

    assert len(df) == 50
    assert df["simulation_id"].nunique() == 50
    assert set(df["status"]) == {"ok"}


@pytest.mark.parametrize("fname", GOLDEN_FILES)
def test_moran_scores_match_golden_master(fname):
    actual_path = SUMMARY / fname

    if not actual_path.exists():
        pytest.skip(
            f"{actual_path} not found. Run: "
            "python reproduce.py --steps moran --classes 6"
        )

    expected = _read_scores(GOLDEN / fname)
    actual = _read_scores(actual_path)

    assert_frame_equal(
        actual,
        expected,
        check_dtype=False,
        rtol=1e-10,
        atol=1e-12,
    )