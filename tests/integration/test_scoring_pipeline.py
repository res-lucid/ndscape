"""Integration tests: full scoring pipeline.

Chains: data -> trees from xlsx -> score_trees -> compare against xlsx ground truth.
A failure means something in that chain changed the scores.

Runtime: long (hours for all datasets). Run with: pytest -m integration
"""
import ast
import hashlib
import json
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from sklearn.preprocessing import StandardScaler

from run_datasets import score_trees

FIXTURES = Path(__file__).parent / "fixtures"
ROOT     = Path(__file__).parent.parent.parent
REAL_DIR = ROOT / "real-data-bias-analysis"

pytestmark = pytest.mark.integration


def _load_xlsx(dataset):
    xlsx = REAL_DIR / dataset / f"v_{dataset}_lr_cluster_tables.xlsx"
    if not xlsx.exists():
        pytest.skip(f"{xlsx.name} not found")
    df    = pd.read_excel(xlsx, sheet_name="tree_metrics")
    trees = [ast.literal_eval(str(t)) for t in df["tree"]]
    return trees, df["score"].to_numpy()


def _run(data_fixture, dataset, atol=1e-6):
    Xtr_raw, ytr, Xte_raw, yte, cats = data_fixture
    sc  = StandardScaler().fit(Xtr_raw)
    Xtr, Xte = sc.transform(Xtr_raw), sc.transform(Xte_raw)
    trees, xlsx_scores = _load_xlsx(dataset)
    score, *_ = score_trees(trees, cats, Xtr, ytr, Xte, yte, "lr")
    np.testing.assert_allclose(score, xlsx_scores, atol=atol,
                               err_msg=f"{dataset}: scores deviate from xlsx ground truth")
    return score


def _snapshot(score, name):
    digest = hashlib.md5(score.astype("f8").tobytes()).hexdigest()
    fp     = FIXTURES / f"score_all_{name}.json"
    if not fp.exists():
        fp.write_text(json.dumps(digest))
        pytest.skip(f"fixture created: {fp.name} -- re-run to validate")
    assert digest == json.loads(fp.read_text())


def test_score_all_glass(glass):
    _snapshot(_run(glass, "glass_identification"), "glass_identification")

def test_score_all_steel_plates(steel_plates):
    _snapshot(_run(steel_plates, "steel_plates_faults"), "steel_plates_faults")

def test_score_all_mice(mice):
    _snapshot(_run(mice, "mice_protein"), "mice_protein")

def test_score_all_urban(urban):
    _snapshot(_run(urban, "urban_land_cover"), "urban_land_cover")

def test_score_all_pen_digits(pen_digits):
    _snapshot(_run(pen_digits, "pen_based_recognition_of_handwritten_digits_81"),
              "pen_based_recognition_of_handwritten_digits_81")

def test_score_all_soybean(soybean):
    _snapshot(_run(soybean, "soybean_large_122"), "soybean_large_122")
