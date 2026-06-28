"""Unit tests for decision-tree generator kernels in run_simulations.py.

Covers:
- build_node_arrays_dt: correct shapes and child-pointer encoding
- compute_true_probabilities_numba_dt: probabilities sum to 1, non-negative, finite
- generate_labels_numba_dt: labels are valid class indices
- csv_to_configs._parse_row with generator_type='decisiontree'
- csv_to_configs._build_nd_params_dt: round-trips split/spec pairing
"""

import numpy as np
import pytest
from run_simulations import (
    build_node_arrays_dt,
    compute_true_probabilities_numba_dt,
    generate_labels_numba_dt,
    _parse_nd_params,
)
from scripts.csv_to_configs import _build_nd_params_dt, _parse_row

# Minimal 3-class DT fixture
# Generating tree: (((0,1),(2,)), ((0,),(1,)))
#   split 0: ((0,1),(2,))  — root, all 3 classes
#   split 1: ((0,),(1,))   — left subtree, 2 classes
# Each split uses a depth-1 decision tree:
#   depth=1, feat=[0], thr=[0.0]
#   leafp=[p_left, p_right]  — P(going right in the ND split)

ND_PARAMS_3C = {
    ((0, 1), (2,)): {"depth": 1, "feat": [0], "thr": [0.0], "leafp": [0.2, 0.8]},
    ((0,),   (1,)): {"depth": 1, "feat": [0], "thr": [0.0], "leafp": [0.3, 0.7]},
}
N_FEATURES = 2
N_CLASSES  = 3


@pytest.fixture(scope="module")
def dt_arrays():
    return build_node_arrays_dt(ND_PARAMS_3C, N_FEATURES)


# build_node_arrays_dt

def test_build_node_arrays_dt_sizes(dt_arrays):
    dt_depth, dt_feat, dt_thr, dt_leafp, left_child, right_child, root_idx = dt_arrays
    n_nodes = len(ND_PARAMS_3C)
    assert dt_depth.shape  == (n_nodes,)
    assert left_child.shape  == (n_nodes,)
    assert right_child.shape == (n_nodes,)


def test_build_node_arrays_dt_root_is_largest(dt_arrays):
    dt_depth, dt_feat, dt_thr, dt_leafp, left_child, right_child, root_idx = dt_arrays
    # Root covers all 3 classes; other node covers 2
    nodes = list(ND_PARAMS_3C.keys())
    root_split = nodes[root_idx]
    assert len(root_split[0]) + len(root_split[1]) == N_CLASSES


def test_build_node_arrays_dt_leaf_encoding(dt_arrays):
    """Leaf children should be encoded as -(class_index + 1)."""
    dt_depth, dt_feat, dt_thr, dt_leafp, left_child, right_child, root_idx = dt_arrays
    # The right child of the root split ((0,1),(2,)) is the leaf for class 2
    # encoded as -(2+1) = -3
    assert right_child[root_idx] == -3


def test_build_node_arrays_dt_internal_child_is_non_negative(dt_arrays):
    """The left child of the root should point to the internal node for split ((0,),(1,))."""
    dt_depth, dt_feat, dt_thr, dt_leafp, left_child, right_child, root_idx = dt_arrays
    assert left_child[root_idx] >= 0


def test_build_node_arrays_dt_depth_values(dt_arrays):
    dt_depth, *_ = dt_arrays
    assert (dt_depth == 1).all(), "all splits use depth=1 in the fixture"


# compute_true_probabilities_numba_dt

def test_dt_proba_shape(dt_arrays):
    dt_depth, dt_feat, dt_thr, dt_leafp, left_child, right_child, root_idx = dt_arrays
    rng = np.random.default_rng(0)
    X = rng.standard_normal((50, N_FEATURES))
    P = compute_true_probabilities_numba_dt(
        X, dt_depth, dt_feat, dt_thr, dt_leafp,
        left_child, right_child, root_idx, N_CLASSES,
    )
    assert P.shape == (50, N_CLASSES)


def test_dt_proba_sums_to_one(dt_arrays):
    dt_depth, dt_feat, dt_thr, dt_leafp, left_child, right_child, root_idx = dt_arrays
    rng = np.random.default_rng(1)
    X = rng.standard_normal((200, N_FEATURES))
    P = compute_true_probabilities_numba_dt(
        X, dt_depth, dt_feat, dt_thr, dt_leafp,
        left_child, right_child, root_idx, N_CLASSES,
    )
    np.testing.assert_allclose(P.sum(axis=1), 1.0, atol=1e-9,
                               err_msg="DT probabilities must sum to 1")


def test_dt_proba_non_negative(dt_arrays):
    dt_depth, dt_feat, dt_thr, dt_leafp, left_child, right_child, root_idx = dt_arrays
    rng = np.random.default_rng(2)
    X = rng.standard_normal((200, N_FEATURES))
    P = compute_true_probabilities_numba_dt(
        X, dt_depth, dt_feat, dt_thr, dt_leafp,
        left_child, right_child, root_idx, N_CLASSES,
    )
    assert (P >= 0).all()
    assert np.isfinite(P).all()


def test_dt_proba_class2_from_root(dt_arrays):
    """Samples with X[:,0] > 0 should route through the right leaf of the root (class 2)."""
    dt_depth, dt_feat, dt_thr, dt_leafp, left_child, right_child, root_idx = dt_arrays
    # X[:,0] >> 0: DT at root sees feat[0]=0, thr=0.0 → X>thr → k=2 → leafp[1]=0.8
    # So p_right=0.8 → 80% weight to class 2
    X = np.ones((10, N_FEATURES)) * 5.0
    P = compute_true_probabilities_numba_dt(
        X, dt_depth, dt_feat, dt_thr, dt_leafp,
        left_child, right_child, root_idx, N_CLASSES,
    )
    # Most weight should be on class 2 (index 2)
    assert P[:, 2].mean() > 0.5


# generate_labels_numba_dt

def test_dt_labels_valid_classes(dt_arrays):
    dt_depth, dt_feat, dt_thr, dt_leafp, left_child, right_child, root_idx = dt_arrays
    rng = np.random.default_rng(3)
    X = rng.standard_normal((500, N_FEATURES)).astype(np.float64)
    labels = generate_labels_numba_dt(
        X, seed=42,
        dt_depth=dt_depth, dt_feat=dt_feat, dt_thr=dt_thr, dt_leafp=dt_leafp,
        left_child=left_child, right_child=right_child, root_idx=root_idx,
    )
    assert labels.shape == (500,)
    assert set(labels).issubset({0, 1, 2}), f"unexpected labels: {set(labels)}"


def test_dt_labels_all_classes_present(dt_arrays):
    """With enough samples and balanced probabilities all 3 classes should appear."""
    dt_depth, dt_feat, dt_thr, dt_leafp, left_child, right_child, root_idx = dt_arrays
    rng = np.random.default_rng(4)
    X = rng.standard_normal((2000, N_FEATURES)).astype(np.float64)
    labels = generate_labels_numba_dt(
        X, seed=0,
        dt_depth=dt_depth, dt_feat=dt_feat, dt_thr=dt_thr, dt_leafp=dt_leafp,
        left_child=left_child, right_child=right_child, root_idx=root_idx,
    )
    assert len(set(labels)) == N_CLASSES, f"only got classes {set(labels)}"


def test_dt_labels_deterministic(dt_arrays):
    """Same seed must produce identical label arrays."""
    dt_depth, dt_feat, dt_thr, dt_leafp, left_child, right_child, root_idx = dt_arrays
    rng = np.random.default_rng(5)
    X = rng.standard_normal((300, N_FEATURES)).astype(np.float64)
    kw = dict(dt_depth=dt_depth, dt_feat=dt_feat, dt_thr=dt_thr, dt_leafp=dt_leafp,
              left_child=left_child, right_child=right_child, root_idx=root_idx)
    y1 = generate_labels_numba_dt(X, seed=77, **kw)
    y2 = generate_labels_numba_dt(X, seed=77, **kw)
    np.testing.assert_array_equal(y1, y2)


# csv_to_configs._build_nd_params_dt and _parse_row

def test_build_nd_params_dt_keys_are_strings():
    """_build_nd_params_dt must return string keys (as stored in JSON configs)."""
    generated_nd = str([((0, 1), (2,)), ((0,), (1,))])
    specs = [
        {"depth": 1, "feat": [0], "thr": [0.0], "leafp": [0.2, 0.8]},
        {"depth": 1, "feat": [0], "thr": [0.0], "leafp": [0.3, 0.7]},
    ]
    result = _build_nd_params_dt(generated_nd, str(specs))
    assert all(isinstance(k, str) for k in result.keys())
    assert len(result) == 2


def test_build_nd_params_dt_values_roundtrip():
    """Values stored in nd_params must survive ast.literal_eval round-trip."""
    generated_nd = str([((0, 1), (2,)), ((0,), (1,))])
    specs = [
        {"depth": 1, "feat": [0], "thr": [0.0], "leafp": [0.2, 0.8]},
        {"depth": 1, "feat": [0], "thr": [0.0], "leafp": [0.3, 0.7]},
    ]
    raw = _build_nd_params_dt(generated_nd, str(specs))
    parsed = _parse_nd_params(raw)
    for key, val in parsed.items():
        assert isinstance(key, tuple)
        assert "depth" in val and "feat" in val and "thr" in val and "leafp" in val


def test_parse_row_decisiontree():
    """_parse_row with generator_type='decisiontree' must return a valid config dict."""
    specs = [
        {"depth": 1, "feat": [0], "thr": [0.0], "leafp": [0.2, 0.8]},
        {"depth": 1, "feat": [0], "thr": [0.0], "leafp": [0.3, 0.7]},
    ]
    row = {
        "simulation_id":    "dt_test_1",
        "seed":             "42",
        "n_classes":        "3",
        "n_features":       "2",
        "generator_type":   "decisiontree",
        "generated_nd":     str([((0, 1), (2,)), ((0,), (1,))]),
        "node_spec":        str(specs),
        "distribution_types": "['Col1: normal(0.0, 1.0)', 'Col2: normal(0.0, 1.0)']",
    }
    cfg = _parse_row(row)
    assert cfg["generator_type"] == "decisiontree"
    assert cfg["n_classes"] == 3
    assert len(cfg["nd_params"]) == 2
    # All nd_params keys should be strings (for JSON serialisation)
    assert all(isinstance(k, str) for k in cfg["nd_params"])


def test_parse_row_lr_unchanged():
    """_parse_row for LR (default) must still work with 'covariates' field."""
    row = {
        "simulation_id":    "lr_test_1",
        "seed":             "7",
        "n_classes":        "3",
        "n_features":       "2",
        "generated_nd":     str([((0, 1), (2,)), ((0,), (1,))]),
        "covariates":       str([[1.0, 0.5, -0.3], [0.2, -0.1, 0.8]]),
        "distribution_types": "['Col1: normal(0.0, 1.0)', 'Col2: normal(0.0, 1.0)']",
        "experiment":       "3_class_2_features_test",
    }
    cfg = _parse_row(row)
    assert cfg["generator_type"] == "lr"
    assert len(cfg["nd_params"]) == 2
