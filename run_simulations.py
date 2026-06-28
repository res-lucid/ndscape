"""Bias-variance simulation for nested dichotomies.

Reads configs from a CSV (zenodo/simulation_configs.csv) or JSON file,
runs R bootstrap reps in memory, writes one CSV per simulation to the output dir.
Invoked by reproduce.py; see README for the full pipeline.
"""

import argparse
import ast
import json
import os
import sys
import warnings

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from numba import njit, prange
from numpy.random import Generator, PCG64, SeedSequence
from pandas.errors import PerformanceWarning

import core as mh
from scripts.csv_to_configs import iter_rows, _parse_row

warnings.filterwarnings("ignore", category=PerformanceWarning)

# Configuration

DEFAULT_OUTPUT = "bias-analysis/raw_simulation_results"
R = 100          # bootstrap repetitions
N_TEST = 20_000
MODEL_NAME = "lr"
SAMPLE_SIZES = [5000]
N_JOBS = 16      # parallel threads for split fitting

@njit(cache=True)
def sigmoid_numba(x):
    if x >= 0:
        z = np.exp(-x)
        return 1.0 / (1.0 + z)
    z = np.exp(x)
    return z / (1.0 + z)

@njit(parallel=True, cache=True)
def compute_true_probabilities_numba(X, coefs, left_child, right_child, root_idx, n_classes):
    n_samples, n_features = X.shape
    n_nodes = coefs.shape[0]
    P = np.zeros((n_samples, n_classes), dtype=np.float64)

    for i in prange(n_samples):
        node_stack = np.empty(n_nodes, dtype=np.int64)
        weight_stack = np.empty(n_nodes, dtype=np.float64)
        top = 0
        node_stack[0] = root_idx
        weight_stack[0] = 1.0

        while top >= 0:
            node = node_stack[top]
            weight = weight_stack[top]
            top -= 1

            z = coefs[node, 0]
            for j in range(n_features):
                z += coefs[node, j + 1] * X[i, j]

            p = sigmoid_numba(z)
            left_weight = weight * (1.0 - p)
            right_weight = weight * p

            left_node = left_child[node]
            right_node = right_child[node]

            if left_node >= 0:
                top += 1
                node_stack[top] = left_node
                weight_stack[top] = left_weight
            else:
                P[i, -left_node - 1] += left_weight

            if right_node >= 0:
                top += 1
                node_stack[top] = right_node
                weight_stack[top] = right_weight
            else:
                leaf_ix_fixed = -right_node - 1
                P[i, leaf_ix_fixed] += right_weight

    return P

@njit(cache=True)
def generate_labels_numba(X, seed, coefs, left_child, right_child, root_idx):
    np.random.seed(seed)
    n_samples, n_features = X.shape
    y = np.empty(n_samples, dtype=np.int64)

    for i in range(n_samples):
        node = root_idx
        while True:
            z = coefs[node, 0]
            for j in range(n_features):
                z += coefs[node, j + 1] * X[i, j]
            p = sigmoid_numba(z)
            if np.random.random() < p:
                node = right_child[node]
            else:
                node = left_child[node]
            if node < 0:
                y[i] = -node - 1
                break
    return y

@njit(parallel=True, cache=True)
def accumulate_trees(tree_split_idx, tree_split_n, split_preds,
                     split_left, split_right, split_left_n, split_right_n,
                     pred_sum, pred_sq_sum, n_test, n_classes):
    for ti in prange(tree_split_idx.shape[0]):
        P = np.ones((n_test, n_classes))
        for ki in range(tree_split_n[ti]):
            s = tree_split_idx[ti, ki]
            for i in range(n_test):
                p = split_preds[s, i]
                q = 1.0 - p
                for a in range(split_left_n[s]):
                    P[i, split_left[s, a]] *= q
                for a in range(split_right_n[s]):
                    P[i, split_right[s, a]] *= p
        for i in range(n_test):
            rs = 0.0
            for c in range(n_classes):
                rs += P[i, c]
            inv = 1.0 / rs
            for c in range(n_classes):
                v = P[i, c] * inv
                pred_sum[ti, i, c] += v
                pred_sq_sum[ti, i, c] += v * v

# Helpers

def make_features(rg, n_samples, n_features, distribution_types):
    """Reconstruct feature matrix from pre-computed distribution specs."""
    X = np.empty((n_samples, n_features))
    for i, item in enumerate(distribution_types):
        _, distribution = item.split(": ", 1)
        dist_type, params_str = distribution.split("(", 1)
        params = ast.literal_eval(params_str[:-1])
        if dist_type == "normal":
            X[:, i] = rg.normal(loc=params[0], scale=params[1], size=n_samples)
        elif dist_type == "poisson":
            X[:, i] = rg.poisson(lam=params[0], size=n_samples)
        elif dist_type == "gamma":
            X[:, i] = rg.gamma(shape=params[0], scale=params[1], size=n_samples)
        elif dist_type == "laplace":
            X[:, i] = rg.laplace(loc=params[0], scale=params[1], size=n_samples)
        elif dist_type == "binomial":
            X[:, i] = rg.binomial(1, params[0], size=n_samples)
        else:
            raise ValueError(f"Unknown distribution type: {dist_type}")
    return X

def build_node_arrays(nd_params, n_features):
    nodes = list(nd_params.keys())
    node_lookup = {}
    for i, node in enumerate(nodes):
        key = tuple(sorted(node[0] + node[1]))
        node_lookup[key] = i

    root_idx = max(range(len(nodes)), key=lambda i: len(nodes[i][0]) + len(nodes[i][1]))

    coefs = np.empty((len(nodes), n_features + 1), dtype=np.float64)
    left_child = np.empty(len(nodes), dtype=np.int64)
    right_child = np.empty(len(nodes), dtype=np.int64)

    for i, node in enumerate(nodes):
        coefs[i] = np.asarray(nd_params[node], dtype=np.float64)
        left_key = tuple(sorted(node[0]))
        right_key = tuple(sorted(node[1]))
        left_child[i] = node_lookup.get(left_key, -(node[0][0] + 1))
        right_child[i] = node_lookup.get(right_key, -(node[1][0] + 1))

    return coefs, left_child, right_child, root_idx

# Decision-tree generator kernels

def build_node_arrays_dt(nd_params, n_features):
    """Pack DT-generator nd_params into arrays for numba kernels."""
    nodes = list(nd_params.keys())
    node_lookup = {tuple(sorted(node[0] + node[1])): i for i, node in enumerate(nodes)}
    root_idx = max(range(len(nodes)), key=lambda i: len(nodes[i][0]) + len(nodes[i][1]))

    max_depth = max(int(v["depth"]) for v in nd_params.values())
    max_int   = 2 ** max_depth - 1
    max_leaf  = 2 ** max_depth

    dt_depth    = np.empty(len(nodes), dtype=np.int64)
    dt_feat     = np.full((len(nodes), max_int),  -1,  dtype=np.int64)
    dt_thr      = np.zeros((len(nodes), max_int),      dtype=np.float64)
    dt_leafp    = np.zeros((len(nodes), max_leaf),     dtype=np.float64)
    left_child  = np.empty(len(nodes), dtype=np.int64)
    right_child = np.empty(len(nodes), dtype=np.int64)

    for i, node in enumerate(nodes):
        spec   = nd_params[node]
        depth  = int(spec["depth"])
        n_int  = 2 ** depth - 1
        n_leaf = 2 ** depth
        dt_depth[i]          = depth
        dt_feat[i,  :n_int]  = spec["feat"]
        dt_thr[i,   :n_int]  = spec["thr"]
        dt_leafp[i, :n_leaf] = spec["leafp"]
        left_key   = tuple(sorted(node[0]))
        right_key  = tuple(sorted(node[1]))
        left_child[i]  = node_lookup.get(left_key,  -(node[0][0] + 1))
        right_child[i] = node_lookup.get(right_key, -(node[1][0] + 1))

    return dt_depth, dt_feat, dt_thr, dt_leafp, left_child, right_child, root_idx

@njit(parallel=True, cache=True)
def compute_true_probabilities_numba_dt(X, dt_depth, dt_feat, dt_thr, dt_leafp,
                                        left_child, right_child, root_idx, n_classes):
    n_samples = X.shape[0]
    n_nodes   = dt_depth.shape[0]
    P = np.zeros((n_samples, n_classes), dtype=np.float64)

    for i in prange(n_samples):
        node_stack   = np.empty(n_nodes, dtype=np.int64)
        weight_stack = np.empty(n_nodes, dtype=np.float64)
        top = 0
        node_stack[0]   = root_idx
        weight_stack[0] = 1.0

        while top >= 0:
            node   = node_stack[top]
            weight = weight_stack[top]
            top   -= 1
            depth  = dt_depth[node]
            k = 0
            for _ in range(depth):
                if X[i, dt_feat[node, k]] > dt_thr[node, k]:
                    k = 2 * k + 2
                else:
                    k = 2 * k + 1
            p = dt_leafp[node, k - (2 ** depth - 1)]
            if left_child[node] >= 0:
                top += 1
                node_stack[top]   = left_child[node]
                weight_stack[top] = weight * (1.0 - p)
            else:
                P[i, -left_child[node] - 1] += weight * (1.0 - p)
            if right_child[node] >= 0:
                top += 1
                node_stack[top]   = right_child[node]
                weight_stack[top] = weight * p
            else:
                P[i, -right_child[node] - 1] += weight * p

    return P

@njit(cache=True)
def generate_labels_numba_dt(X, seed, dt_depth, dt_feat, dt_thr, dt_leafp,
                              left_child, right_child, root_idx):
    np.random.seed(seed)
    n_samples = X.shape[0]
    y = np.empty(n_samples, dtype=np.int64)

    for i in range(n_samples):
        node = root_idx
        while True:
            depth = dt_depth[node]
            k = 0
            for _ in range(depth):
                if X[i, dt_feat[node, k]] > dt_thr[node, k]:
                    k = 2 * k + 2
                else:
                    k = 2 * k + 1
            p = dt_leafp[node, k - (2 ** depth - 1)]
            if np.random.random() < p:
                node = right_child[node]
            else:
                node = left_child[node]
            if node < 0:
                y[i] = -node - 1
                break
    return y

def build_split_tree_metadata(unique_splits, unique_trees, n_classes):
    split_to_idx = {split: i for i, split in enumerate(unique_splits)}
    max_side = max(max(len(left), len(right)) for left, right in unique_splits)

    split_left = np.full((len(unique_splits), max_side), -1, dtype=np.int64)
    split_right = np.full((len(unique_splits), max_side), -1, dtype=np.int64)
    split_left_n = np.zeros(len(unique_splits), dtype=np.int64)
    split_right_n = np.zeros(len(unique_splits), dtype=np.int64)
    split_keep = np.zeros((len(unique_splits), n_classes), dtype=bool)
    split_right_mask = np.zeros((len(unique_splits), n_classes), dtype=bool)

    for i, (left, right) in enumerate(unique_splits):
        split_left_n[i] = len(left)
        split_right_n[i] = len(right)
        split_left[i, :len(left)] = left
        split_right[i, :len(right)] = right
        split_keep[i, list(left) + list(right)] = True
        split_right_mask[i, list(right)] = True

    max_splits_per_tree = max(len(T) for T in unique_trees)
    tree_split_idx = np.full((len(unique_trees), max_splits_per_tree), -1, dtype=np.int64)
    tree_split_n = np.zeros(len(unique_trees), dtype=np.int64)
    tree_texts = []

    for i, T in enumerate(unique_trees):
        idxs = [split_to_idx[s] for s in T]
        tree_split_idx[i, :len(idxs)] = idxs
        tree_split_n[i] = len(idxs)
        tree_texts.append(repr(T))

    return (split_left, split_right, split_left_n, split_right_n,
            split_keep, split_right_mask, tree_split_idx, tree_split_n, tree_texts)

def _parse_nd_params(raw):
    return {ast.literal_eval(k): v for k, v in raw.items()}

# Main simulation loop

def run_simulation(cfg, output_dir, r=R, n_test=N_TEST, model_name=MODEL_NAME, sample_sizes=None):
    if sample_sizes is None:
        sample_sizes = SAMPLE_SIZES

    simulation_id = str(cfg["simulation_id"])
    base_seed = int(cfg["seed"])
    n_classes = int(cfg["n_classes"])
    n_features = int(cfg["n_features"])
    nd_params = _parse_nd_params(cfg["nd_params"])

    distribution_types = cfg.get("distribution_types")
    if not distribution_types:
        raise ValueError(f"Config {simulation_id} is missing distribution_types")

    print(f"Starting simulation {simulation_id} | C={n_classes} F={n_features}")

    gen_type = cfg.get("generator_type", "lr")
    if gen_type == "decisiontree":
        gen_arrays = build_node_arrays_dt(nd_params, n_features)
        model_name = "decisiontree"
    else:
        gen_arrays = build_node_arrays(nd_params, n_features)

    if n_classes > 7:
        art = mh.get_trees_and_artifact(range(n_classes), N=15_000, seed=0, cache_dir='cache_nd')
        unique_trees = art["trees"]
    else:
        unique_trees = mh.all_trees(n_classes)

    unique_splits = sorted({s for T in unique_trees for s in T}, key=str)
    (split_left, split_right, split_left_n, split_right_n,
     split_keep, split_right_mask, tree_split_idx, tree_split_n,
     tree_texts) = build_split_tree_metadata(unique_splits, unique_trees, n_classes)

    test_ss = SeedSequence([base_seed, int(simulation_id), n_test, 999999])
    rg_test = Generator(PCG64(test_ss))
    X_test = make_features(rg_test, n_test, n_features, distribution_types)
    if gen_type == "decisiontree":
        P_true = compute_true_probabilities_numba_dt(X_test, *gen_arrays, n_classes)
    else:
        P_true = compute_true_probabilities_numba(X_test, *gen_arrays, n_classes)

    split_pred_shape = (len(unique_splits), n_test)

    for n_samples in sample_sizes:
        out_file = os.path.join(output_dir, f"boot_results_{simulation_id}_{r}_runs_.csv")
        if os.path.exists(out_file):
            print(f"  {out_file} exists, skipping.")
            continue

        pred_sum    = np.zeros((len(unique_trees), n_test, n_classes), dtype=np.float64)
        pred_sq_sum = np.zeros((len(unique_trees), n_test, n_classes), dtype=np.float64)

        for rep in range(r):
            rep_ss = SeedSequence([base_seed, int(simulation_id), n_samples, rep])
            rg = Generator(PCG64(rep_ss))
            X_train = make_features(rg, n_samples, n_features, distribution_types)
            label_seed = int(SeedSequence([base_seed, int(simulation_id), n_samples, rep, 12345]).generate_state(1)[0])
            if gen_type == "decisiontree":
                y_train = generate_labels_numba_dt(X_train, label_seed, *gen_arrays)
            else:
                y_train = generate_labels_numba(X_train, label_seed, *gen_arrays)

            def fit_split(split_idx):
                keep_mask = split_keep[split_idx][y_train]
                y_node = split_right_mask[split_idx][y_train[keep_mask]].astype(np.uint8)
                if y_node.size == 0:
                    return split_idx, np.full(n_test, 0.5, dtype=np.float32)
                if y_node.min() == y_node.max():
                    return split_idx, np.full(n_test, float(y_node[0]), dtype=np.float32)
                model = mh.get_model(model_name)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    model.fit(X_train[keep_mask], y_node)
                return split_idx, model.predict_proba(X_test)[:, 1].astype(np.float32)

            split_preds = np.empty(split_pred_shape, dtype=np.float32)
            for split_idx, p in Parallel(n_jobs=N_JOBS, prefer="threads")(
                delayed(fit_split)(i) for i in range(len(unique_splits))
            ):
                split_preds[split_idx] = p

            accumulate_trees(tree_split_idx, tree_split_n, split_preds,
                             split_left, split_right, split_left_n, split_right_n,
                             pred_sum, pred_sq_sum, n_test, n_classes)

            if (rep + 1) % 10 == 0 or rep + 1 == r:
                print(f"  sim {simulation_id} n={n_samples} rep {rep + 1}/{r}")

        mean      = pred_sum / r
        bias2_vec = ((mean - P_true) ** 2).sum(axis=2).mean(axis=1)
        var_vec   = ((pred_sq_sum / r) - mean ** 2).sum(axis=2).mean(axis=1)

        T = len(unique_trees)
        df_bv = pd.DataFrame({
            "simulation_id":   [simulation_id] * T,
            "experiment_name": [cfg.get("experiment", "")] * T,
            "n_samples":       [n_samples] * T,
            "tree_idx":        np.arange(T),
            "tree_text":       tree_texts,
            "bias2":           bias2_vec,
            "var":             var_vec,
            "mse":             bias2_vec + var_vec,
        })
        df_bv.to_csv(out_file, index=False)
        print(f"  Wrote {out_file}")

def _iter_configs(input_path):
    if input_path.endswith(".csv"):
        for fields in iter_rows(input_path):
            try:
                yield _parse_row(fields)
            except Exception as e:
                print(f"  [warn] skipping row: {e}", file=sys.stderr)
    else:
        with open(input_path) as f:
            for cfg in json.load(f):
                yield cfg

def main():
    parser = argparse.ArgumentParser(description="Standalone ND bias-variance simulation")
    parser.add_argument("--input", default=None,
                        help="simulation_configs.csv or simulation_configs.json")
    parser.add_argument("--output", default=DEFAULT_OUTPUT,
                        help="Output directory for CSV files")
    parser.add_argument("--reps", type=int, default=R,
                        help="Bootstrap repetitions (default: 100)")
    args = parser.parse_args()

    input_path = args.input
    if input_path is None:
        for candidate in ["zenodo/simulation_configs.csv", "simulation_configs.json"]:
            if os.path.exists(candidate):
                input_path = candidate
                break
    if input_path is None:
        sys.exit(
            "No input found. Pass --input or download zenodo/simulation_configs.csv with:\n"
            "  python scripts/download_zenodo.py --what sims"
        )

    os.makedirs(args.output, exist_ok=True)
    for cfg in _iter_configs(input_path):
        run_simulation(cfg, output_dir=args.output, r=args.reps)

if __name__ == "__main__":
    main()
