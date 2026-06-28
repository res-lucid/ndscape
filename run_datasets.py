"""Score every ND tree on each paper dataset and save results to xlsx.

Usage
-----
    python run_datasets.py
    python run_datasets.py --datasets glass_identification steel_plates_faults
    python run_datasets.py --model lr --bootstrap 100
"""
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import config
import core as mh

TEST_SIZE = 0.30
BOOTSTRAP = 100
N_TREES   = 15_000
N_JOBS    = min(8, os.cpu_count() or 1)
OUT_DIR   = "real-data-bias-analysis"
CACHE_DIR = "cache_nd"

DATASETS = [
    "glass_identification",
    "steel_plates_faults",
    "mice_protein",
    "urban_land_cover",
    "pen_based_recognition_of_handwritten_digits_81",
    "soybean_large_122",
]


def load_dataset(name):
    p = Path(f"zenodo/data/{name}.csv")
    if not p.exists():
        p = Path(f"data/{name}.csv")
    df = pd.read_csv(p)
    X  = df.drop(columns=[c for c in ("y", "split") if c in df.columns]).to_numpy(float)
    y  = df["y"]
    if y.dtype == object:
        enc = {v: i for i, v in enumerate(sorted(y.unique()))}
        print(f"  label remap: {enc}")
        y = y.map(enc)
    y    = y.to_numpy(int)
    cats = tuple(map(int, np.unique(y)))
    if "split" in df.columns:
        tr = df.index[df["split"] == "train"].to_numpy()
        te = df.index[df["split"] == "test"].to_numpy()
    else:
        tr, te = train_test_split(
            np.arange(len(y)), test_size=TEST_SIZE, stratify=y, random_state=0
        )
    Xtr_raw, Xte_raw = X[tr], X[te]
    ytr, yte = y[tr], y[te]
    sc = StandardScaler().fit(Xtr_raw)
    return sc.transform(Xtr_raw), sc.transform(Xte_raw), ytr, yte, cats, Xtr_raw, Xte_raw


def _predict(tree, cats, Xte, model):
    P, _ = mh.nd_predict_proba(Xte, tree, cats, base=model)
    return np.asarray(P)


def score_trees(trees, cats, Xtr, ytr, Xte, yte, model):
    config.X, config.y = Xtr, ytr
    config.model_cache = {}

    cat_arr = np.array(cats)
    y_idx   = np.searchsorted(cat_arr, yte)
    n       = len(yte)

    def _one(ti):
        P    = _predict(trees[ti], cats, Xte, model)
        pred = cat_arr[P.argmax(1)]
        ok   = (pred == yte)
        nll  = -np.log(np.clip(P[np.arange(n), y_idx], 1e-15, 1.0)).sum()
        bacc = balanced_accuracy_score(yte, pred)
        mf1  = f1_score(yte, pred, average="macro", zero_division=0)
        return ok, nll, bacc, mf1

    rows   = Parallel(n_jobs=N_JOBS, prefer="threads")(delayed(_one)(i) for i in range(len(trees)))
    correct_mat = np.array([r[0] for r in rows])
    nll    = np.array([r[1] for r in rows])
    bacc   = np.array([r[2] for r in rows])
    f1     = np.array([r[3] for r in rows])
    score  = correct_mat.sum(1) / n
    return score, nll / n, correct_mat.astype(float).var(1, ddof=1), bacc, f1


def bootstrap_variance(trees, cats, Xtr_raw, ytr, Xte_raw, yte, model, B):
    T, n, C = len(trees), len(yte), len(cats)
    prob_sum = np.zeros((T, n, C), dtype=np.float64)
    prob_sq  = np.zeros((T, n, C), dtype=np.float64)
    rng   = np.random.default_rng(0)

    for b in range(B):
        idx = rng.integers(0, len(Xtr_raw), len(Xtr_raw))
        Xb, yb = Xtr_raw[idx], ytr[idx]
        sc = StandardScaler().fit(Xb)
        Xb_s, Xte_s = sc.transform(Xb), sc.transform(Xte_raw)
        config.X, config.y = Xb_s, yb
        config.model_cache = {}

        Ps    = Parallel(n_jobs=N_JOBS, prefer="threads")(
            delayed(_predict)(trees[ti], cats, Xte_s, model) for ti in range(T)
        )
        P_arr = np.array(Ps)
        prob_sum += P_arr
        prob_sq  += P_arr * P_arr
        if (b + 1) % 10 == 0:
            print(f"  bootstrap {b+1}/{B}")

    mean = prob_sum / B
    return ((prob_sq / B - mean ** 2) * (B / (B - 1))).mean(axis=(1, 2))


def run_dataset(name, model, B):
    print(f"\n=== {name} (model={model}) ===")
    Xtr, Xte, ytr, yte, cats, Xtr_raw, Xte_raw = load_dataset(name)
    art   = mh.get_trees_and_artifact(cats, N=N_TREES, seed=0, cache_dir=CACHE_DIR)
    trees = art["trees"]
    print(f"  trees={len(trees)}  C={len(cats)}  n_train={len(ytr)}  n_test={len(yte)}")

    score, logloss, var_01, bacc, f1 = score_trees(trees, cats, Xtr, ytr, Xte, yte, model)
    print(f"  accuracy  min={score.min():.4f}  max={score.max():.4f}")

    print(f"  computing bootstrap variance (B={B}) ...")
    model_var = bootstrap_variance(trees, cats, Xtr_raw, ytr, Xte_raw, yte, model, B)

    df_out = pd.DataFrame({
        "tree":            trees,
        "score":           score,
        "error_rate":      1 - score,
        "accuracy_var_01": var_01,
        "accuracy_se":     np.sqrt(var_01 / len(yte)),
        "logloss":         logloss,
        "loglik":          -logloss,
        "balanced_acc":    bacc,
        "macro_f1":        f1,
        "model_var":       model_var,
    })

    out_dir = os.path.join(OUT_DIR, name)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"v_{name}_{model}_cluster_tables.xlsx")
    with pd.ExcelWriter(path) as w:
        df_out.to_excel(w, sheet_name="tree_metrics", index=False)
    print(f"  saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="Score ND trees on real datasets")
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--model",     default="lr")
    parser.add_argument("--bootstrap", type=int, default=BOOTSTRAP)
    args = parser.parse_args()
    os.makedirs(OUT_DIR,   exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    for name in args.datasets:
        run_dataset(name, args.model, args.bootstrap)


if __name__ == "__main__":
    main()
