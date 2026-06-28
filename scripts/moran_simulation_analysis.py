"""Compute Moran's I / Geary's C / bivariate Moran on simulation bias-variance results.

Aligns each boot_results CSV to the artifact tree order and writes one summary
CSV per experiment. Invoked by reproduce.py step 4; see README.
"""

import argparse
import ast
import glob
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from esda import Geary, Moran
from esda.moran import Moran_BV
from joblib import load
from scipy.spatial import cKDTree
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from core import build_knn_weights  # noqa: E402

N_BY_CLASS = {6: 945, 7: 10395, 8: 15000, 9: 15000, 10: 15000, 19: 15000}
DEFAULT_EXPERIMENTS = [
    "6_class_10_features_dense_sparsity",
    "6_class_1000_features_dense_sparsity",
    "6_class_1000_features_sparse_sparsity",
    "7_class_10_features_dense_sparsity",
    "7_class_1000_features_dense_sparsity",
    "7_class_1000_features_sparse_sparsity",
    "8_class_1000_features_sparse_sparsity",
    "9_class_1000_features_sparse_sparsity",
    "10_class_1000_features_sparse_sparsity",
]

warnings.filterwarnings("ignore")


def canonical_tree(tree):
    return frozenset(tuple(sorted((tuple(sorted(a)), tuple(sorted(b))))) for a, b in tree)


def mean_leaf_depth(tree_text):
    splits = ast.literal_eval(tree_text) if isinstance(tree_text, str) else tree_text
    i, depths = 0, []

    def walk(classes, depth):
        nonlocal i
        if len(classes) == 1:
            depths.append(depth)
            return
        left, right = splits[i]
        i += 1
        walk(left, depth + 1)
        walk(right, depth + 1)

    root = tuple(sorted(set(splits[0][0]) | set(splits[0][1])))
    walk(root, 0)
    return float(np.mean(depths))


def load_artifact(n_classes, cache_dir, prefix="art_1"):
    fp = Path(cache_dir) / f"{prefix}_C{n_classes}_N{N_BY_CLASS[n_classes]}_s0_d2.joblib"
    if not fp.exists():
        raise FileNotFoundError(f"Missing artifact: {fp}")
    art = load(fp)
    coords = np.asarray(art.get("coords_plot", art.get("coords")), float)
    trees = list(map(str, art.get("trees_plot", art["trees"])))
    return coords, trees


def find_result_file(sim_id, result_dirs, reps):
    for d in result_dirs:
        matches = sorted(glob.glob(str(Path(d) / f"boot_results_{sim_id}_{reps}_runs*.csv")))
        if matches:
            return matches[0]
        matches = sorted(glob.glob(str(Path(d) / f"v2_boot_results_{sim_id}_{reps}_runs*.csv")))
        if matches:
            return matches[0]
    return None


def add_spatial(row, name, y, depth, w, permutations):
    mi = Moran(y, w, permutations=permutations)
    gc = Geary(y, w, permutations=permutations)
    mb = Moran_BV(y, depth, w, permutations=permutations)

    row[f"{name}_moran_I"] = float(mi.I)
    row[f"{name}_moran_EI"] = float(mi.EI)
    row[f"{name}_moran_p_sim"] = float(mi.p_sim)
    row[f"{name}_geary_C"] = float(gc.C)
    row[f"{name}_geary_EC"] = float(gc.EC)
    row[f"{name}_geary_p_sim"] = float(gc.p_sim)
    row[f"{name}_depth_bv_moran_I"] = float(mb.I)
    row[f"{name}_depth_bv_moran_p_sim"] = float(mb.p_sim)

    depth_r = spearmanr(y, depth)
    row[f"{name}_depth_spearman_rho"] = float(depth_r.statistic)
    row[f"{name}_depth_spearman_p"] = float(depth_r.pvalue)
    depth_p = pearsonr(y, depth)
    row[f"{name}_depth_pearson_r"] = float(depth_p.statistic)
    row[f"{name}_depth_pearson_p"] = float(depth_p.pvalue)


def add_proximity(row, name, y, coords, gen_idx):
    dist = np.linalg.norm(coords - coords[gen_idx], axis=1)
    dist[gen_idx] = np.inf
    order = np.argsort(dist)

    dist_r = spearmanr(dist[np.isfinite(dist)], y[np.isfinite(dist)])
    row[f"{name}_distance_to_generating_spearman_rho"] = float(dist_r.statistic)
    row[f"{name}_distance_to_generating_spearman_p"] = float(dist_r.pvalue)

    for pct in [5, 10, 20]:
        k = max(1, int(len(y) * pct / 100))
        near = order[:k]
        good = y <= np.percentile(y, pct)
        n_good = good.sum()
        row[f"{name}_top{pct}_recall_nearest"] = float(good[near].sum() / n_good) if n_good else float("nan")
        try:
            row[f"{name}_top{pct}_auc_by_nearness"] = float(roc_auc_score(good, -dist))
        except ValueError:
            # "good" is all-True or all-False at this percentile — AUC is undefined.
            row[f"{name}_top{pct}_auc_by_nearness"] = float("nan")
        try:
            row[f"{name}_top{pct}_average_precision_by_nearness"] = float(average_precision_score(good, -dist))
        except ValueError:
            row[f"{name}_top{pct}_average_precision_by_nearness"] = float("nan")
        row[f"{name}_top{pct}_enrichment_nearest"] = float(good[near].mean() / good.mean()) if good.mean() else float("nan")
        row[f"{name}_top{pct}_median_distance_good"] = float(np.median(dist[good]))
        row[f"{name}_top{pct}_median_distance_all"] = float(np.median(dist[np.isfinite(dist)]))

    row[f"{name}_best_tree_distance_to_generating"] = float(dist[np.argmin(y)])
    row[f"{name}_best_tree_distance_percentile"] = float(
        (dist < dist[np.argmin(y)]).mean() * 100
    )


def save_metric_plot(path, coords, values, gen_idx, title):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=values, s=8, linewidths=0)
    if gen_idx is not None:
        ax.scatter(coords[gen_idx, 0], coords[gen_idx, 1], s=90, facecolors="none", edgecolors="black", linewidths=1.5)
    ax.set_xlabel("MDS-1")
    ax.set_ylabel("MDS-2")
    ax.set_title(title)
    fig.colorbar(sc, ax=ax)
    fig.savefig(path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def analyse_simulation(sim_row, coords, trees, result_dirs, reps, k, permutations, plot_dir=None):
    sim_id = str(sim_row["simulation_id"])
    result_file = find_result_file(sim_id, result_dirs, reps)
    if result_file is None:
        return [{"experiment": sim_row["experiment"], "simulation_id": sim_id, "status": "missing_result_file"}]

    df = pd.read_csv(result_file)
    if df.empty:
        return [{"experiment": sim_row["experiment"], "simulation_id": sim_id, "status": "empty_result_file", "result_file": result_file}]

    target = canonical_tree(ast.literal_eval(sim_row["generated_nd"]))
    tree_keys = [canonical_tree(ast.literal_eval(t)) for t in trees]
    gen_idx = next((i for i, key in enumerate(tree_keys) if key == target), None)
    depth = np.array([mean_leaf_depth(t) for t in trees], float)
    w = build_knn_weights(coords, k=k)
    out = []

    for n_samples, dfn in df.groupby("n_samples"):
        dfn = dfn.copy()
        dfn["key"] = [canonical_tree(ast.literal_eval(t)) for t in dfn["tree_text"].astype(str)]

        vals = {}
        for col in ["bias2", "var", "mse"]:
            m = dict(zip(dfn["key"], dfn[col]))
            missing = [key for key in tree_keys if key not in m]
            if missing:
                raise ValueError(f"{sim_id}: {len(missing)} artifact trees missing from {result_file}")
            vals[col] = np.array([m[key] for key in tree_keys], float)

        row = {
            "experiment": sim_row["experiment"],
            "simulation_id": sim_id,
            "n_classes": int(sim_row["n_classes"]),
            "n_samples": int(n_samples),
            "status": "ok",
            "result_file": result_file,
            "k_used": int(min(k, len(coords) - 1)),
            "generating_tree_found": gen_idx is not None,
            "median_r": float(np.median(np.linalg.norm(coords, axis=1))),
        }

        nn_k = min(11, len(coords))
        nn = cKDTree(coords).query(coords, k=nn_k)[1][:, 1:]
        row["nn_mse_autocorr_k10"] = float(np.corrcoef(vals["mse"], vals["mse"][nn].mean(axis=1))[0, 1])

        for name, y in vals.items():
            add_spatial(row, name, y, depth, w, permutations)
            if gen_idx is not None:
                add_proximity(row, name, y, coords, gen_idx)

        # IQR and spread of each error component
        for m in ["bias2", "var", "mse"]:
            row[f"{m}_iqr"] = float(
                np.percentile(vals[m], 75) - np.percentile(vals[m], 25)
            )
        row["var_nd_sd"] = float(np.std(vals["var"], ddof=1))

        _bv_r = spearmanr(vals["bias2"], vals["var"])
        row["bias2_var_spearman_rho"] = float(_bv_r.statistic)
        row["bias2_var_spearman_p"]   = float(_bv_r.pvalue)

        _bv_p = pearsonr(vals["bias2"], vals["var"])
        row["bias2_var_pearson_r2"] = float(_bv_p.statistic ** 2)

        mb_vb = Moran_BV(vals["var"], vals["bias2"], w, permutations=permutations)
        row["var_bias2_bv_moran_I"]     = float(mb_vb.I)
        row["var_bias2_bv_moran_p_sim"] = float(mb_vb.p_sim)

        if plot_dir is not None:
            base = plot_dir / str(sim_row["experiment"]) / sim_id
            labels = {"bias2": "squared bias", "var": "variance", "mse": "squared probability loss"}
            for name, y in vals.items():
                save_metric_plot(base / f"{sim_id}_{name}_n{int(n_samples)}.pdf", coords, y, gen_idx, labels[name])
            save_metric_plot(base / f"{sim_id}_avg_depth_n{int(n_samples)}.pdf", coords, depth, gen_idx, "average leaf depth")

        out.append(row)

    return out


def main():
    p = argparse.ArgumentParser(description="Summarise simulation Moran/geary/proximity statistics")
    p.add_argument("--selected", default="zenodo/simulation_configs.csv")
    p.add_argument("--results", nargs="+", default=["zenodo/raw_simulation_results"])
    p.add_argument("--cache", default="cache_nd")
    p.add_argument("--out", default="zenodo/simulation_scores")
    p.add_argument("--plot-dir", default="zenodo/simulation_plots")
    p.add_argument("--plots", action="store_true")
    p.add_argument("--reps", type=int, default=100)
    p.add_argument("--k", type=int, default=50)
    p.add_argument("--permutations", type=int, default=999)
    p.add_argument("--experiments", nargs="+", default=DEFAULT_EXPERIMENTS)
    p.add_argument("--artifact-prefix", default="art_1")
    args = p.parse_args()

    selected = pd.read_csv(args.selected, dtype=str)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    plot_dir = Path(args.plot_dir) if args.plots else None
    cache = {}

    for experiment in args.experiments:
        df_exp = selected[selected["experiment"].eq(experiment)].copy()
        if df_exp.empty:
            print(f"[skip] no rows for {experiment}")
            continue

        rows = []
        print(f"\n=== {experiment}: {len(df_exp)} simulations ===")
        for _, sim_row in df_exp.iterrows():
            n_classes = int(sim_row["n_classes"])
            if n_classes not in cache:
                cache[n_classes] = load_artifact(n_classes, args.cache, args.artifact_prefix)
            coords, trees = cache[n_classes]
            try:
                rows.extend(analyse_simulation(sim_row, coords, trees, args.results, args.reps, args.k, args.permutations, plot_dir))
            except Exception as e:
                rows.append({
                    "experiment": experiment,
                    "simulation_id": str(sim_row["simulation_id"]),
                    "n_classes": n_classes,
                    "status": "error",
                    "error": repr(e),
                })
                print(f"ERROR {sim_row['simulation_id']}: {e!r}")

        out = Path(args.out) / f"{experiment}_simulation_scores.csv"
        pd.DataFrame(rows).to_csv(out, index=False)
        ok = sum(1 for r in rows if r.get("status") == "ok")
        print(f"  wrote {out} ({ok}/{len(rows)} ok)")


if __name__ == "__main__":
    main()
