"""Print LaTeX real-data tables (spatial stats, heuristic placement). Run with --table 5."""
import argparse
import ast
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import load
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# Compatibility shim: nd/ uses np.int / np.float removed in numpy 1.24
np.int   = int    # noqa: E305
np.float = float  # noqa: E305

import config                                        # noqa: E402
import core as mh                                    # noqa: E402
from core import _cache_path, spatial_stats          # noqa: E402
from nd import ACND, BBoK, BestOfK, CBND, NDC, RPND, RandomGeneration  # noqa: E402

REAL_DIR  = ROOT / "real-data-bias-analysis"
CACHE_DIR = ROOT / "zenodo"
PLOT_DIR = REAL_DIR
DATASETS = [
    ("glass_identification",                           "Glass Identification", 6,  945),
    ("steel_plates_faults",                            "Steel Plates Faults",  7,  10_395),
    ("mice_protein",                                   "Mice Protein",         8,  15_000),
    ("urban_land_cover",                               "Urban Land Cover",     9,  15_000),
    ("pen_based_recognition_of_handwritten_digits_81", "Pen-Based Digits",     10, 15_000),
    ("soybean_large_122",                              "Soybean",              19, 15_000),
]

def load_dataset_results(slug, C, N, model="lr"):
    xlsx = REAL_DIR / slug / f"v_{slug}_{model}_cluster_tables.xlsx"
    csv  = ROOT / "zenodo" / "tree_metrics" / f"tree_metrics_{slug}.csv"
    if xlsx.exists():
        df = pd.read_excel(xlsx, sheet_name="tree_metrics")
    elif csv.exists():
        df = pd.read_csv(csv)
    else:
        return None, None, None
    art = load(_cache_path(C, N, seed=0, dim=2, cache_dir=CACHE_DIR))
    coords = np.asarray(art["coords_plot"])
    tw = art.get("trustworthiness")
    if tw is None:
        tw = art["metadata"]["trustworthiness"]

    trees_plot = art.get("trees_plot", art.get("trees"))
    if trees_plot is None:
        fp = _cache_path(C, N, seed=0, dim=2, cache_dir=CACHE_DIR)
        raise KeyError(f"{fp} has no trees_plot or trees field")

    tree_to_idx = {str(t): i for i, t in enumerate(trees_plot)}
    order = [tree_to_idx[t] for t in df["tree"].astype(str)]
    coords = coords[order]
    return df, coords, tw


def table5():
    print(r"\begin{table}[t]\centering")
    print(r"\caption{Spatial statistics across the six datasets. Under no spatial organisation,"
          r" each Moran's $I$ has expected value near zero. Trust@20 is the trustworthiness"
          r" of the 2D embedding at 20 nearest neighbours. All Moran's $I$ have $p \le 0.001$.}")
    print(r"\label{tab:realdata-spatial}\small\renewcommand{\arraystretch}{1.28}")
    print(r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lccccc@{}}")
    print(r"\toprule")
    print(r"Dataset & Classes & Trust@20 & $I$ (accuracy) & $I$ (variance) & $I$ (log loss) \\")
    print(r"\midrule")

    for slug, display, C, N in DATASETS:
        result = load_dataset_results(slug, C, N)
        if result[0] is None:
            print(f"% {display}: xlsx not found — run step 2 first", file=sys.stderr)
            print(f"{display} & {C} & -- & -- & -- & -- \\\\")
            continue
        df, coords, tw = result
        I_acc, _ = spatial_stats(df["score"].to_numpy(),     coords)
        I_var, _ = spatial_stats(df["model_var"].to_numpy(), coords)
        I_ll,  _ = spatial_stats(df["logloss"].to_numpy(),   coords)
        print(f"{display} & {C} & {tw:.3f} & {I_acc:.3f} & {I_var:.3f} & {I_ll:.3f} \\\\")

    print(r"\bottomrule")
    print(r"\end{tabular*}")
    print(r"\end{table}")

def table_heuristics():
    rows = []

    for slug, display, C, N in DATASETS:
        info_fp = PLOT_DIR / slug / "plots" / "nd_info.json"
        xlsx = REAL_DIR / slug / f"v_{slug}_lr_cluster_tables.xlsx"
        csv = ROOT / "zenodo" / "tree_metrics" / f"tree_metrics_{slug}.csv"
        raw_fp = ROOT / "zenodo" / "data" / f"{slug}.csv"

        if not info_fp.exists():
            raise FileNotFoundError(info_fp)
        if not raw_fp.exists():
            raise FileNotFoundError(raw_fp)

        info = pd.DataFrame(json.loads(info_fp.read_text()))
        info_idx = info.set_index("tree_id")

        if xlsx.exists():
            tree_df = pd.read_excel(xlsx, sheet_name="tree_metrics")
        elif csv.exists():
            tree_df = pd.read_csv(csv)
        else:
            raise FileNotFoundError(xlsx)
        trees = [ast.literal_eval(t) if isinstance(t, str) else t for t in tree_df["tree"]]

        raw = pd.read_csv(raw_fp)
        if "split" in raw.columns:
            tr = raw.index[raw["split"] == "train"].to_numpy()
            te = raw.index[raw["split"] == "test"].to_numpy()
            raw = raw.drop(columns="split")
        else:
            y0 = raw["y"].astype(int).to_numpy()
            tr, te = train_test_split(
                np.arange(len(y0)), test_size=0.30, stratify=y0, random_state=0
            )

        X = raw.drop(columns="y").to_numpy(float)
        y = raw["y"].astype(int).to_numpy()
        cats = tuple(map(int, np.unique(y)))
        cat_arr = np.asarray(cats, dtype=int)

        Xtr, Xte = X[tr].copy(), X[te].copy()
        ytr, yte = y[tr], y[te]

        sc = StandardScaler().fit(Xtr)
        Xtr = sc.transform(Xtr)
        Xte = sc.transform(Xte)

        base = mh.get_model("lr")
        bits = {
            "RPND": RPND.generate(Xtr, ytr, lambda **kw: clone(base), seed=0),
            "BBoK": BestOfK.generate(BBoK.generate, C, Xtr, ytr, base, labels=list(cats)),
            "BoK":  BestOfK.generate(RandomGeneration.generate, C, Xtr, ytr, base, labels=list(cats)),
            "ACND": ACND.generate(Xtr, ytr),
            "CBND": CBND.generate(Xtr, ytr, seed=0),
            "NDC":  NDC.generate(Xtr, ytr),
        }

        def decode(bit_nd):
            def mask(m):
                return tuple(int(c) for c in cats if int(m) & (1 << int(c)))
            return tuple((mask(L), mask(R)) for L, R in bit_nd if int(R) != 0)

        def canon(t):
            t = ast.literal_eval(t) if isinstance(t, str) else t
            return frozenset(tuple(sorted((tuple(sorted(L)), tuple(sorted(R))))) for L, R in t)

        lookup = {canon(t): i for i, t in enumerate(trees)}

        vec = TfidfVectorizer(analyzer=mh.split_analyzer(C), use_idf=False, norm="l2")
        X_tree = vec.fit_transform([str(t) for t in trees])

        config.X, config.y, config.X_test, config.y_test = Xtr, ytr, Xte, yte
        config.model_cache = {}

        method_rows = []
        for name, bit_t in bits.items():
            t = decode(bit_t)
            k = canon(t)

            if k in lookup:
                tree_id = lookup[k]
            else:
                x_new = vec.transform([str(t)])
                tree_id = int(cosine_similarity(x_new, X_tree)[0].argmax())

            P, cls = mh.nd_predict_proba(Xte, t, cats, base="lr")
            cls = tuple(map(int, cls))
            if cls != cats:
                col = {c: i for i, c in enumerate(cls)}
                P = P[:, [col[c] for c in cats]]

            acc = float((cat_arr[P.argmax(1)] == yte).mean())

            method_rows.append({
                "method": name,
                "tree_id": tree_id,
                "accuracy": acc,
                "cluster": int(info_idx.loc[tree_id, "cluster"]),
            })

        methods = pd.DataFrame(method_rows)
        means = 100 * info.groupby("cluster")["accuracy"].mean()
        ranks = {c: i + 1 for i, c in enumerate(means.sort_values(ascending=False).index)}

        methods["_acc"] = 100 * methods["accuracy"]
        best = methods.loc[methods["_acc"].idxmax()]
        best_hit = int(best["cluster"])
        rows.append([
            display,
            C,
            methods.loc[methods["_acc"].idxmax(), "method"],
            f"{ranks[int(best_hit)]} of {len(means)}",
            means.max() - means.loc[best_hit],
        ])

    print(r"\begin{table}[t]\centering")
    print(r"\caption{Embedding placement of construction heuristics on the real-data maps."
          r" The best heuristic is the highest-accuracy tree selected by any of the six"
            r" heuristics. The best region reached is the embedding cluster containing"
            r" that best heuristic tree, with rank 1 the highest-mean cluster. Region gap is"
            r" the mean accuracy of the best cluster minus the mean accuracy of that"
            r" heuristic's cluster.}")
    print(r"\label{tab:realdata-heuristic}\small\renewcommand{\arraystretch}{1.25}")
    print(r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lclcc@{}}")
    print(r"\toprule")
    print(r"Dataset & $M$ & Best heuristic & Best region reached & Region gap (pp) \\")
    print(r"\midrule")

    for r in rows:
        print(f"{r[0]} & {r[1]} & {r[2]} & {r[3]} & {r[4]:.1f} \\\\")

    print(r"\bottomrule")
    print(r"\end{tabular*}")
    print(r"\end{table}")

def main():
    parser = argparse.ArgumentParser(description="Print LaTeX tables from real-dataset results")
    parser.add_argument("--table", default="5", choices=["5", "heuristic"])
    parser.add_argument("--plot-dir", default=None,
                        help="Directory containing <dataset>/plots/nd_info.json files "
                             "(default: real-data-bias-analysis/)")
    args = parser.parse_args()
    if args.table == "5":
        table5()
    else:
        global PLOT_DIR
        if args.plot_dir is not None:
            PLOT_DIR = Path(args.plot_dir)
        table_heuristics()


if __name__ == "__main__":
    main()
