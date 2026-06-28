"""Produce PDF scatter plots of the simulation embedding for specific figures.

Generates the bias², variance, and average-depth panels used in Figures 2–5
and the appendix. Output goes to paper_outputs/figures/simulations/.

Usage
-----
    python scripts/make_simulation_plots.py
    python scripts/make_simulation_plots.py --out paper_outputs/figures
    python scripts/make_simulation_plots.py --cache zenodo --results zenodo/raw_simulation_results
"""
import argparse
import ast
import glob
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
from joblib import load

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Figures to produce: (figure_num, panel_label, metric, simulation_id, n_classes)
PANELS = [
    # Figure 2 — squared bias, C=6
    (2, "topleft",     "bias2", "32993955013039740201065367455521256289798316034433680879034057974239280966362", 6),
    (2, "topright",    "bias2", "4809671270476855952210955267578072743732154719824877432737302482905348475547",  6),
    (2, "bottomleft",  "bias2", "21058846706421649102013786436120711240772727888240321459826791980331024892107", 6),
    (2, "bottomright", "bias2", "52854554759155356866124608620373503046910175073742541526702659068090243830056", 6),
    # Figure 3 — squared bias, C=7
    (3, "topleft",     "bias2", "46740025262664400170010037571718979152104697193920353418395985175978999166546", 7),
    (3, "topright",    "bias2", "53455843516418765408960174407304355908953721956747948764381365756568146175860", 7),
    (3, "bottomleft",  "bias2", "45932407632833261625521107446282430173320778743327868550375466303150522369760", 7),
    (3, "bottomright", "bias2", "12489175780025747152392377229814644268350214435231934413074034157813159663870", 7),
    # Figure 4 — variance + depth, C=6
    (4, "left",  "var",   "21058846706421649102013786436120711240772727888240321459826791980331024892107", 6),
    (4, "right", "depth", "21058846706421649102013786436120711240772727888240321459826791980331024892107", 6),
    # Figure 5 — variance + depth, C=7
    (5, "left",  "var",   "12489175780025747152392377229814644268350214435231934413074034157813159663870", 7),
    (5, "right", "depth", "12489175780025747152392377229814644268350214435231934413074034157813159663870", 7),
    # Appendix — C=9 bias², C=10 depth
    ("appendix", "C9",  "bias2", "110419469063603175368764080032581765619318785780519676756800016710819665785321", 9),
    ("appendix", "C10", "depth", "43632327680187798062217975243642513841299383369925962153426305454809148021682",  10),
]

N_BY_CLASS = {6: 945, 7: 10395, 9: 15000, 10: 15000}

METRIC_LABEL = {
    "bias2": "squared bias",
    "var":   "variance",
    "depth": "average leaf depth",
}

_BRIGHT = LinearSegmentedColormap.from_list(
    "bright_RYG", ["#ff2d2d", "#fff176", "#00e676"], N=256
)
CMAP = {
    "bias2": _BRIGHT.reversed(),
    "var":   _BRIGHT.reversed(),
    "depth": plt.cm.Purples_r,
}


def canonical_tree(tree):
    return frozenset(
        tuple(sorted((tuple(sorted(a)), tuple(sorted(b)))))
        for a, b in tree
    )


def mean_leaf_depth(tree_text):
    splits = ast.literal_eval(tree_text) if isinstance(tree_text, str) else tree_text
    idx = [0]
    depths = []

    def walk(classes, depth):
        if len(classes) == 1:
            depths.append(depth)
            return
        left, right = splits[idx[0]]
        idx[0] += 1
        walk(left, depth + 1)
        walk(right, depth + 1)

    root = tuple(sorted(set(splits[0][0]) | set(splits[0][1])))
    walk(root, 0)
    return float(np.mean(depths))


def load_artifact(n_classes, cache_dir):
    fp = Path(cache_dir) / f"art_1_C{n_classes}_N{N_BY_CLASS[n_classes]}_s0_d2.joblib"
    if not fp.exists():
        raise FileNotFoundError(f"Missing artifact: {fp}")
    art = load(fp)
    coords = np.asarray(art.get("coords_plot", art.get("coords")), float)
    trees = list(map(str, art.get("trees_plot", art["trees"])))
    return coords, trees


def load_values(sim_id, metric, trees, results_dir):
    """Load per-tree metric values aligned to artifact tree order."""
    if metric == "depth":
        return np.array([mean_leaf_depth(t) for t in trees], float)

    pattern = str(Path(results_dir) / f"boot_results_{sim_id}_100_runs*.csv")
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No result file for sim {sim_id[:20]}...")

    df = pd.read_csv(matches[0])
    df = df[df.n_samples == df.n_samples.max()]

    tree_to_val = dict(zip(df.tree_text.astype(str), df[metric]))
    return np.array([tree_to_val[t] for t in trees], float)


def find_generator(sim_id, trees, configs_csv):
    df = pd.read_csv(configs_csv, dtype=str)
    row = df[df.simulation_id == sim_id]
    if row.empty:
        return None
    nd = ast.literal_eval(row.iloc[0]["generated_nd"])
    target = canonical_tree(nd)
    for i, t in enumerate(trees):
        if canonical_tree(ast.literal_eval(t)) == target:
            return i
    return None


def make_plot(coords, values, gen_idx, metric, out_path):
    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    cmap = CMAP[metric]

    vmin, vmax = values.min(), values.max()
    sc = ax.scatter(
        coords[:, 0], coords[:, 1],
        c=values, cmap=cmap, vmin=vmin, vmax=vmax,
        s=30, linewidths=0, rasterized=True,
    )
    if gen_idx is not None:
        ax.scatter(
            coords[gen_idx, 0], coords[gen_idx, 1],
            marker="x", s=220, c="black", linewidths=2.2, zorder=12,
        )
    ax.set_xlabel("MDS-1")
    ax.set_ylabel("MDS-2")
    fig.colorbar(sc, ax=ax, label=METRIC_LABEL[metric])
    fig.tight_layout()
    fig.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


def main():
    p = argparse.ArgumentParser(description="Produce simulation figure panels")
    p.add_argument("--cache",   default="zenodo")
    p.add_argument("--results", default="zenodo/raw_simulation_results")
    p.add_argument("--configs", default="zenodo/simulation_configs.csv")
    p.add_argument("--out",     default="paper_outputs/figures")
    args = p.parse_args()

    out_dir = ROOT / args.out / "simulations"
    out_dir.mkdir(parents=True, exist_ok=True)

    cache_dir    = ROOT / args.cache
    results_dir  = ROOT / args.results
    configs_csv  = ROOT / args.configs

    artifact_cache = {}

    for fig_num, panel, metric, sim_id, n_classes in PANELS:
        if n_classes not in artifact_cache:
            artifact_cache[n_classes] = load_artifact(n_classes, cache_dir)
        coords, trees = artifact_cache[n_classes]

        values  = load_values(sim_id, metric, trees, results_dir)
        gen_idx = find_generator(sim_id, trees, configs_csv) if metric != "depth" else \
                  find_generator(sim_id, trees, configs_csv)

        if fig_num == "appendix":
            fname = f"figappendix_{panel}_{metric}.pdf"
        else:
            fname = f"fig{fig_num}_{panel}_C{n_classes}_{metric}.pdf"
        make_plot(coords, values, gen_idx, metric, out_dir / fname)


if __name__ == "__main__":
    main()
