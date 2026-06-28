"""Produce PDF and interactive HTML embedding plots for each real dataset.

Usage
-----
    python scripts/make_dataset_plots.py
    python scripts/make_dataset_plots.py --datasets glass_identification steel_plates_faults
    python scripts/make_dataset_plots.py --no-methods
"""
import argparse
import ast
import json
import sys
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_hex
from matplotlib.lines import Line2D
from bokeh.io import output_file, save
from bokeh.layouts import row as bk_row
from bokeh.models import ColorBar, ColumnDataSource, CustomJS, Div, HoverTool, LinearColorMapper, TapTool
from bokeh.plotting import figure
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

import core as mh                                              # noqa: E402
from nd import ACND, BBoK, BestOfK, CBND, NDC, RPND, RandomGeneration  # noqa: E402

DATASETS = [
    ("glass_identification",                           6),
    ("steel_plates_faults",                            7),
    ("mice_protein",                                   8),
    ("urban_land_cover",                               9),
    ("pen_based_recognition_of_handwritten_digits_81", 10),
    ("soybean_large_122",                              19),
]
BRIGHT   = LinearSegmentedColormap.from_list("bright_RYG", ["#ff2d2d", "#fff176", "#00e676"], N=100)
MARKERS  = {"RPND": "*", "BBoK": "P", "BoK": "^", "ACND": "X", "CBND": "D", "NDC": "s"}

CLASS_NAMES = {
    "glass_identification": {
        0: "building_windows_float_processed",
        1: "building_windows_non_float_processed",
        2: "vehicle_windows_float_processed",
        3: "containers", 4: "tableware", 5: "headlamps",
    },
    "steel_plates_faults": {
        0: "Pastry", 1: "Z_Scratch", 2: "K_Scratch", 3: "Stains",
        4: "Dirtiness", 5: "Bumps", 6: "Other_Faults",
    },
    "mice_protein": {
        0: "control, stimulated, memantine",   1: "control, stimulated, saline",
        2: "control, not stimulated, memantine", 3: "control, not stimulated, saline",
        4: "trisomic, stimulated, memantine",  5: "trisomic, stimulated, saline",
        6: "trisomic, not stimulated, memantine", 7: "trisomic, not stimulated, saline",
    },
    "urban_land_cover": {
        0: "asphalt", 1: "building", 2: "car", 3: "concrete", 4: "grass",
        5: "pool", 6: "shadow", 7: "soil", 8: "tree",
    },
    "pen_based_recognition_of_handwritten_digits_81": {i: str(i) for i in range(10)},
    "soybean_large_122": {
        0: "2,4-D injury", 1: "alternaria leaf spot", 2: "anthracnose",
        3: "bacterial blight", 4: "bacterial pustule", 5: "brown spot",
        6: "brown stem rot", 7: "charcoal rot", 8: "cyst nematode",
        9: "diaporthe pod/stem blight", 10: "diaporthe stem canker",
        11: "downy mildew", 12: "frog-eye leaf spot", 13: "herbicide injury",
        14: "phyllosticta leaf spot", 15: "phytophthora rot",
        16: "powdery mildew", 17: "purple seed stain", 18: "rhizoctonia root rot",
    },
}

SHORT_NAMES = {
    "glass_identification": {
        "building_windows_float_processed":     "building (float)",
        "building_windows_non_float_processed": "building (non float)",
        "vehicle_windows_float_processed":      "vehicle (float)",
    },
}

NAMES = {
    ds: {k: SHORT_NAMES.get(ds, {}).get(v, v) for k, v in cls.items()}
    for ds, cls in CLASS_NAMES.items()
}


def static_pdf(coords, vals, best_i, method_df, label, cmap, path):
    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=vals, cmap=cmap,
                    vmin=np.nanmin(vals), vmax=np.nanmax(vals),
                    s=30, linewidths=0, rasterized=True)
    fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02).set_label(label)
    if method_df is not None:
        for _, r in method_df.iterrows():
            ax.scatter(r.x, r.y, marker=MARKERS[r.method], s=190,
                       facecolors="white", edgecolors="black", linewidths=1.2,  alpha=0.75, zorder=10)
        ax.legend(
            handles=[Line2D([0], [0], marker=MARKERS[n], linestyle="", markersize=8,
                            markerfacecolor="white", markeredgecolor="black",
                            markeredgewidth=1.1, label=f": {n}")
                     for n in method_df.method],
            loc="lower center", bbox_to_anchor=(0.5, 1.01),
            ncol=len(method_df), frameon=False, fontsize=8,
            handletextpad=0.15, columnspacing=1.15, borderaxespad=0.0)
    ax.scatter(coords[best_i, 0], coords[best_i, 1],
               marker="x", s=220, c="black", linewidths=2.2, zorder=12)
    ax.set_xlabel("MDS-1"); ax.set_ylabel("MDS-2")
    ax.grid(True, lw=0.3, alpha=0.4)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def nd_to_ascii(t, names, C):
    split_map = {frozenset(L)|frozenset(R): (frozenset(L), frozenset(R)) for L, R in t}
    root = max(split_map, key=len)

    def lbl(node):
        def n(c): return names.get(c, str(c)) if C <= 10 else str(c)
        if node not in split_map:
            return n(next(iter(node)))
        L, R = split_map[node]
        return ".".join(n(c) for c in sorted(L)) + " | " + ".".join(n(c) for c in sorted(R))

    GAP, centers = 3, {}

    def measure(node, offset):
        label = lbl(node)
        if node not in split_map:
            centers[node] = offset + len(label) // 2
            return len(label)
        L, R = split_map[node]
        lw  = measure(L, offset)
        rw  = measure(R, offset + lw + GAP)
        half = (len(label) + 1) // 2
        extra = max(0, centers[L] + centers[R] + 2*half - 2*offset - 2*lw - 2*GAP - 2*rw)
        if extra:
            rw = measure(R, offset + lw + GAP + extra)
        centers[node] = (centers[L] + centers[R]) // 2
        return lw + GAP + extra + rw

    measure(root, 0)
    canvas = {}

    def place(s, row, col):
        for i, c in enumerate(s):
            canvas.setdefault((row, col + i), c)

    node_row = {root: 0}
    q = deque([root])
    while q:
        node = q.popleft()
        row = node_row[node]
        label = lbl(node)
        place(label, row, centers[node] - len(label) // 2)
        if node in split_map:
            L, R = split_map[node]
            place("/", row + 1, centers[L])
            place("\\", row + 1, centers[R])
            node_row[L] = node_row[R] = row + 2
            q.extend([L, R])

    if not canvas: return ""
    mr, mc = max(r for r, _ in canvas), max(c for _, c in canvas)
    lines = ["".join(canvas.get((r, c), " ") for c in range(mc + 1)).rstrip()
             for r in range(mr + 1)]
    if C > 10:
        lines += ["", "Classes:"] + [f"  {c}: {n}" for c, n in sorted(names.items())]
    return "\n".join(lines)


def bokeh_html(coords, labels, df, metric, label, path, method_df=None, dataset=None, C=6):
    vals    = df[metric].to_numpy(float)
    palette = [to_hex(BRIGHT(i / 99)) for i in range(100)]
    if metric == "model_var":
        palette = list(reversed(palette))
    names = NAMES.get(dataset, {})
    trees = [nd_to_ascii(ast.literal_eval(t) if isinstance(t, str) else t, names, C)
             for t in df["tree"]]
    source = ColumnDataSource(dict(
        x=coords[:, 0], y=coords[:, 1], value=vals,
        accuracy=df["score"].to_numpy(float),
        model_var=df["model_var"].to_numpy(float),
        logloss=df["logloss"].to_numpy(float),
        cluster=labels, tree_id=np.arange(len(coords)), tree=trees,
    ))
    mapper = LinearColorMapper(palette=palette,
                               low=float(np.nanmin(vals)), high=float(np.nanmax(vals)))
    p = figure(width=800, height=520, x_axis_label="MDS-1", y_axis_label="MDS-2",
               tools="pan,wheel_zoom,box_zoom,reset,save,tap", title=Path(path).stem)
    pts = p.scatter("x", "y", source=source, size=6, line_color="black", line_width=0.4,
                    fill_color={"field": "value", "transform": mapper}, fill_alpha=0.9)
    best_i = int(np.nanargmax(df["score"].to_numpy()))
    p.scatter([coords[best_i, 0]], [coords[best_i, 1]],
              marker="x", size=16, line_color="black", line_width=2, legend_label="Best ND")
    if method_df is not None:
        bm = {"RPND": "star", "BBoK": "plus", "BoK": "triangle",
            "ACND": "circle_x", "CBND": "diamond", "NDC": "square"}
        for _, row in method_df.iterrows():
            p.scatter([row.x], [row.y], marker=bm.get(row.method, "circle"),
                      size=14, line_color="black", fill_color="white",
                      line_width=1.2, fill_alpha=0.75, legend_label=row.method)
    p.add_tools(HoverTool(renderers=[pts], tooltips=[
        ("accuracy", "@accuracy{0.000}"), ("model_var", "@model_var{0.000000}"),
        ("logloss", "@logloss{0.000}"), ("cluster", "@cluster"),
    ]))
    p.add_layout(ColorBar(color_mapper=mapper, label_standoff=8, title=label), "right")
    if p.legend:
        lg = p.legend[0]
        p.legend.remove(lg)
        lg.click_policy = "mute"
        lg.label_text_font_size = "9pt"
        lg.stylesheets = [":host * { cursor: pointer !important; }"]
        p.add_layout(lg, "right")
    info = Div(text="<i>Click a point to inspect tree.</i>", width=600, height=520)
    source.selected.js_on_change("indices", CustomJS(args=dict(source=source, info=info), code='''
        if (!source.selected.indices.length) return;
        const i = source.selected.indices[0], d = source.data;
        info.text = '<b>Tree ' + d.tree_id[i] + '</b>  '
            + 'acc=' + d.accuracy[i].toFixed(3)
            + '  var=' + d.model_var[i].toFixed(6)
            + '  loss=' + d.logloss[i].toFixed(3)
            + '<pre style="font-size:11px;line-height:1.35;margin-top:8px;">'
            + d.tree[i] + '</pre>';
    '''))
    p.select_one(TapTool)
    output_file(path); save(bk_row(p, info))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets",   nargs="+", default=[d for d, _ in DATASETS])
    parser.add_argument("--model",      default="lr")
    parser.add_argument("--no-methods", action="store_true")
    parser.add_argument("--cache", default="zenodo")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    C_by_name = dict(DATASETS)
    for dataset in args.datasets:
        C   = C_by_name[dataset]
        xlsx = ROOT / "real-data-bias-analysis" / dataset / f"v_{dataset}_{args.model}_cluster_tables.xlsx"
        csv  = ROOT / "zenodo" / "tree_metrics" / f"tree_metrics_{dataset}.csv"
        print(f"\n=== {dataset} ===")

        if xlsx.exists():
            df = pd.read_excel(xlsx, sheet_name="tree_metrics")
        elif csv.exists():
            df = pd.read_csv(csv)
        else:
            raise FileNotFoundError(f"No tree metrics for {dataset}: run step 2 or download from Zenodo")
        art = mh.get_trees_and_artifact(list(range(C)), N=15_000, seed=0, cache_dir=args.cache)
        coords_raw = np.asarray(art["coords_plot"])
        labels_raw = np.asarray(art["labels_plot"])

        # Align artifact tree order to xlsx row order (guards against a reordered sheet)
        trees_plot = art.get("trees_plot", art.get("trees"))
        tree_to_idx = {str(t): i for i, t in enumerate(trees_plot)}
        order = [tree_to_idx[t] for t in df["tree"].astype(str)]
        coords = coords_raw[order]
        labels = labels_raw[order]

        method_df = None
        if not args.no_methods:
            raw = pd.read_csv(ROOT / "zenodo" / "data" / f"{dataset}.csv")
            X   = raw.drop(columns=[c for c in ("y", "split") if c in raw.columns]).to_numpy(float)
            y   = raw["y"].to_numpy(int)
            cats = tuple(map(int, np.unique(y)))
            tr  = (raw.index[raw["split"] == "train"].to_numpy() if "split" in raw.columns
                   else train_test_split(np.arange(len(y)), test_size=0.3, stratify=y, random_state=0)[0])
            sc  = StandardScaler().fit(X[tr])
            Xtr, ytr = sc.transform(X[tr]), y[tr]

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
                def mask(m): return tuple(int(c) for c in cats if int(m) & (1 << int(c)))
                return tuple((mask(L), mask(R)) for L, R in bit_nd if int(R) != 0)

            def canon(t):
                t = ast.literal_eval(t) if isinstance(t, str) else t
                return frozenset(tuple(sorted((tuple(sorted(L)), tuple(sorted(R))))) for L, R in t)

            # lookup is keyed by artifact index (not xlsx order); use coords_raw
            lookup = {canon(t): i for i, t in enumerate(art["trees"])}

            rows = []
            for name, bit_t in bits.items():
                t = decode(bit_t)
                k = canon(t)
                if k in lookup:
                    xy = coords_raw[lookup[k]]
                else:
                    vec = TfidfVectorizer(analyzer=mh.split_analyzer(C), use_idf=False, norm="l2")
                    X_  = vec.fit_transform([str(tr) for tr in art["trees"]] + [str(t)])
                    xy  = coords_raw[cosine_similarity(X_[-1], X_[:-1])[0].argmax()]
                rows.append({"method": name, "x": float(xy[0]), "y": float(xy[1])})
            method_df = pd.DataFrame(rows)

        out = Path(args.out) / dataset / "plots" if args.out else ROOT / "real-data-bias-analysis" / dataset / "plots"
        out.mkdir(parents=True, exist_ok=True)
        best_i = int(np.nanargmax(df["score"].to_numpy()))

        static_pdf(coords, df["score"].to_numpy(float),     best_i, method_df, "Accuracy",       BRIGHT,           out / f"{dataset}_{args.model}_accuracy.pdf")
        static_pdf(coords, df["model_var"].to_numpy(float), best_i, method_df, "Model variance",  BRIGHT.reversed(), out / f"{dataset}_{args.model}_model_variance.pdf")
        bokeh_html(coords, labels, df, "score",     "Accuracy",       str(out / f"{dataset}_{args.model}_accuracy_hover.html"),       method_df, dataset, C)
        bokeh_html(coords, labels, df, "model_var", "Model variance", str(out / f"{dataset}_{args.model}_model_variance_hover.html"), method_df, dataset, C)

        (out / "nd_info.json").write_text(json.dumps([
            {"tree_id": i, "accuracy": float(df["score"].iloc[i]),
             "model_var": float(df["model_var"].iloc[i]),
             "cluster": int(labels[i]),
             "x": float(coords[i, 0]),
             "y": float(coords[i, 1])}
            for i in range(len(df))
        ]))


if __name__ == "__main__":
    main()
