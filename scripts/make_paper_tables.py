"""Generate all paper LaTeX tables by dispatching to the individual table scripts."""
import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper_outputs" / "tables"
SIMULATION_SCORES_DIR    = ROOT / "zenodo" / "simulation_scores"
DT_SIMULATION_SCORES_DIR = ROOT / "zenodo" / "simulation_scores_dt"

_SIMULATION_EXPERIMENTS = [
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

_SIMULATION_TABLE_NAMES = {"scaling", "enrichment", "appendix-morans"}

_DT_EXPERIMENTS = [
    "6_class_10_features_dense_sparsity",
    "6_class_1000_features_sparse_sparsity",
    "6_class_1000_features_dense_sparsity",
]

TABLES = {
    "ablation-main": (
        "ablation_1000_sparse.tex",
        ["scripts/make_ablation_tables.py", "--table", "main"],
    ),
    "ablation-appendix": (
        "ablation_appendix.tex",
        ["scripts/make_ablation_tables.py", "--table", "appendix"],
    ),
    "scaling": (
        "simulation_scaling.tex",
        ["scripts/make_simulation_tables.py", "--table", "moran-main", "--summary-dir", str(SIMULATION_SCORES_DIR)],
    ),
    "enrichment": (
        "nearest_generator_enrichment.tex",
        ["scripts/make_simulation_tables.py", "--table", "enrichment", "--summary-dir", str(SIMULATION_SCORES_DIR)],
    ),
    "appendix-morans": (
        "appendix_morans.tex",
        ["scripts/make_simulation_tables.py", "--table", "appendix-morans", "--summary-dir", str(SIMULATION_SCORES_DIR)],
    ),
    "realdata-spatial": (
        "realdata_spatial.tex",
        ["scripts/make_dataset_tables.py", "--table", "5"],
    ),
    "realdata-heuristic": (
        "realdata_heuristic.tex",
        ["scripts/make_dataset_tables.py", "--table", "heuristic", "--plot-dir", "paper_outputs/figures"],
    ),
    "dt-simulation": (
        "simulation_dt.tex",
        ["scripts/make_simulation_tables.py", "--table", "dt",
         "--summary-dir", str(DT_SIMULATION_SCORES_DIR)],
    ),
    "dt-simulation-extended": (
        "simulation_dt_extended.tex",
        ["scripts/make_simulation_tables.py", "--table", "dt-extended",
         "--summary-dir", str(DT_SIMULATION_SCORES_DIR)],
    ),
}

DATASET_TABLE = r"""\begin{table}[t]\centering
\caption{The six real datasets. Trees evaluated is the full enumeration at 6 and 7 classes and a 15{,}000-tree sample for the 8-, 9-, 10-, and 19-class datasets.}
\label{tab:realdata-datasets}\small\renewcommand{\arraystretch}{1.28}
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lccccc@{}}
\toprule
Dataset & Classes & Samples & Features & Split & Trees evaluated \\
\midrule
Glass Identification & 6  & 214     & 9   & 70/30 stratified & 945 \\
Steel Plates Faults  & 7  & 1{,}941 & 27  & 70/30 stratified & 10{,}395 \\
Mice Protein         & 8  & 1{,}080 & 77  & 70/30 stratified & 15{,}000 \\
Urban Land Cover     & 9  & 675     & 147 & UCI predefined   & 15{,}000 \\
Pen-Based Digits     & 10 & 10{,}992 & 16  & 70/30 stratified & 15{,}000 \\
Soybean              & 19 & 307     & 35  & 70/30 stratified & 15{,}000 \\
\bottomrule
\end{tabular*}
\end{table}
"""


def compute_simulation_scores():
    expected = [SIMULATION_SCORES_DIR / f"{e}_simulation_scores.csv"
                for e in _SIMULATION_EXPERIMENTS]
    if all(p.exists() for p in expected):
        print("  Simulation scores already present — skipping recomputation")
        return
    src = ROOT / "zenodo" / "raw_simulation_results"
    if not src.exists():
        print(f"  Raw simulation results not found at {src} — skipping")
        return
    SIMULATION_SCORES_DIR.mkdir(parents=True, exist_ok=True)
    print("Computing simulation scores...")
    subprocess.run([
        sys.executable,
        "scripts/moran_simulation_analysis.py",
        "--selected", "zenodo/simulation_configs.csv",
        "--results", str(src),
        "--cache", "zenodo",
        "--out", str(SIMULATION_SCORES_DIR),
        "--experiments", *_SIMULATION_EXPERIMENTS,
    ], cwd=ROOT, check=True)


def compute_dt_simulation_scores():
    expected = [DT_SIMULATION_SCORES_DIR / f"{e}_simulation_scores_dt.csv"
                for e in _DT_EXPERIMENTS]
    if all(p.exists() for p in expected):
        print("  DT simulation scores already present — skipping recomputation")
        return
    src = ROOT / "zenodo" / "raw_simulation_results_dt"
    if not src.exists():
        print(f"  DT boot results not found at {src} — skipping DT scores")
        return
    DT_SIMULATION_SCORES_DIR.mkdir(parents=True, exist_ok=True)
    print("Computing DT simulation scores...")
    subprocess.run([
        sys.executable,
        "scripts/moran_simulation_analysis.py",
        "--selected", "zenodo/simulation_configs_dt.csv",
        "--results", str(src),
        "--cache", "zenodo",
        "--out", str(DT_SIMULATION_SCORES_DIR),
        "--experiments", *_DT_EXPERIMENTS,
    ], cwd=ROOT, check=True)


def write_table(name, out):
    fname, cmd = TABLES[name]
    path = out / fname
    print(path)

    with path.open("w", encoding="utf-8") as f:
        subprocess.run([sys.executable, *cmd], cwd=ROOT, stdout=f, check=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--table", choices=list(TABLES))
    p.add_argument("--all", action="store_true")
    p.add_argument("--out", default=OUT)
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    names = list(TABLES) if (args.all or not args.table) else [args.table]

    if args.all or not args.table or args.table in _SIMULATION_TABLE_NAMES:
        compute_simulation_scores()

    if args.all or args.table in ("dt-simulation", "dt-simulation-extended"):
        compute_dt_simulation_scores()

    for name in names:
        if name in ("dt-simulation", "dt-simulation-extended") and not DT_SIMULATION_SCORES_DIR.exists():
            print(f"Skipping {name} (no scores at {DT_SIMULATION_SCORES_DIR})")
            continue
        write_table(name, out)

    if args.all or args.table is None:
        path = out / "realdata_datasets.tex"
        path.write_text(DATASET_TABLE, encoding="utf-8")
        print(path)


if __name__ == "__main__":
    main()