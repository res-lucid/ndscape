"""Run moran_simulation_analysis.py for all ablation methods and experiments."""
import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ZENODO = ROOT / "zenodo"
OUT = ROOT / "paper_outputs" / "ablation_summaries"

EXPERIMENTS = [
    "6_class_10_features_dense_sparsity",
    "6_class_1000_features_dense_sparsity",
    "6_class_1000_features_sparse_sparsity",
    "7_class_10_features_dense_sparsity",
    "7_class_1000_features_dense_sparsity",
    "7_class_1000_features_sparse_sparsity",
]

METHODS = [
    ("tokeniser", "art_1"),
    ("bow", "bow_1"),
    ("rf", "rf_1"),
]

p = argparse.ArgumentParser()
p.add_argument("--selected", default=str(ZENODO / "simulation_configs.csv"))
p.add_argument("--results", default=ZENODO / "raw_simulation_results")
p.add_argument("--out", default=OUT)
p.add_argument("--permutations", type=int, default=999)
args = p.parse_args()

Path(args.out).mkdir(parents=True, exist_ok=True)

for name, prefix in METHODS:
    print(f"\n=== {name} ===")

    subprocess.run([
        sys.executable,
        "scripts/moran_simulation_analysis.py",
        "--selected", str(args.selected),
        "--results", str(args.results),
        "--cache", str(ZENODO),
        "--out", str(Path(args.out) / name),
        "--artifact-prefix", prefix,
        "--permutations", str(args.permutations),
        "--experiments", *EXPERIMENTS,
    ], cwd=ROOT, check=True)