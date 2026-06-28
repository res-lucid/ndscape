"""Download large files from the paper's Zenodo deposit.

Usage
-----
    python scripts/download_zenodo.py                         # all files
    python scripts/download_zenodo.py --what artifacts        # .joblib embedding artifacts
    python scripts/download_zenodo.py --what sims             # zenodo/simulation_configs.csv
    python scripts/download_zenodo.py --what sims_dt          # zenodo/simulation_configs_dt.csv
    python scripts/download_zenodo.py --what tree_metrics     # per-tree score CSVs
    python scripts/download_zenodo.py --what simulation_scores  # Moran summary CSVs
    python scripts/download_zenodo.py --what boot             # raw_simulation_results/ boot CSVs (zipped)
    python scripts/download_zenodo.py --what rawdata          # raw UCI dataset CSVs → zenodo/data/

Zenodo folder layout
--------------------
    zenodo/
    ├── *.joblib                  # embedding artifacts  (group: artifacts)
    ├── simulation_scores/        # Moran summary CSVs   (group: simulation_scores)
    ├── tree_metrics/             # per-tree score CSVs  (group: tree_metrics)
    ├── raw_simulation_results/   # boot results CSVs    (group: boot, zip on Zenodo)
    └── data/                    # raw UCI datasets      (group: rawdata)

    zenodo/simulation_configs.csv     # simulation configs  (group: sims)
    zenodo/simulation_configs_dt.csv  # DT simulation configs (group: sims_dt)
"""
import argparse
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).parent.parent

ZENODO_DOI  = "10.5281/zenodo.20456943"
ZENODO_BASE = f"https://zenodo.org/record/{ZENODO_DOI.split('.')[-1]}/files"

# Each entry: (remote path on Zenodo, local destination, group)
FILES = [
    # ── embedding artifacts (.joblib) ─────────────────────────────────────────
    ("art_1_C6_N945_s0_d2.joblib",
     ROOT / "zenodo" / "art_1_C6_N945_s0_d2.joblib", "artifacts"),
    ("art_1_C7_N10395_s0_d2.joblib",
     ROOT / "zenodo" / "art_1_C7_N10395_s0_d2.joblib", "artifacts"),
    ("art_1_C8_N15000_s0_d2.joblib",
     ROOT / "zenodo" / "art_1_C8_N15000_s0_d2.joblib", "artifacts"),
    ("art_1_C9_N15000_s0_d2.joblib",
     ROOT / "zenodo" / "art_1_C9_N15000_s0_d2.joblib", "artifacts"),
    ("art_1_C10_N15000_s0_d2.joblib",
     ROOT / "zenodo" / "art_1_C10_N15000_s0_d2.joblib", "artifacts"),
    ("art_1_C19_N15000_s0_d2.joblib",
     ROOT / "zenodo" / "art_1_C19_N15000_s0_d2.joblib", "artifacts"),
    ("bow_1_C6_N945_s0_d2.joblib",
     ROOT / "zenodo" / "bow_1_C6_N945_s0_d2.joblib", "artifacts"),
    ("bow_1_C7_N10395_s0_d2.joblib",
     ROOT / "zenodo" / "bow_1_C7_N10395_s0_d2.joblib", "artifacts"),
    ("rf_1_C6_N945_s0_d2.joblib",
     ROOT / "zenodo" / "rf_1_C6_N945_s0_d2.joblib", "artifacts"),
    ("rf_1_C7_N10395_s0_d2.joblib",
     ROOT / "zenodo" / "rf_1_C7_N10395_s0_d2.joblib", "artifacts"),

    # ── simulation configs ────────────────────────────────────────────────────
    ("simulation_configs.csv",
     ROOT / "zenodo" / "simulation_configs.csv", "sims"),
    ("simulation_configs_dt.csv",
     ROOT / "zenodo" / "simulation_configs_dt.csv", "sims_dt"),

    # ── DT Moran summary CSVs → zenodo/simulation_scores_dt/ ─────────────────
    ("simulation_scores_dt/6_class_10_features_dense_sparsity_simulation_scores_dt.csv",
     ROOT / "zenodo" / "simulation_scores_dt" / "6_class_10_features_dense_sparsity_simulation_scores_dt.csv", "simulation_scores_dt"),
    ("simulation_scores_dt/6_class_1000_features_sparse_sparsity_simulation_scores_dt.csv",
     ROOT / "zenodo" / "simulation_scores_dt" / "6_class_1000_features_sparse_sparsity_simulation_scores_dt.csv", "simulation_scores_dt"),
    ("simulation_scores_dt/6_class_1000_features_dense_sparsity_simulation_scores_dt.csv",
     ROOT / "zenodo" / "simulation_scores_dt" / "6_class_1000_features_dense_sparsity_simulation_scores_dt.csv", "simulation_scores_dt"),

    # ── per-tree score CSVs → zenodo/tree_metrics/ ───────────────────────────
    ("tree_metrics/tree_metrics_glass_identification.csv",
     ROOT / "zenodo" / "tree_metrics" / "tree_metrics_glass_identification.csv", "tree_metrics"),
    ("tree_metrics/tree_metrics_steel_plates_faults.csv",
     ROOT / "zenodo" / "tree_metrics" / "tree_metrics_steel_plates_faults.csv", "tree_metrics"),
    ("tree_metrics/tree_metrics_mice_protein.csv",
     ROOT / "zenodo" / "tree_metrics" / "tree_metrics_mice_protein.csv", "tree_metrics"),
    ("tree_metrics/tree_metrics_urban_land_cover.csv",
     ROOT / "zenodo" / "tree_metrics" / "tree_metrics_urban_land_cover.csv", "tree_metrics"),
    ("tree_metrics/tree_metrics_pen_based_recognition_of_handwritten_digits_81.csv",
     ROOT / "zenodo" / "tree_metrics" / "tree_metrics_pen_based_recognition_of_handwritten_digits_81.csv", "tree_metrics"),
    ("tree_metrics/tree_metrics_soybean_large_122.csv",
     ROOT / "zenodo" / "tree_metrics" / "tree_metrics_soybean_large_122.csv", "tree_metrics"),

    # ── Moran summary CSVs → zenodo/simulation_scores/ ───────────────────────
    ("simulation_scores/6_class_10_features_dense_sparsity_simulation_scores.csv",
     ROOT / "zenodo" / "simulation_scores" / "6_class_10_features_dense_sparsity_simulation_scores.csv", "simulation_scores"),
    ("simulation_scores/6_class_1000_features_dense_sparsity_simulation_scores.csv",
     ROOT / "zenodo" / "simulation_scores" / "6_class_1000_features_dense_sparsity_simulation_scores.csv", "simulation_scores"),
    ("simulation_scores/6_class_1000_features_sparse_sparsity_simulation_scores.csv",
     ROOT / "zenodo" / "simulation_scores" / "6_class_1000_features_sparse_sparsity_simulation_scores.csv", "simulation_scores"),
    ("simulation_scores/7_class_10_features_dense_sparsity_simulation_scores.csv",
     ROOT / "zenodo" / "simulation_scores" / "7_class_10_features_dense_sparsity_simulation_scores.csv", "simulation_scores"),
    ("simulation_scores/7_class_1000_features_dense_sparsity_simulation_scores.csv",
     ROOT / "zenodo" / "simulation_scores" / "7_class_1000_features_dense_sparsity_simulation_scores.csv", "simulation_scores"),
    ("simulation_scores/7_class_1000_features_sparse_sparsity_simulation_scores.csv",
     ROOT / "zenodo" / "simulation_scores" / "7_class_1000_features_sparse_sparsity_simulation_scores.csv", "simulation_scores"),
    ("simulation_scores/8_class_1000_features_sparse_sparsity_simulation_scores.csv",
     ROOT / "zenodo" / "simulation_scores" / "8_class_1000_features_sparse_sparsity_simulation_scores.csv", "simulation_scores"),
    ("simulation_scores/9_class_1000_features_sparse_sparsity_simulation_scores.csv",
     ROOT / "zenodo" / "simulation_scores" / "9_class_1000_features_sparse_sparsity_simulation_scores.csv", "simulation_scores"),
    ("simulation_scores/10_class_1000_features_sparse_sparsity_simulation_scores.csv",
     ROOT / "zenodo" / "simulation_scores" / "10_class_1000_features_sparse_sparsity_simulation_scores.csv", "simulation_scores"),

    # ── raw UCI datasets → zenodo/data/ ──────────────────────────────────────
    ("data/glass_identification.csv",
     ROOT / "zenodo" / "data" / "glass_identification.csv", "rawdata"),
    ("data/steel_plates_faults.csv",
     ROOT / "zenodo" / "data" / "steel_plates_faults.csv", "rawdata"),
    ("data/mice_protein.csv",
     ROOT / "zenodo" / "data" / "mice_protein.csv", "rawdata"),
    ("data/urban_land_cover.csv",
     ROOT / "zenodo" / "data" / "urban_land_cover.csv", "rawdata"),
    ("data/pen_based_recognition_of_handwritten_digits_81.csv",
     ROOT / "zenodo" / "data" / "pen_based_recognition_of_handwritten_digits_81.csv", "rawdata"),
    ("data/soybean_large_122.csv",
     ROOT / "zenodo" / "data" / "soybean_large_122.csv", "rawdata"),
]


def _download_boot_dt():
    """Download raw_simulation_results_dt.zip and extract into zenodo/raw_simulation_results_dt/."""
    dest_dir = ROOT / "zenodo" / "raw_simulation_results_dt"
    dest_dir.mkdir(parents=True, exist_ok=True)

    zip_url = f"{ZENODO_BASE}/raw_simulation_results_dt.zip?download=1"
    zip_path = ROOT / "zenodo" / "raw_simulation_results_dt.zip"
    print("Downloading raw_simulation_results_dt.zip ...")
    _download(zip_url, zip_path)

    print("Extracting ...")
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            fname = Path(member).name
            if not fname.endswith(".csv"):
                continue
            (dest_dir / fname).write_bytes(zf.read(member))
    zip_path.unlink()
    extracted = list(dest_dir.glob("boot_results_*.csv"))
    print(f"  extracted {len(extracted)} boot CSVs → zenodo/raw_simulation_results_dt/")


def _download_boot():
    """Download raw_simulation_results.zip and extract into zenodo/raw_simulation_results/."""
    dest_dir = ROOT / "zenodo" / "raw_simulation_results"
    dest_dir.mkdir(parents=True, exist_ok=True)

    zip_url = f"{ZENODO_BASE}/raw_simulation_results.zip?download=1"
    zip_path = ROOT / "zenodo" / "raw_simulation_results.zip"
    print("Downloading raw_simulation_results.zip ...")
    _download(zip_url, zip_path)

    print("Extracting ...")
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            # Strip any leading directory component so CSVs land flat in dest_dir
            fname = Path(member).name
            if not fname.endswith(".csv"):
                continue
            (dest_dir / fname).write_bytes(zf.read(member))
    zip_path.unlink()
    extracted = list(dest_dir.glob("boot_results_*.csv"))
    print(f"  extracted {len(extracted)} boot CSVs → zenodo/raw_simulation_results/")


def _download(url, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  {dest.name} ... ", end="", flush=True)
    urllib.request.urlretrieve(url, dest)
    size = dest.stat().st_size / 1024
    print(f"{size:.0f} KB")


def main():
    parser = argparse.ArgumentParser(description="Download Zenodo deposit files")
    parser.add_argument(
        "--what", nargs="+",
        choices=["artifacts", "sims", "sims_dt", "tree_metrics", "simulation_scores",
                 "simulation_scores_dt", "rawdata", "boot", "boot_dt"],
        default=["artifacts", "sims", "sims_dt", "tree_metrics", "simulation_scores",
                 "simulation_scores_dt", "rawdata", "boot", "boot_dt"],
        help="Which file groups to download (default: all groups)",
    )
    args = parser.parse_args()

    if ZENODO_DOI == "FIXME":
        sys.exit("Set ZENODO_DOI in scripts/download_zenodo.py before running.")

    groups = set(args.what)
    do_boot = "boot" in groups
    do_boot_dt = "boot_dt" in groups
    groups.discard("boot")
    groups.discard("boot_dt")

    # Download all non-boot files first (ensures simulation_configs.csv is present
    # before the boot group tries to read it to enumerate filenames).
    for fname, dest, group in FILES:
        if group not in groups:
            continue
        # The Zenodo deposit is flat (no folders); fname carries a
        # directory-style prefix here only for readability/grouping above.
        url = f"{ZENODO_BASE}/{Path(fname).name}?download=1"
        _download(url, dest)

    if do_boot:
        _download_boot()

    if do_boot_dt:
        _download_boot_dt()


if __name__ == "__main__":
    main()
