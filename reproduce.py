"""Single-command entry point for reproducing all paper results. See README."""
import argparse
import csv
import subprocess
import sys
from pathlib import Path
import shutil

SIMS_CSV    = "zenodo/simulation_configs.csv"
DT_SIMS_CSV   = "zenodo/simulation_configs_dt.csv"
SELECTED_CSV = "_selected_sims.csv"


def run(cmd, **kwargs):
    print(f"\n$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True, **kwargs)


def _simulation_input(classes):
    """Return path to the simulation CSV to use, filtering by class count if given."""
    if not classes:
        if Path(SIMS_CSV).exists():
            return SIMS_CSV
        if Path("zenodo/simulation_configs.json").exists():
            return "zenodo/simulation_configs.json"
        raise SystemExit(
            f"No simulation input found. Download {SIMS_CSV} with:\n"
            "  python scripts/download_zenodo.py --what sims\n"
            "or run: python scripts/csv_to_configs.py"
        )

    keep = set(classes)
    with open(SIMS_CSV, newline="") as f:
        reader = csv.DictReader(f)
        rows = [r for r in reader if int(r["n_classes"]) in keep]
        fieldnames = reader.fieldnames

    if not rows:
        raise SystemExit(f"No simulations found for n_classes in {sorted(keep)}")

    with open(SELECTED_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"  Selected {len(rows)} simulations for classes {sorted(keep)}")
    return SELECTED_CSV


def step_paper():
    print("\n=== Paper figures and tables ===")

    tmp = Path("paper_outputs/ablation_summaries")

    run([sys.executable, "scripts/make_dataset_plots.py", "--cache", "zenodo", "--out", "paper_outputs/figures"])
    run([sys.executable, "scripts/make_simulation_plots.py", "--out", "paper_outputs/figures"])
    run([sys.executable, "scripts/make_ablation_summaries.py", "--out", str(tmp)])
    run([sys.executable, "scripts/make_paper_tables.py", "--all", "--out", "paper_outputs/tables"])

    shutil.rmtree(tmp, ignore_errors=True)
    print(f"\nRemoved temporary files: {tmp}")


def step_artifacts():
    print("\n=== Step 1: Build tree-space artifacts ===")
    run([sys.executable, "scripts/make_artifacts.py", "--all"])


def step_score(bootstrap):
    print("\n=== Step 2: Score real datasets ===")
    run([sys.executable, "run_datasets.py", "--bootstrap", str(bootstrap)])


def step_simulation(classes):
    print("\n=== Step 3: Bias-variance simulation ===")
    inp = _simulation_input(classes)
    run([sys.executable, "run_simulations.py", "--input", inp])
    if Path(DT_SIMS_CSV).exists():
        run([sys.executable, "run_simulations.py", "--input", DT_SIMS_CSV,
             "--output", "bias-analysis/raw_simulation_results_dt"])


def step_moran(classes):
    print("\n=== Step 4: Simulation Moran analysis ===")
    inp = _simulation_input(classes)
    run([
        sys.executable, "scripts/moran_simulation_analysis.py",
        "--selected", inp,
        "--results", "bias-analysis/raw_simulation_results",
        "--out", "zenodo/simulation_scores",
    ])
    if (Path(DT_SIMS_CSV).exists() and
            Path("bias-analysis/raw_simulation_results_dt").exists()):
        run([
            sys.executable, "scripts/moran_simulation_analysis.py",
            "--selected", DT_SIMS_CSV,
            "--results", "bias-analysis/raw_simulation_results_dt",
            "--out", "zenodo/simulation_scores_dt",
        ])


def step_tables():
    print("\n=== Step 5: LaTeX tables ===")
    print("\n-- Table 5 (real-dataset spatial stats) --")
    run([sys.executable, "scripts/make_dataset_tables.py", "--table", "5"])
    print("\n-- Simulation tables --")
    for table in ("moran-main", "enrichment", "appendix-morans"):
        run([sys.executable, "scripts/make_simulation_tables.py", "--table", table,
             "--summary-dir", "zenodo/simulation_scores"])
    if Path("zenodo/simulation_scores_dt").exists():
        print("\n-- DT simulation tables --")
        for table in ("dt", "dt-extended"):
            run([sys.executable, "scripts/make_simulation_tables.py", "--table", table,
                 "--summary-dir", "zenodo/simulation_scores_dt"])


def step_plots(from_zenodo=False):
    print("\n=== Step 6: Embedding plots ===")
    cmd = [sys.executable, "scripts/make_dataset_plots.py"]
    if from_zenodo:
        run([sys.executable, "scripts/download_zenodo.py", "--what", "artifacts"])
        cmd += ["--cache", "zenodo"]
    run(cmd)


def step_download():
    print("\n=== Step 0: Download Zenodo artifacts ===")
    run([sys.executable, "scripts/download_zenodo.py", "--what",
         "artifacts", "sims", "sims_dt", "tree_metrics",
         "simulation_scores", "simulation_scores_dt", "rawdata"])


STEPS = ("download", "artifacts", "score", "sim", "moran", "tables", "plots")


def main():
    parser = argparse.ArgumentParser(description="Reproduce all paper results")
    parser.add_argument("--paper", action="store_true",
                help="Recreate exact paper figures and tables from zenodo/")
    parser.add_argument(
        "--steps", nargs="+", default=list(STEPS), choices=list(STEPS),
        help="Which steps to run (default: all)",
    )
    parser.add_argument(
        "--bootstrap", type=int, default=100,
        help="Bootstrap reps for step 2 (default: 100)",
    )
    parser.add_argument(
        "--classes", nargs="+", type=int, default=None,
        help="Restrict sim/moran steps to these class counts, e.g. --classes 6 7",
    )
    args = parser.parse_args()
    if args.paper:
        step_paper()
        return

    for step in args.steps:
        if step == "score":
            step_score(args.bootstrap)
        elif step == "sim":
            step_simulation(args.classes)
        elif step == "moran":
            step_moran(args.classes)
        elif step == "tables":
            step_tables()
        elif step == "plots":
            step_plots()
        elif step == "download":
            step_download()
        else:
            step_artifacts()


if __name__ == "__main__":
    main()
