# ndscape: Nested-Dichotomy Tree-Space Embedding

Code for the paper. Embeds the space of nested-dichotomy (ND) structures via depth-weighted tokenisation and metric MDS, then maps bias, variance, and held-out performance across that space.

## Installation

```bash
pip install -r requirements.txt
```

Versions of `numpy`, `scipy`, and `scikit-learn` are pinned in `requirements.txt`. The MDS coordinates in `cache_nd/` were built with those exact versions; changing them will produce different (but statistically equivalent) coordinates.

## Quick check (~30 s)

```bash
pytest tests/
```

To also run the slower integration tests (require pre-built C=6 and C=7 cache files, or rebuild from scratch):

```bash
pytest tests/ -m integration
```

## Reproduce paper outputs (fast — no re-simulation)

Download the Zenodo deposit first, then regenerate all figures and tables in minutes:

```bash
python scripts/download_zenodo.py
python reproduce.py --paper
```

`--paper` runs four scripts in order:

1. `scripts/make_dataset_plots.py --cache zenodo --out paper_outputs/figures` — PDF and interactive HTML embedding maps for all 6 real datasets
2. `scripts/make_simulation_plots.py --out paper_outputs/figures` — simulation bias/variance/depth PDF maps (Figs 2–5 and appendix)
3. `scripts/make_ablation_summaries.py` — Moran spatial analysis for the tokeniser / BoW / RF ablation (reads `zenodo/raw_simulation_results/`)
4. `scripts/make_paper_tables.py --all --out paper_outputs/tables` — all LaTeX tables; uses pre-built scores from `zenodo/simulation_scores/` and `zenodo/simulation_scores_dt/` if present, otherwise recomputes from `zenodo/raw_simulation_results{,_dt}/`

Output lands in `paper_outputs/figures/` and `paper_outputs/tables/`.

## Full reproduction from scratch

```bash
python reproduce.py
```

| Step | Name | Script | Output |
|------|------|--------|--------|
| 0 | `download` | `scripts/download_zenodo.py` | `zenodo/` (artifacts, configs, tree\_metrics, simulation\_scores, data, boot results) |
| 1 | `artifacts` | `scripts/make_artifacts.py --all` | `cache_nd/*.joblib` |
| 2 | `score` | `run_datasets.py` | `real-data-bias-analysis/<dataset>/v_<dataset>_lr_cluster_tables.xlsx` |
| 3 | `sim` | `run_simulations.py` | `bias-analysis/raw_simulation_results/` (LR) and `bias-analysis/raw_simulation_results_dt/` (DT) |
| 4 | `moran` | `scripts/moran_simulation_analysis.py` | `zenodo/simulation_scores/` (LR) and `zenodo/simulation_scores_dt/` (DT) |
| 5 | `tables` | `scripts/make_paper_tables.py --all` | `paper_outputs/tables/*.tex` |
| 6 | `plots` | `scripts/make_dataset_plots.py`, `scripts/make_simulation_plots.py` | `paper_outputs/figures/` |

Run individual steps or subsets:

```bash
python reproduce.py --steps artifacts
python reproduce.py --steps score --bootstrap 20     # quick test with fewer bootstrap reps
python reproduce.py --steps sim moran --classes 6 7  # 6- and 7-class only
python reproduce.py --steps plots
```

**Note on step 2:** `run_datasets.py` scores every ND tree on each real dataset and writes xlsx results. These are also available pre-built in `zenodo/tree_metrics/` and are used automatically by `make_dataset_plots.py` and `make_dataset_tables.py` when the xlsx files are absent.

## Real-data tables and figures without re-scoring

`make_dataset_plots.py` and `make_dataset_tables.py` fall back to `zenodo/tree_metrics/<dataset>.csv` when the step-2 xlsx files are not present. After running `python scripts/download_zenodo.py`, the `--paper` command produces all real-data figures and tables without running step 2.

## Tests

```bash
pytest tests/                  # unit + snapshot tests (fast)
pytest tests/ -m integration   # integration tests (slow; requires pre-built C=6,7 artifacts)
```

The `tests/fixtures/` directory holds MD5 hashes of the tree lists, tokeniser output, and per-dataset scores.

## Datasets

The six datasets are from the UCI Machine Learning Repository (CC BY 4.0) and are archived on Zenodo. Download:

```bash
python scripts/download_zenodo.py --what rawdata
```

Or place CSV files (with a `y` column for the class label) in `zenodo/data/`:

| File | Classes | Trees evaluated |
|------|---------|-----------------|
| `glass_identification.csv` | 6 | 945 (exhaustive) |
| `steel_plates_faults.csv` | 7 | 10 395 (exhaustive) |
| `mice_protein.csv` | 8 | 15 000 (sampled) |
| `urban_land_cover.csv` | 9 | 15 000 (sampled) |
| `pen_based_recognition_of_handwritten_digits_81.csv` | 10 | 15 000 (sampled) |
| `soybean_large_122.csv` | 19 | 15 000 (sampled) |

### Data sources

| Dataset | UCI URL |
|---------|---------|
| Glass Identification | https://archive.ics.uci.edu/dataset/42 |
| Steel Plates Faults | https://archive.ics.uci.edu/dataset/198 |
| Mice Protein Expression | https://archive.ics.uci.edu/dataset/342 |
| Urban Land Cover | https://archive.ics.uci.edu/dataset/295 |
| Pen-Based Digit Recognition | https://archive.ics.uci.edu/dataset/81 |
| Soybean (Large) | https://archive.ics.uci.edu/dataset/90 |

## Key files

```
core.py                               Tree enumeration, embedding, models, artifact building
generate_data.py                      Synthetic data generation for the simulation
run_datasets.py                       Score all trees on each real dataset; write xlsx results
run_simulations.py                    Bias-variance simulation (numba-accelerated)
reproduce.py                          Single-command entry point (--paper for fast path)

nd/                                   ND algorithm implementations
  NestedDichotomy.py                  Core tree data structure and training
  RandomGeneration.py                 Uniform random tree sampling
  BBoK.py                             Balanced best-of-K generator
  BestOfK.py                          Best-of-K wrapper (K=10)
  ACND.py, NDC.py, RPND.py, CBND.py  Four further construction heuristics

scripts/make_artifacts.py             Build MDS embeddings from configs/artifacts.yml
scripts/verify_artifacts.py           Check built artifacts against configs/artifacts.yml
scripts/make_dataset_plots.py                 Embedding PDF + HTML maps per real dataset
scripts/make_simulation_plots.py      Simulation bias/variance/depth PDF maps
scripts/moran_simulation_analysis.py  Spatial statistics on simulation results
scripts/make_dataset_tables.py                LaTeX real-data tables (spatial stats + heuristic placement)
scripts/make_simulation_tables.py     LaTeX simulation tables
scripts/make_ablation_summaries.py    Moran analysis for tokeniser/BoW/RF ablations
scripts/make_ablation_tables.py       LaTeX ablation tables
scripts/make_paper_tables.py          Orchestrates all tables → paper_outputs/tables/
scripts/download_zenodo.py            Download pre-built data from Zenodo
scripts/csv_to_configs.py             Convert zenodo/simulation_configs.csv to JSON

configs/artifacts.yml                 Manifest of all artifacts to build
cache_nd/                             Cached .joblib artifacts (auto-created by step 1)
zenodo/                               Pre-built deposit data (see Zenodo section below)
real-data-bias-analysis/              Scoring results (auto-created by step 2)
bias-analysis/raw_simulation_results/ Simulation bootstrap CSVs (auto-created by step 3)
```

## Zenodo

Pre-built data is archived on Zenodo (doi:10.5281/zenodo.20456943). The deposit maps to the local `zenodo/` directory:

| Zenodo path | Contents | Download group |
|-------------|----------|----------------|
| `*.joblib` | Embedding artifacts for all class counts + ablations | `artifacts` |
| `simulation_scores/` | LR Moran summary CSVs (one per experiment) | `simulation_scores` |
| `simulation_scores_dt/` | DT Moran summary CSVs (one per experiment) | `simulation_scores_dt` |
| `tree_metrics/` | Per-tree scores for all 6 real datasets | `tree_metrics` |
| `raw_simulation_results/` | LR bootstrap CSVs (415 files, ~1 GB) | `boot` |
| `raw_simulation_results_dt/` | DT bootstrap CSVs (150 files) | `boot_dt` |
| `data/` | Raw UCI dataset CSVs | `rawdata` |
| `simulation_configs.csv` | 415 LR-generator simulation configurations | `sims` |
| `simulation_configs_dt.csv` | Decision-tree generator simulation configurations | `sims_dt` |

Download everything (all groups included by default):

```bash
python scripts/download_zenodo.py
```

Download specific groups:

```bash
python scripts/download_zenodo.py --what artifacts
python scripts/download_zenodo.py --what rawdata
python scripts/download_zenodo.py --what tree_metrics
python scripts/download_zenodo.py --what simulation_scores
python scripts/download_zenodo.py --what simulation_scores_dt
python scripts/download_zenodo.py --what sims
python scripts/download_zenodo.py --what sims_dt
python scripts/download_zenodo.py --what boot
python scripts/download_zenodo.py --what boot_dt
```

See `zenodo/manifest.csv` for a full listing of every file and what uses it.

## Simulation input

Steps 3 and 4 read from `zenodo/simulation_configs.csv` (LR-generator configs) and, if present, `zenodo/simulation_configs_dt.csv` (decision-tree generator configs). Both are processed automatically:

```bash
python reproduce.py --steps sim moran              # runs LR and DT sims if configs present
python reproduce.py --steps sim --classes 6 7      # 6- and 7-class only
```

LR boot results land in `bias-analysis/raw_simulation_results/` and Moran scores in `zenodo/simulation_scores/`.
DT boot results land in `bias-analysis/raw_simulation_results_dt/` and Moran scores in `zenodo/simulation_scores_dt/`.

To convert the LR CSV to JSON:

```bash
python scripts/csv_to_configs.py --output simulation_configs.json
```

`zenodo/simulation_configs_dt.csv` is committed as a static artifact; it was generated once from
the original Postgres simulation DB and is not regenerated as part of this pipeline.

## Using your own data

Drop a CSV in `zenodo/data/` with a `y` column (integer or string class labels) and any number of numeric feature columns. A `split` column (`"train"` / `"test"`) is used if present; otherwise a stratified 70/30 split is applied.

```bash
python run_datasets.py --datasets my_dataset
```

Output goes to `real-data-bias-analysis/my_dataset/v_my_dataset_lr_cluster_tables.xlsx`.

## Artifacts (cache_nd/)

The `cache_nd/` directory holds `.joblib` artifacts for C=4 through C=19. All artifacts are tracked via Git LFS (`.gitattributes`) and are also available from Zenodo:

```bash
python scripts/download_zenodo.py --what artifacts
```

To rebuild from scratch (re-runs MDS, requires pinned package versions):

```bash
python scripts/make_artifacts.py --all
```

Artifact specs (class count, tree count, seed) are declared in `configs/artifacts.yml`.
Use `scripts/verify_artifacts.py` to check a built artifact against its spec.
