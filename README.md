# ndscape: Nested Dichotomy Tree-Space Embedding

[![PyPI](https://img.shields.io/pypi/v/ndscape.svg)](https://pypi.org/project/ndscape/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20456943.svg)](https://doi.org/10.5281/zenodo.20456943)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A classification model assigns an observation to one of several classes, and when there are more than two, the task becomes a multiclass problem. A nested dichotomy (ND) gives that multiclass problem a visible structure: it arranges the classes in a binary tree and fits a classifier at each split, so the final prediction is built from a sequence of simpler binary decisions.

Any binary classifier can be used at the nodes. With probabilistic classifiers, the probability of each class is the product of the branch probabilities along its path from root to leaf. And because the tree records which classes are grouped together, where they separate, and in what order, you can inspect the fitted model and compare its structure with what you already know about the problem.

That structure also matters for prediction. There are (2M−3)!! possible NDs for M classes, each defining a different decomposition of the same classification task, and changing the tree can produce very different fitted models even when the data and binary classifier stay the same. This gives NDs a large, interpretable model space where changes in predictive behaviour can be related directly to changes in structure.

`ndscape` embeds ND tree structures using depth-weighted structural tokens and metric MDS, so structurally similar trees sit close together. It can be used as a Python package, and this repository also contains the code needed to reproduce the [paper](https://arxiv.org/abs/2610.18050) outputs.

Interactive maps from the [paper](https://arxiv.org/abs/2610.18050): [res-lucid.github.io/ndscape-plots](https://res-lucid.github.io/ndscape-plots/)

## Installation

For package use:

```bash
pip install ndscape
```

To reproduce the paper outputs, clone the repository and install the pinned requirements:
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Versions of `numpy`, `scipy`, and `scikit-learn` are pinned in `requirements.txt`. The MDS coordinates in `cache_nd/` were built with those exact versions, so changing numpy, scipy, or scikit-learn can produce slightly different coordinates.

## How to use this repository

| Goal | Command |
|------|---------|
| Use `ndscape` as a package | `pip install ndscape` |
| Test the repository install | `pytest tests/` |
| Recreate paper figures and tables from archived outputs | `python scripts/download_zenodo.py` then `python reproduce.py --paper` |
| Run the full reproduction pipeline | `python reproduce.py` |
| Run only selected steps | `python reproduce.py --steps artifacts` |
| Score a new dataset | `python run_datasets.py --datasets my_dataset` |

## Reproduce paper outputs from archived results

Download the Zenodo deposit, then regenerate the paper figures and tables without re-running the simulations:

```bash
python scripts/download_zenodo.py
python reproduce.py --paper
```

<details>
<summary>Scripts run by <code>python reproduce.py --paper</code></summary>

1. `scripts/make_dataset_plots.py --cache zenodo --out paper_outputs/figures`
   - PDF and interactive HTML embedding maps for all six real datasets.
2. `scripts/make_simulation_plots.py --out paper_outputs/figures`
   - Simulation bias, variance, and depth maps.
3. `scripts/make_ablation_summaries.py`
   - Moran spatial analysis for the tokeniser, BoW, and RF ablations.
4. `scripts/make_paper_tables.py --all --out paper_outputs/tables`
   - LaTeX tables.

Output is saved to `paper_outputs/figures/` and `paper_outputs/tables/`.
</details>

## Full reproduction pipeline

This rebuilds the derived outputs locally, using the archived data and configuration files downloaded into `zenodo/`.

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

## Using your own data

Drop a CSV in `zenodo/data/` with a `y` column (integer or string class labels) and any number of numeric feature columns. A `split` column (`"train"` / `"test"`) is used if present, otherwise a stratified 70/30 split is applied.

```bash
python run_datasets.py --datasets my_dataset
```

Output goes to `real-data-bias-analysis/my_dataset/v_my_dataset_lr_cluster_tables.xlsx`.

## Tests

```bash
pytest tests/                  # unit + snapshot tests (fast)
pytest tests/ -m integration   # integration tests (slow and requires pre-built C=6,7 artifacts)
```

The `tests/fixtures/` directory holds golden-master fixtures: MD5 hashes for the larger artifacts, and full tree lists and generator outputs for the smaller ones.

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

```text
reproduce.py                          Main reproduction entry point
core.py                               Tree enumeration, embedding, models, artifact building
run_datasets.py                       Score all trees on real datasets
run_simulations.py                    Bias-variance simulation
nd/                                   Nested dichotomy implementations
scripts/                              Figure, table, artifact, download, and analysis scripts
configs/artifacts.yml                 Artifact specifications
cache_nd/                             Cached embedding artifacts
zenodo/                               Downloaded Zenodo archive
real-data-bias-analysis/              Real-data scoring outputs
bias-analysis/raw_simulation_results/ Simulation bootstrap outputs
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

`zenodo/simulation_configs_dt.csv` is committed as a static artifact and is not regenerated as part of this pipeline.

## Artifacts (cache_nd/)

The `cache_nd/` directory holds `.joblib` artifacts for C=4 through C=19. It is not tracked in git — the artifacts are downloaded from Zenodo into `zenodo/` (tracked with Git LFS, see `.gitattributes`) or rebuilt locally:

```bash
python scripts/download_zenodo.py --what artifacts
```

To rebuild from scratch (re-runs MDS, requires pinned package versions):

```bash
python scripts/make_artifacts.py --all
```

Artifact specs (class count, tree count, seed) are declared in `configs/artifacts.yml`.
Use `scripts/verify_artifacts.py` to check a built artifact against its spec.

## Citation

Dix-Matthews, M., Hodkiewicz, M., & Polpo, A. (2026). ndscape: a structural map of the nested dichotomy space. arXiv:2610.18050 [stat.ME]. https://doi.org/10.48550/arXiv.2610.18050

Metadata is in [`CITATION.cff`](CITATION.cff); GitHub renders it under "Cite this repository". The archived data and artifacts have their own DOI, [10.5281/zenodo.20456943](https://doi.org/10.5281/zenodo.20456943).
