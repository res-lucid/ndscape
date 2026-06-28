# Paper artifact guide

Maps each table and figure in the paper to the script that produces it and the output it writes.

---

## Fast path — reproduce paper outputs from pre-built data

If you have downloaded the Zenodo deposit, run:

```bash
python reproduce.py --paper
```

This does three things in order:
1. `scripts/make_dataset_plots.py --cache zenodo --out paper_outputs/figures` — PDF + interactive HTML embedding maps for all 6 datasets, plus `nd_info.json` per dataset
2. `scripts/make_ablation_summaries.py` — Moran analysis for the tokeniser / BoW / RF ablations → `paper_outputs/ablation_summaries/` (cleaned up after step 3)
3. `scripts/make_paper_tables.py --all --out paper_outputs/tables` — all LaTeX tables

Output: `paper_outputs/figures/` and `paper_outputs/tables/`.

---

## Full reproduction (all steps)

```bash
python reproduce.py --steps artifacts score sim moran tables plots
```

---

## Step 1 — Build tree-space embeddings

**Script:** `python scripts/make_artifacts.py --all`
**Output:** `cache_nd/art_1_C{C}_N{N}_s0_d2.joblib`
**Verify:** `python scripts/verify_artifacts.py`

One artifact per class count. Each holds the full list of trees, 2-D MDS coordinates, and cluster labels.

| Artifact | Classes | Trees | Notes |
|----------|---------|-------|-------|
| `art_1_C6_N945_s0_d2` | 6 | 945 | exhaustive |
| `art_1_C7_N10395_s0_d2` | 7 | 10 395 | exhaustive |
| `art_1_C8_N15000_s0_d2` | 8 | 15 000 | sampled |
| `art_1_C9_N15000_s0_d2` | 9 | 15 000 | sampled |
| `art_1_C10_N15000_s0_d2` | 10 | 15 000 | sampled |
| `art_1_C19_N15000_s0_d2` | 19 | 15 000 | sampled (Soybean) |

---

## Step 2 — Score real datasets

**Script:** `python run_datasets.py`
**Output:** `real-data-bias-analysis/<dataset>/v_<dataset>_lr_cluster_tables.xlsx`, sheet `tree_metrics`

Columns: `tree`, `score`, `error_rate`, `accuracy_var_01`, `accuracy_se`, `logloss`, `loglik`, `balanced_acc`, `macro_f1`, `model_var`.

---

## Step 3 — Bias-variance simulation

**Script:** `python run_simulations.py --input zenodo/simulation_configs.csv`
**Output:** `bias-analysis/raw_simulation_results/boot_results_{simulation_id}_100_runs_.csv`

Columns: `simulation_id`, `experiment_name`, `n_samples`, `tree_idx`, `tree_text`, `bias2`, `var`, `mse`.

| Classes | Trees | Generators | Reps |
|---------|-------|------------|------|
| 6 | 945 | 50 | 100 |
| 7 | 10 395 | 50 | 100 |
| 8 | 15 000 | 5 | 100 |
| 9 | 15 000 | 5 | 100 |
| 10 | 15 000 | 5 | 100 |

---

## Step 4 — Simulation Moran analysis

**Script:** `python scripts/moran_simulation_analysis.py --selected zenodo/simulation_configs.csv`
**Output:** `zenodo/simulation_scores/{experiment}_simulation_scores.csv` (LR); `zenodo/simulation_scores_dt/{experiment}_simulation_scores.csv` (DT)

Computes Moran's I, Geary's C, bivariate Moran, and proximity statistics for each simulation.

---

## Step 5 — LaTeX tables

### Table 5 — Spatial statistics across the six real datasets

**Script:** `python scripts/make_dataset_tables.py --table 5`
**Inputs:** step 1 artifacts + step 2 xlsx files

Moran's I for accuracy, model variance, and log loss; trustworthiness at 50 neighbours.

### Table 3 — Spatial autocorrelation across class counts (simulation)

**Script:** `python scripts/make_simulation_tables.py --table moran-main --summary-dir zenodo/simulation_scores`
**Inputs:** step 4 summary CSVs

### Table 4 — Enrichment near the generating ND (simulation)

**Script:** `python scripts/make_simulation_tables.py --table enrichment --summary-dir zenodo/simulation_scores`

### Higher class count table

**Script:** `python scripts/make_simulation_tables.py --table high-class --summary-dir zenodo/simulation_scores`

All tables at once (uses zenodo/ simulation_scores as source):

**Script:** `python scripts/make_paper_tables.py --all --out paper_outputs/tables`

---

## Step 6 — Embedding plots

**Script:** `python scripts/make_dataset_plots.py --cache zenodo --out paper_outputs/figures`
**Output:** `paper_outputs/figures/<dataset>/plots/*.pdf`, `*_hover.html`, `nd_info.json`

Produces PDF embedding maps coloured by accuracy and model variance, interactive HTML hover plots, and `nd_info.json` (per-tree stats used by the heuristic table). Note: `make_dataset_tables.py --table heuristic` requires `nd_info.json` to exist, so plots must be generated before that table.

---

## Ablation summaries

**Script:** `python scripts/make_ablation_summaries.py`
**Inputs:** `zenodo/simulation_configs.csv`, `zenodo/raw_simulation_results/`, `zenodo/*.joblib`
**Output:** `paper_outputs/ablation_summaries/{tokeniser,bow,rf}/*_simulation_scores.csv`

Runs Moran analysis for the three distance-metric variants (TF-IDF tokeniser, bag-of-words, Random Forest). Called automatically by `--paper`; the output folder is cleaned up after `make_paper_tables.py` completes.

---

## Reproducibility notes

- MDS coordinates are sensitive to `numpy`, `scipy`, and `scikit-learn` versions. Pin them with `pip install -r requirements.txt`.
- `tests/fixtures/` holds golden-master hashes for tree lists, tokeniser output, and per-tree scores. Score hashes were seeded from the reference results. Coordinate hashes are expected to differ on fresh builds.
- `zenodo/simulation_configs.csv` is required for steps 3 and 4. Download it with `python scripts/download_zenodo.py --what sims`.
