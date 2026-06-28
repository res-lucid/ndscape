"""Print LaTeX ablation tables comparing tokeniser, BoW, and RF representations."""
import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "paper_outputs" / "ablation_summaries"
SETTING = "1000_features_sparse_sparsity"

METHODS = ["tokeniser", "bow", "rf"]
NAMES = ["Tokeniser", "BoW", "RF"]

ROWS = [
    ("Moran $I$ (bias$^2$)", "bias2_moran_I"),
    ("Moran $I$ (var)", "var_moran_I"),
    ("Moran $I$ (SPL)", "mse_moran_I"),
]

APPENDIX_6 = [
    ("6_class_10_features_dense_sparsity", "10-feat (dense)"),
    ("6_class_1000_features_sparse_sparsity", "1000-feat (sparse)"),
    ("6_class_1000_features_dense_sparsity", "1000-feat (dense)"),
]

APPENDIX_7 = [
    ("7_class_10_features_dense_sparsity", "10-feat (dense)"),
    ("7_class_1000_features_sparse_sparsity", "1000-feat (sparse)"),
    ("7_class_1000_features_dense_sparsity", "1000-feat (dense)"),
]


def read(base, method, experiment):
    df = pd.read_csv(base / method / f"{experiment}_simulation_scores.csv")
    return df[df.status.eq("ok")]


def mean_sd(df, col):
    x = df[col].dropna().astype(float)
    return x.mean(), x.std(ddof=1)


def fmt(m, s, bold=False):
    txt = rf"{m:.3f} $\pm$ {s:.3f}"
    return rf"\textbf{{{txt}}}" if bold else txt


def print_rows(base, experiment):
    dfs = [read(base, method, experiment) for method in METHODS]

    for label, col in ROWS:
        vals = [mean_sd(df, col) for df in dfs]
        best = max(m for m, _ in vals)
        cells = [fmt(m, s, m == best) for m, s in vals]
        print(" & ".join([label, *cells]) + r" \\")


def table_main(base):
    print(r"""\begin{table}[h]
\centering
\caption{Ablation across three structural representations (tokeniser, BoW, Robinson--Foulds) in the 1000-feature sparse simulation. Values are mean $\pm$ SD over 50 generators. SPL denotes squared probability loss. Bold marks the row maximum. See Section~\ref{sec:spatial_summaries} for Moran's $I$ and AP@10 definitions.}
\label{tab:ablation-1000-sparse}
\small
\renewcommand{\arraystretch}{1.28}
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lccc@{}}
\toprule
Measure & Tokeniser & BoW & RF \\
\midrule""")

    for C in [6, 7]:
        print(rf"\multicolumn{{4}}{{l}}{{\textit{{{C} classes}}}} \\")
        print_rows(base, f"{C}_class_{SETTING}")
        if C == 6:
            print(r"\midrule")

    print(r"""\bottomrule
\end{tabular*}
\end{table}""")


def _appendix_table(base, experiments, caption, label):
    print(rf"""\begin{{table}}[t]
\centering
\caption{{{caption}}}
\label{{{label}}}
\small
\renewcommand{{\arraystretch}}{{1.28}}
\begin{{tabular*}}{{\textwidth}}{{@{{\extracolsep{{\fill}}}}lccc@{{}}}}
\toprule
Measure & Tokeniser & BoW & RF \\
\midrule""")

    for i, (experiment, label_) in enumerate(experiments):
        if i:
            print(r"\midrule")
        print(rf"\multicolumn{{4}}{{l}}{{\textit{{{label_}}}}} \\")
        print_rows(base, experiment)

    print(r"""\bottomrule
\end{tabular*}
\end{table}""")


def table_appendix(base):
    _appendix_table(
        base, APPENDIX_6,
        caption=(r"Ablation across feature settings at 6 classes. Mean $\pm$ SD over 50 "
                 r"generators. Bold marks the row maximum. The 1000-feature sparse row "
                 r"reproduces Table~\ref{tab:ablation-1000-sparse}."),
        label="tab:ablation-appendix-6",
    )
    print()
    _appendix_table(
        base, APPENDIX_7,
        caption=(r"Ablation across feature settings at 7 classes. Mean $\pm$ SD over 50 "
                 r"generators. Bold marks the row maximum. The 1000-feature sparse row "
                 r"reproduces Table~\ref{tab:ablation-1000-sparse}."),
        label="tab:ablation-appendix-7",
    )


p = argparse.ArgumentParser()
p.add_argument("--table", choices=["main", "appendix"], default="main")
p.add_argument("--summary-dir", default=BASE)
args = p.parse_args()

base = Path(args.summary_dir)

if args.table == "main":
    table_main(base)
else:
    table_appendix(base)
