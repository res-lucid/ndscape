"""Convert zenodo/simulation_configs.csv to JSON (also handles the legacy 18-column format)."""

import argparse
import ast
import csv
import io
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent

FULL_COLS = [
    "simulation_id", "name", "seed", "n_classes", "n_features", "n_samples",
    "distribution_types", "generated_nd", "covariates", "class_freq",
    "nd_dot", "nd_params", "x_hash", "y_hash", "full_dataset_hash",
    "extra_condition", "experiment", "notes",
]

NEEDED_COLS = {
    "simulation_id", "seed", "n_classes", "n_features",
    "distribution_types", "generated_nd", "experiment",
}


def _clean_dist_types(raw):
    """Return strings like 'Col1: normal(0.5, 1.2)' from the stored CSV value."""
    s = str(raw).strip()
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]

    out = []
    for item in re.split(r"''\s*,\s*''", s):
        item = item.strip().strip("'")
        item = re.sub(r"np\.float64\(([^)]+)\)", r"\1", item)
        item = re.sub(r"np\.int64\(([^)]+)\)", r"\1", item)
        if item:
            out.append(item)
    return out


def _build_nd_params(generated_nd, covariates):
    splits = ast.literal_eval(str(generated_nd))
    coefs = ast.literal_eval(str(covariates))
    return {str(split): list(map(float, coef)) for split, coef in zip(splits, coefs)}


def _build_nd_params_dt(generated_nd, node_spec):
    """Build nd_params for a decision-tree generator.

    node_spec is a list of dicts, one per split, each with keys
    depth, feat, thr, leafp (as stored by new_generation.ipynb).
    """
    splits = ast.literal_eval(str(generated_nd))
    specs  = ast.literal_eval(str(node_spec))
    return {str(split): spec for split, spec in zip(splits, specs)}


def _parse_row(row_or_fields):
    """Convert either a dict row or an original 18-field row to a config dict."""
    if isinstance(row_or_fields, dict):
        row = row_or_fields
    else:
        row = dict(zip(FULL_COLS, row_or_fields))

    experiment = row.get("experiment") or row.get("name") or ""

    gen_type = str(row.get("generator_type") or "lr").strip()
    if gen_type == "decisiontree":
        nd_params = _build_nd_params_dt(row["generated_nd"], row["node_spec"])
    else:
        nd_params = _build_nd_params(row["generated_nd"], row["covariates"])
    return {
        "simulation_id": str(row["simulation_id"]).strip(),
        "experiment": str(experiment).strip(),
        "seed": int(row["seed"]),
        "n_classes": int(row["n_classes"]),
        "n_features": int(row["n_features"]),
        "generator_type": gen_type,
        "nd_params": nd_params,
        "distribution_types": _clean_dist_types(row["distribution_types"]),
    }


def _looks_like_trimmed_csv(path):
    with open(path, encoding="utf-8", newline="") as f:
        header = next(csv.reader(f))
    return NEEDED_COLS.issubset(set(header))


def iter_rows(path):
    """Yield rows from either the trimmed CSV or the original multiline all_sims.csv."""
    if _looks_like_trimmed_csv(path):
        with open(path, encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                yield row
        return

    with open(path, encoding="utf-8") as f:
        f.readline()
        buf = ""
        for line in f:
            if line and line[0].isdigit() and buf:
                try:
                    fields = next(csv.reader(io.StringIO(buf)))
                    if len(fields) == len(FULL_COLS):
                        yield fields
                except (csv.Error, StopIteration) as e:
                    print(f"  [warn] skipping malformed chunk: {e}", file=sys.stderr)
                buf = line
            else:
                buf += line

        if buf.strip():
            try:
                fields = next(csv.reader(io.StringIO(buf)))
                if len(fields) == len(FULL_COLS):
                    yield fields
            except (csv.Error, StopIteration) as e:
                print(f"  [warn] skipping malformed chunk: {e}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Convert all_sims CSV to simulation config JSON")
    parser.add_argument("--input", default=str(ROOT / "zenodo" / "simulation_configs.csv"))
    parser.add_argument("--output", default=str(ROOT / "simulation_configs.json"))
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    configs = []
    for i, row in enumerate(iter_rows(args.input)):
        if args.limit is not None and i >= args.limit:
            break
        try:
            configs.append(_parse_row(row))
        except Exception as e:
            print(f"  [warn] row {i}: {e}", file=sys.stderr)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(configs, f, indent=2)

    print(f"Wrote {len(configs)} configs to {args.output}")


if __name__ == "__main__":
    main()
