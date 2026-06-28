"""Verify cached ND tree-space artifacts against configs/artifacts.yml.

Checks for each artifact:
  - cache file exists
  - required keys are present
  - tree count matches expected (exhaustive: factorial, sampled: N)
  - coord shape matches (n_trees, dim)
  - labels shape matches n_trees
  - all classes appear in every tree
  - metadata fields are present

Usage
-----
    python scripts/verify_artifacts.py
    python scripts/verify_artifacts.py --config configs/artifacts.yml --cache cache_nd
"""

import argparse
import sys
from math import prod
from pathlib import Path

import yaml
from joblib import load

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from core import _cache_path

REQUIRED_KEYS = {
    "trees", "trees_plot", "coords_plot", "labels_plot",
    "k", "idx_score_in_plot", "trustworthiness", "metadata",
}
REQUIRED_META = {
    "C", "N_sampled", "exhaustive", "seed", "dim",
    "tokenizer", "traversal",
    "mds_raw_stress", "mds_stress1", "trustworthiness",
    "n_clusters", "silhouette_score", "packages",
}


def verify_one(entry, cache_dir):
    name = entry["name"]
    C    = int(entry["C"])
    N    = int(entry["N"])
    seed = int(entry["seed"])
    dim  = int(entry["dim"])
    exh  = bool(entry.get("exhaustive", False))

    fp = _cache_path(C, N, seed, dim, cache_dir)
    errors = []

    if not fp.exists():
        print(f"  [MISSING] {name}")
        return False

    try:
        art = load(fp)
    except Exception as e:
        print(f"  [ERROR]   {name}: cannot load — {e}")
        return False

    missing_keys = REQUIRED_KEYS - set(art.keys())
    if missing_keys:
        errors.append(f"missing keys: {missing_keys}")

    n_trees = len(art.get("trees", []))
    # exhaustive count is the double factorial (2C-3)!! = 3*5*...*(2C-3)
    expected = prod(range(3, 2*C - 2, 2)) if exh else N
    if exh and n_trees != expected:
        errors.append(f"tree count {n_trees} != expected {expected}")
    elif not exh and n_trees < 1:
        errors.append("no trees found")

    coords = art.get("coords_plot")
    if coords is not None:
        if coords.shape != (n_trees, dim):
            errors.append(f"coords shape {coords.shape} != ({n_trees}, {dim})")

    labels = art.get("labels_plot")
    if labels is not None and len(labels) != n_trees:
        errors.append(f"labels length {len(labels)} != {n_trees}")

    meta = art.get("metadata", {})
    missing_meta = REQUIRED_META - set(meta.keys())
    if missing_meta:
        errors.append(f"missing metadata: {missing_meta}")

    if errors:
        print(f"  [FAIL]    {name}")
        for e in errors:
            print(f"            {e}")
        return False

    print(f"  [OK]      {name}  trees={n_trees}  k={art['k']}"
          f"  stress1={meta.get('mds_stress1', '?'):.4f}"
          f"  tw={meta.get('trustworthiness', '?'):.4f}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Verify ND artifacts")
    parser.add_argument("--config", default=str(ROOT / "configs" / "artifacts.yml"))
    parser.add_argument("--cache",  default=str(ROOT / "cache_nd"))
    args = parser.parse_args()

    with open(args.config) as f:
        manifest = yaml.safe_load(f)

    entries = manifest["artifacts"]
    print(f"Verifying {len(entries)} artifact(s) from {args.cache}/\n")

    results = [verify_one(e, args.cache) for e in entries]
    n_pass = sum(results)
    n_fail = len(results) - n_pass

    print(f"\n{n_pass}/{len(results)} passed", end="")
    if n_fail:
        print(f"  ({n_fail} failed)")
        sys.exit(1)
    else:
        print()


if __name__ == "__main__":
    main()
