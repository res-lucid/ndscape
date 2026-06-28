"""Build ND tree-space artifacts from configs/artifacts.yml. Run with --all or --target NAME."""

import argparse
import json
import sys
from pathlib import Path

import yaml
from joblib import dump

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from core import _cache_path, build_artifact  # noqa: E402


def _build_one(entry, cache_dir, force=False):
    name = entry["name"]
    C    = int(entry["C"])
    N    = int(entry["N"])
    seed = int(entry["seed"])
    dim  = int(entry["dim"])
    cats = list(range(C))

    fp = _cache_path(C, N, seed, dim, cache_dir)

    if fp.exists() and not force:
        print(f"  [skip]  {name}  (cache exists: {fp})")
        return

    print(f"  [build] {name}  C={C} N={N} seed={seed} dim={dim}")
    art = build_artifact(cats, N=N, seed=seed, dim=dim, cache_dir=cache_dir)
    fp.parent.mkdir(parents=True, exist_ok=True)
    dump(art, fp)

    meta = art["metadata"]
    print(f"          trees={meta['N_sampled']}  k={meta['n_clusters']}"
          f"  stress1={meta['mds_stress1']:.4f}"
          f"  trustworthiness={meta['trustworthiness']:.4f}"
          f"  silhouette={meta['silhouette_score']:.4f}")
    print(f"          saved -> {fp}")

    meta_fp = fp.with_suffix(".metadata.json")
    meta_fp.write_text(json.dumps(meta, indent=2))
    print(f"          metadata -> {meta_fp}")


def main():
    parser = argparse.ArgumentParser(description="Build ND tree-space artifacts")
    parser.add_argument("--all",    action="store_true", help="Build all artifacts")
    parser.add_argument("--target", metavar="NAME",      help="Build one artifact by name")
    parser.add_argument("--force",  action="store_true", help="Rebuild even if cache exists")
    parser.add_argument("--config", default=str(ROOT / "configs" / "artifacts.yml"),
                        help="Path to artifacts.yml")
    parser.add_argument("--cache",  default=str(ROOT / "cache_nd"),
                        help="Directory for .joblib cache files")
    args = parser.parse_args()

    if not args.all and not args.target:
        parser.error("Specify --all or --target NAME")

    with open(args.config) as f:
        manifest = yaml.safe_load(f)

    entries = manifest["artifacts"]

    if args.target:
        matches = [e for e in entries if e["name"] == args.target]
        if not matches:
            names = [e["name"] for e in entries]
            sys.exit(f"Unknown target '{args.target}'. Available:\n  " + "\n  ".join(names))
        entries = matches

    print(f"Building {len(entries)} artifact(s) into {args.cache}/\n")
    for entry in entries:
        _build_one(entry, cache_dir=args.cache, force=args.force)

    print("\nDone.")

if __name__ == "__main__":
    main()
