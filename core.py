"""Core nested-dichotomy methods: tree enumeration, embedding, models, artifact building."""

import ast
import importlib.metadata
import os
from itertools import combinations
from math import comb, sqrt
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import MDS, trustworthiness
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import config
from esda import Moran
from libpysal.weights import KNN
import warnings

warnings.filterwarnings("ignore", module=r"sklearn\..*")

# Tree sampling (proportional balanced design)

def sample_trees(n, N):
    """Sample up to N nested-dichotomy trees over n classes, proportionally balanced."""
    tree_counts = [0, 1, 1] + [0] * max(0, n - 2)
    for m in range(3, n + 1):
        tree_counts[m] = (2 * m - 3) * tree_counts[m - 1]

    def _draw(classes, budget):
        m = len(classes)
        if m <= 1:
            return [()]
        if m == 2:
            return [((tuple(classes[:1]), tuple(classes[1:])),)]
        if budget == 1:
            k = m // 2
            right, left = tuple(classes[k:]), tuple(classes[:k])
            return [((right, left),) + _draw(list(right), 1)[0] + _draw(list(left), 1)[0]]
        left_sizes = range(1, m // 2 + 1)
        weights = [comb(m, k) // (2 if 2 * k == m else 1) * tree_counts[k] * tree_counts[m - k] for k in left_sizes]
        total = sum(weights)
        alloc = [budget * w // total for w in weights]
        for i in sorted(range(len(weights)), key=lambda i: -(budget * weights[i] % total))[:budget - sum(alloc)]:
            alloc[i] += 1
        out = []
        for k, k_budget in zip(left_sizes, alloc):
            if k_budget == 0:
                continue
            n_combos = comb(m, k) // (2 if 2 * k == m else 1)
            per_combo, remainder = divmod(k_budget, n_combos)
            seen = 0
            for left in combinations(classes, k):
                if 2 * k == m and classes[0] not in left:
                    continue
                combo_budget = per_combo + (seen < remainder)
                seen += 1
                if combo_budget == 0:
                    if per_combo == 0 and seen >= remainder:
                        break
                    continue
                left = tuple(left)
                right = tuple(x for x in classes if x not in left)
                left_budget = int(sqrt(combo_budget * tree_counts[k] / tree_counts[m - k]) + 0.5)
                lb_round_fixed = max(left_budget, (combo_budget + tree_counts[m - k] - 1) // tree_counts[m - k])
                lb_round_fixed = min(lb_round_fixed, tree_counts[k])
                right_budget = (combo_budget + lb_round_fixed - 1) // lb_round_fixed
                drawn = 0
                for right_sub in _draw(list(right), right_budget):
                    for left_sub in _draw(list(left), lb_round_fixed):
                        out.append(((right, left),) + right_sub + left_sub)
                        drawn += 1
                        if drawn == combo_budget:
                            break
                    if drawn == combo_budget:
                        break
        return out

    return _draw(list(range(n)), min(N, tree_counts[n]))

# Tree enumeration

def all_trees(n: int):
    """All ND trees over labels 0..n-1."""
    return list(_gen(tuple(range(n))))

def _gen(labels):
    labels = tuple(sorted(labels))
    if len(labels) <= 1:
        yield ()
        return
    s = labels[0]
    for r in range(1, len(labels)):
        for rest in combinations(labels[1:], r - 1):
            left  = tuple(sorted((s,) + rest))
            right = tuple(x for x in labels if x not in left)
            pair  = tuple(sorted((left, right), key=lambda t: (-len(t), t)))
            L = ((),) if len(left)  == 1 else tuple(_gen(left))
            R = ((),) if len(right) == 1 else tuple(_gen(right))
            for lt in L:
                for rt in R:
                    yield (pair,) + lt + rt

# Tree embedding

def split_analyzer(n_classes):
    """Return a TF-IDF token-analyser function for ND trees."""
    def _analyze(doc):
        splits = ast.literal_eval(doc) if isinstance(doc, str) else doc
        splits = sorted(splits, key=lambda lr: -len(set(lr[0]) | set(lr[1])))

        max_depth   = min(8, n_classes - 2)
        small_group = 3
        all_classes = frozenset(x for L, R in splits for x in tuple(L) + tuple(R))
        depth       = {all_classes: 0}
        extra_depth_w = max(0, n_classes - 6)
        toks = []

        for L, R in splits:
            U = frozenset(L) | frozenset(R)
            d = depth[U]
            if len(L) > 1: depth[frozenset(L)] = d + 1
            if len(R) > 1: depth[frozenset(R)] = d + 1
            if d > max_depth:
                continue

            L, R = tuple(sorted(L)), tuple(sorted(R))
            A, B = sorted([L, R], key=lambda x: (-len(x), x))
            dbin = min(d, 5)
            w    = max(1, max_depth + 1 - d)

            toks.extend([f"sz:{len(A)}|{len(B)}"] * max(1, w // 2))
            if d == 0:
                toks.extend([f"root:{A}|{B}"] * 2)
                toks.extend([f"root_sz:{len(A)}|{len(B)}"] * 4)

            for i in A:
                for j in B:
                    a, b = sorted((i, j))
                    toks.extend([f"sep:{a}-{b}:sz:{len(A)}|{len(B)}"] * max(1, w // 2))
                    toks.append(f"sep:{a}-{b}:d{dbin}")
                    if d == 0:
                        toks.extend([f"root_sep:{a}-{b}"] * 2)
                        toks.extend([f"root_sep:{a}-{b}:sz:{len(A)}|{len(B)}"] * 2)

            for side in (A, B):
                for i, j in combinations(side, 2):
                    toks.append(f"same:{i}-{j}")
                if n_classes >= 7 and 1 < len(side) <= small_group:
                    toks.extend([f"small_group_size:{len(side)}:d{dbin}"] * extra_depth_w)

        return toks
    return _analyze

# Models

def get_model(base):
    if base == "lr":
        return LogisticRegression(
            penalty="l2", solver="newton-cholesky",
            C=getattr(config, "C", 0.1), max_iter=2000,
        )
    if base == "lda":
        return LinearDiscriminantAnalysis(solver="svd")
    if base == "svm":
        return SVC(kernel="rbf", C=1.0, gamma="scale", probability=True)
    if base == "decisiontree":
        return DecisionTreeClassifier(criterion="entropy", min_samples_leaf=5, random_state=0)
    raise ValueError(f"Unknown base model: '{base}'")

def get_or_train_split_model(left, right, base="lr"):
    key = (base, tuple(left), tuple(right))
    if key in config.model_cache:
        return config.model_cache[key]
    mask   = np.isin(config.y, tuple(left) + tuple(right))
    y_node = np.where(np.isin(config.y[mask], left), 0, 1)
    if len(np.unique(y_node)) < 2:
        config.model_cache[key] = None
        return None
    model = get_model(base)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        model.fit(config.X[mask], y_node)
    config.model_cache[key] = model
    return model

# Prediction

def nd_predict_proba(X, T, classes, base="lr"):
    X     = np.asarray(X)
    idx   = {c: i for i, c in enumerate(classes)}
    proba = np.ones((X.shape[0], len(classes)), dtype=float)

    splits = [
        (get_or_train_split_model(L, R, base=base),
         np.array([idx[c] for c in L]),
         np.array([idx[c] for c in R]))
        for L, R in T
    ]
    for model, li, ri in splits:
        if model is None:
            proba[:, ri] = 0.0
        else:
            p = model.predict_proba(X)   # (n, 2)
            proba[:, li] *= p[:, 0:1]
            proba[:, ri] *= p[:, 1:2]

    row_sums = proba.sum(axis=1, keepdims=True)
    return proba / np.where(row_sums == 0, 1.0, row_sums), classes


# Artifact building

_PINNED = {"numpy": "2.2.6", "scikit-learn": "1.7.2", "scipy": "1.15.3"}

def _check_versions():
    for pkg, pinned in _PINNED.items():
        try:
            installed = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            continue
        if installed != pinned:
            warnings.warn(
                f"{pkg}=={installed} installed, but artifacts were built with "
                f"{pkg}=={pinned}. MDS coords may differ — rebuild artifacts or "
                f"pin versions with: pip install -r requirements.txt",
                stacklevel=3,
            )

_EXHAUSTIVE_N = {6: 945, 7: 10_395}

def _cache_path(C, N, seed, dim, cache_dir):
    if C in _EXHAUSTIVE_N:
        N, seed, dim = _EXHAUSTIVE_N[C], 0, 2
    return Path(cache_dir) / f"art_1_C{C}_N{N}_s{seed}_d{dim}.joblib"

def build_artifact(cats, N=5000, seed=0, dim=2, cache_dir="cache_nd"):
    """Build a tree-space artifact from scratch (always regenerates)."""
    _check_versions()
    cats = list(cats)
    C    = len(cats)

    if C <= 7:
        # Call _gen directly so cats (already string labels) are used as-is;
        # all_trees() works on integer labels 0..n-1 and would require remapping.
        trees = list(_gen(tuple(cats)))
    else:
        m     = dict(enumerate(cats))
        trees = [
            tuple((tuple(m[i] for i in L), tuple(m[i] for i in R)) for L, R in t)
            for t in sample_trees(C, N)
        ]

    df = pd.DataFrame({"tree": trees})
    df["tree_encoded"] = df["tree"]
    vec   = TfidfVectorizer(analyzer=split_analyzer(C), use_idf=False, norm="l2")
    tfidf_vecs   = vec.fit_transform(df["tree_encoded"])
    sim   = cosine_similarity(tfidf_vecs)
    dist   = np.sqrt(np.maximum(0, 2.0 * (1.0 - sim)))
    np.fill_diagonal(dist, 0.0)

    mds    = MDS(n_components=dim, dissimilarity="precomputed",
                 random_state=seed, n_init=1, max_iter=1000, eps=1e-6, n_jobs=-1)
    coords = mds.fit_transform(dist)
    print("MDS iters:", mds.n_iter_)
    print("raw stress:", mds.stress_)

    upper_tri = np.triu_indices_from(dist, 1)
    stress1 = float(np.sqrt(mds.stress_ / (dist[upper_tri]**2).sum()))
    n_neighbors = max(1, min(20, len(trees) // 2 - 1))
    trust   = float(trustworthiness(dist, coords, n_neighbors=n_neighbors, metric="precomputed"))

    k_min = 14 if C >= 11 else 4
    k_max = min(100 if C >= 11 else 20, len(trees) // 2)
    best_score, best_labels, n_clusters = -1, None, k_min
    for k in range(k_min, k_max):
        lab   = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(coords)
        score = silhouette_score(coords, lab)
        if score > best_score:
            best_score, best_labels, n_clusters = score, lab, k

    metadata = dict(
        C=C, N_sampled=len(trees), exhaustive=C <= 7, seed=seed, dim=dim,
        tokenizer="split_analyzer", traversal="bfs",
        mds_raw_stress=float(mds.stress_), mds_stress1=stress1,
        trustworthiness=trust, n_clusters=n_clusters,
        silhouette_score=float(best_score),
        packages={n: importlib.metadata.version(n) for n in ("numpy", "scikit-learn", "scipy", "joblib")},
    )

    return dict(
        trees=trees, trees_plot=list(map(str, trees)),
        coords_plot=coords, labels_plot=best_labels.astype(int),
        k=n_clusters, idx_score_in_plot=np.arange(len(trees), dtype=int),
        trustworthiness=trust, metadata=metadata,
    )

def get_trees_and_artifact(cats, N=5000, seed=0, dim=2, cache_dir="cache_nd"):
    """Load from cache if present, otherwise build and cache."""
    cats = list(cats)
    C    = len(cats)
    fp   = _cache_path(C, N, seed, dim, cache_dir)
    if fp.exists():
        return load(fp)
    art = build_artifact(cats, N=N, seed=seed, dim=dim, cache_dir=cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    dump(art, fp)
    return art


# Spatial statistics

def build_knn_weights(coords, k=50):
    """KNN spatial weights matrix for Moran's I, symmetrised and row-standardised."""
    k = min(k, len(coords) - 1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        w = KNN.from_array(coords, k=k)
    w.transform = "r"
    return w

def spatial_stats(values, coords, k=50, permutations=999):
    """Global Moran's I for an array of tree-level values on a 2-D embedding.

    Returns (I, p_sim) using a KNN spatial weights graph.
    """
    w = build_knn_weights(coords, k=k)
    mi = Moran(values, w, permutations=permutations)
    return mi.I, mi.p_sim
