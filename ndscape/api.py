"""Fit, score, and embed nested-dichotomy (ND) trees on real datasets.

Self-contained: this module does not import core.py, config.py, or anything
else from the rest of the repo, so it can be packaged and installed on its
own via pip.

A tree is a list of (left, right) tuples of class labels, e.g.
    [((0, 1), (2, 3)), ((0,), (1,)), ((2,), (3,))]
read top-down: first split {0,1} vs {2,3}, then split each side further.
"""

from itertools import combinations
from math import comb, sqrt
from pathlib import Path

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import MDS
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.tree import DecisionTreeClassifier


# ---------------------------------------------------------------- models

def _get_model(base):
    """Resolve `base` to a fresh, unfitted binary classifier.

    `base` is either a string shorthand ("lr", "decisiontree") or an actual
    unfitted scikit-learn estimator, e.g. SVC(probability=True) or your own
    model implementing fit/predict_proba. A fresh clone is made per split.
    """
    if isinstance(base, str):
        if base == "lr":
            return LogisticRegression(
                penalty="l2", solver="newton-cholesky", C=0.1, max_iter=2000
            )
        if base in ("decisiontree", "dt"):
            return DecisionTreeClassifier(
                criterion="entropy", min_samples_leaf=5, random_state=0
            )
        raise ValueError(
            f"Unknown base model '{base}'. Pass 'lr', 'decisiontree', "
            "or an unfitted scikit-learn estimator."
        )
    return clone(base)


# ---------------------------------------------------------------- tree enumeration

def _gen(labels):
    labels = tuple(sorted(labels))
    if len(labels) <= 1:
        yield ()
        return
    s = labels[0]
    for r in range(1, len(labels)):
        for rest in combinations(labels[1:], r - 1):
            left = tuple(sorted((s,) + rest))
            right = tuple(x for x in labels if x not in left)
            pair = tuple(sorted((left, right), key=lambda t: (-len(t), t)))
            L = ((),) if len(left) == 1 else tuple(_gen(left))
            R = ((),) if len(right) == 1 else tuple(_gen(right))
            for lt in L:
                for rt in R:
                    yield (pair,) + lt + rt


def all_trees(classes):
    """Enumerate every ND tree over `classes` (exhaustive — use sample_trees for many classes)."""
    return list(_gen(tuple(classes)))


def _sample_tree_indices(n, N):
    """Sample up to N ND trees over positions 0..n-1, proportionally balanced."""
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
        weights = [
            comb(m, k) // (2 if 2 * k == m else 1) * tree_counts[k] * tree_counts[m - k]
            for k in left_sizes
        ]
        total = sum(weights)
        alloc = [budget * w // total for w in weights]
        for i in sorted(range(len(weights)), key=lambda i: -(budget * weights[i] % total))[
            : budget - sum(alloc)
        ]:
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
                left_budget = max(left_budget, (combo_budget + tree_counts[m - k] - 1) // tree_counts[m - k])
                left_budget = min(left_budget, tree_counts[k])
                right_budget = (combo_budget + left_budget - 1) // left_budget
                drawn = 0
                for right_sub in _draw(list(right), right_budget):
                    for left_sub in _draw(list(left), left_budget):
                        out.append(((right, left),) + right_sub + left_sub)
                        drawn += 1
                        if drawn == combo_budget:
                            break
                    if drawn == combo_budget:
                        break
        return out

    return _draw(list(range(n)), min(N, tree_counts[n]))


def _relabel_tree(tree, classes):
    m = dict(enumerate(classes))
    return [(tuple(m[i] for i in left), tuple(m[i] for i in right)) for left, right in tree]


def sample_trees(classes, N):
    """Sample up to N ND trees over `classes`, proportionally balanced by subtree size."""
    classes = list(classes)
    raw = _sample_tree_indices(len(classes), N)
    return [_relabel_tree(t, classes) for t in raw]


# ---------------------------------------------------------------- ND

class ND:
    """A single nested-dichotomy tree fitted to a dataset.

    Parameters
    ----------
    tree : list of (left, right) tuples of class labels.
    classes : the full set of class labels, in the order predict_proba columns follow.
    models : optional, already-fitted binary classifiers — skips .fit(). Either
        a list of fitted models in the same order as `tree`, or a dict mapping
        (left, right) -> fitted model.
    """

    def __init__(self, tree, classes, models=None):
        self.tree = [(tuple(left), tuple(right)) for left, right in tree]
        self.classes = list(classes)
        if models is None:
            self.models = {}
        elif isinstance(models, dict):
            self.models = dict(models)
        else:
            self.models = dict(zip(self.tree, models))

    @classmethod
    def from_trained(cls, tree, classes, models):
        """Wrap a tree whose per-split binary models are already fitted.

        `models` is a list of fitted models in the same order as `tree`,
        or a dict mapping (left, right) -> fitted model.
        """
        return cls(tree, classes, models=models)

    def fit(self, X, y, base="lr"):
        """Fit one binary classifier per split on (X, y). Returns self.

        `base` is "lr", "decisiontree", or any unfitted scikit-learn
        estimator, e.g. base=SVC(probability=True, kernel="linear").
        """
        X, y = np.asarray(X), np.asarray(y)
        for left, right in self.tree:
            mask = np.isin(y, left + right)
            y_node = np.where(np.isin(y[mask], left), 0, 1)
            if len(np.unique(y_node)) < 2:
                self.models[(left, right)] = None
                continue
            model = _get_model(base)
            model.fit(X[mask], y_node)
            self.models[(left, right)] = model
        return self

    def predict_proba(self, X):
        """Class-probability matrix, columns ordered as `self.classes`."""
        X = np.asarray(X)
        idx = {c: i for i, c in enumerate(self.classes)}
        proba = np.ones((X.shape[0], len(self.classes)))
        for left, right in self.tree:
            model = self.models.get((left, right))
            li = [idx[c] for c in left]
            ri = [idx[c] for c in right]
            if model is None:
                proba[:, ri] = 0.0
                continue
            p = model.predict_proba(X)
            proba[:, li] *= p[:, 0:1]
            proba[:, ri] *= p[:, 1:2]
        row_sums = proba.sum(axis=1, keepdims=True)
        return proba / np.where(row_sums == 0, 1.0, row_sums)

    def predict(self, X):
        """Predicted class label per row."""
        proba = self.predict_proba(X)
        return np.asarray(self.classes)[proba.argmax(axis=1)]

    def score(self, X, y):
        """Accuracy and mean log-loss of this ND on (X, y)."""
        y = np.asarray(y)
        proba = self.predict_proba(X)
        pred = np.asarray(self.classes)[proba.argmax(axis=1)]
        idx = {c: i for i, c in enumerate(self.classes)}
        y_idx = np.array([idx[v] for v in y])
        logloss = -np.log(np.clip(proba[np.arange(len(y)), y_idx], 1e-15, 1.0)).mean()
        return {"accuracy": float((pred == y).mean()), "logloss": float(logloss)}


def fit(X, y, classes, tree=None, base="lr"):
    """Fit a single ND on (X, y).

    If `tree` is omitted, one tree is sampled automatically. `base` is "lr",
    "decisiontree", or any unfitted scikit-learn estimator you want to use
    at each split, e.g. base=SVC(probability=True, kernel="linear").
    """
    classes = list(classes)
    if tree is None:
        tree = sample_trees(classes, 1)[0]
    return ND(tree, classes).fit(X, y, base=base)


# ---------------------------------------------------------------- embedding + spatial stats

def _split_analyzer(n_classes):
    """TF-IDF token-analyser function for ND trees (same scheme used to build the paper's artifacts)."""
    def _analyze(doc):
        splits = doc
        splits = sorted(splits, key=lambda lr: -len(set(lr[0]) | set(lr[1])))

        max_depth = min(8, n_classes - 2)
        small_group = 3
        all_classes = frozenset(x for L, R in splits for x in tuple(L) + tuple(R))
        depth = {all_classes: 0}
        extra_depth_w = max(0, n_classes - 6)
        toks = []

        for L, R in splits:
            U = frozenset(L) | frozenset(R)
            d = depth[U]
            if len(L) > 1:
                depth[frozenset(L)] = d + 1
            if len(R) > 1:
                depth[frozenset(R)] = d + 1
            if d > max_depth:
                continue

            L, R = tuple(sorted(L)), tuple(sorted(R))
            A, B = sorted([L, R], key=lambda x: (-len(x), x))
            dbin = min(d, 5)
            w = max(1, max_depth + 1 - d)

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


def embed_trees(classes, trees=None, N=2000, seed=0, dim=2, cache=None):
    """Place a set of ND trees in `dim`-D tree-space (TF-IDF + cosine distance + MDS).

    If `trees` is omitted, all trees are used for <=7 classes and a balanced
    sample of N otherwise. Returns (trees, coords); coords[i] is the
    embedding of trees[i].

    `cache` is an optional .joblib path. If it exists, (trees, coords) are
    loaded from it instead of recomputed (the MDS step is the slow part).
    Otherwise they're computed as usual and saved there for next time.
    """
    if cache is not None and Path(cache).exists():
        return joblib.load(cache)

    classes = list(classes)
    n = len(classes)
    if trees is None:
        trees = all_trees(classes) if n <= 7 else sample_trees(classes, N)

    vec = TfidfVectorizer(analyzer=_split_analyzer(n), use_idf=False, norm="l2")
    tfidf = vec.fit_transform(trees)
    sim = cosine_similarity(tfidf)
    dist = np.sqrt(np.maximum(0, 2.0 * (1.0 - sim)))
    np.fill_diagonal(dist, 0.0)

    mds = MDS(
        n_components=dim, dissimilarity="precomputed",
        random_state=seed, n_init=1, max_iter=1000, eps=1e-6, n_jobs=-1,
    )
    coords = mds.fit_transform(dist)

    if cache is not None:
        joblib.dump((trees, coords), cache)
    return trees, coords


def spatial_autocorrelation(values, coords, k=50, permutations=999):
    """Global Moran's I for `values` over a tree-space embedding.

    Needs the optional 'spatial' extra: pip install ndscape[spatial]
    """
    try:
        from esda import Moran
        from libpysal.weights import KNN
    except ImportError as e:
        raise ImportError(
            "spatial_autocorrelation needs esda and libpysal. "
            "Install with: pip install ndscape[spatial]"
        ) from e

    import warnings
    k = min(k, len(coords) - 1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        w = KNN.from_array(coords, k=k)
    w.transform = "r"
    mi = Moran(np.asarray(values), w, permutations=permutations)
    return {"I": mi.I, "p_sim": mi.p_sim}


def analyze(X, y, classes, X_test=None, y_test=None, base="lr", N=2000, seed=0, cache=None):
    """Fit every candidate ND tree on (X, y), score it, and place it in tree-space.

    If X_test/y_test are omitted, scores are computed on (X, y). Returns a
    list of dicts: tree, accuracy, logloss, coord.

    `cache` is an optional .joblib path for the embedding (see `embed_trees`)
    — reuse it across calls so the MDS step doesn't rerun every time.
    """
    classes = list(classes)
    Xte = X if X_test is None else X_test
    yte = y if y_test is None else y_test

    trees, coords = embed_trees(classes, N=N, seed=seed, cache=cache)
    rows = []
    for tree, coord in zip(trees, coords):
        nd = ND(tree, classes).fit(X, y, base=base)
        s = nd.score(Xte, yte)
        rows.append({"tree": tree, "accuracy": s["accuracy"], "logloss": s["logloss"], "coord": coord})
    return rows
