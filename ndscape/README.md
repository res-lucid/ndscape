# ndscape

Fit, score, and embed nested-dichotomy (ND) trees for multi-class classification.

A nested dichotomy reduces a C-class problem to a tree of binary splits
(e.g. {0,1,2} vs {3,4}, then {0} vs {1,2}, ...). ndscape lets you fit one,
score it, or place a whole population of candidate trees in a 2-D
"tree-space" to see how a property (accuracy, variance, ...) varies across
tree structures.

## Install

```
pip install ndscape
pip install ndscape[spatial]   # adds Moran's I support (esda, libpysal)
pip install ndscape[plot]      # adds plotting (matplotlib, bokeh)
```

## Quickstart

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import ndscape as nds

X, y = load_iris(return_X_y=True)
classes = sorted(set(y))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

nd = nds.fit(X_train, y_train, classes=classes, base="lr")
nd.predict(X_test)
nd.score(X_test, y_test)   # {"accuracy": ..., "logloss": ...}
```

`classes` is the list of class labels in your `y` (`sorted(set(y))` works for
integer or string labels). `nds.fit` samples one ND tree automatically; pass
`tree=...` to use a specific one (see "A `tree` is..." below).

## Use cases

**You have a dataset and a binary classifier.**

`base` can be the string `"lr"` or `"decisiontree"`, or your own unfitted
scikit-learn estimator — a fresh clone of it is fit at every split.

```python
from sklearn.svm import SVC

nd = nds.fit(X_train, y_train, classes=classes, base=SVC(probability=True, kernel="linear"))
```

**You have a train/test split and want a score.**

```python
nd = nds.ND(tree, classes).fit(X_train, y_train, base="lr")
nd.score(X_test, y_test)   # {"accuracy": ..., "logloss": ...}
```

**You already trained the per-split models yourself.**

```python
# models in the same order as tree, or a {(left, right): model} dict — either works
nd = nds.ND.from_trained(tree, classes, models=[fitted_model_1, fitted_model_2, ...])
nd.predict_proba(X_test)
```

**You already scored a set of trees and want to see where they sit in tree-space.**

```python
trees, coords = nds.embed_trees(classes)
nds.spatial_autocorrelation(my_scores, coords)   # {"I": ..., "p_sim": ...}
```

**You just want the whole picture: fit, score, and embed every candidate tree.**

```python
rows = nds.analyze(X_train, y_train, classes, X_test=X_test, y_test=y_test, base="lr")
# [{"tree": ..., "accuracy": ..., "logloss": ..., "coord": array([...])}, ...]
```

**You want a picture of that tree-space.**

```python
nds.plot(rows, metric="accuracy", path="tree_space.png")        # static PNG/PDF
nds.plot_interactive(rows, metric="accuracy", path="tree_space.html")  # pan/zoom/hover
```

Both color points by `metric` and mark the best tree with a black x. Needs
`ndscape[plot]`.

**The embedding is slow to recompute and you want to reuse it.**

```python
rows = nds.analyze(X_train, y_train, classes, cache="embedding.joblib")
```

The MDS step in `embed_trees`/`analyze` is the slow part. Pass `cache=` a
`.joblib` path: the first call computes the embedding and saves it there,
later calls with the same path just load it.

A `tree` is a list of `(left, right)` tuples of class labels, e.g.
`[((0, 1), (2, 3)), ((0,), (1,)), ((2,), (3,))]`. Use `nds.all_trees(classes)`
(exhaustive, for small C) or `nds.sample_trees(classes, N)` (for larger C)
to generate candidates.

`base` accepts `"lr"`, `"decisiontree"`, or any unfitted scikit-learn
estimator with `fit`/`predict_proba`.
