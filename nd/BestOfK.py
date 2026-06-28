import numpy as np
from sklearn.base import clone
from nd import NestedDichotomy

def generate(gen, n, X, y, base, labels=None, K=10, seed=0):
    labels = list(range(n)) if labels is None else list(labels)
    c = max(map(int, labels)) + 1

    def score(t):
        nd = NestedDichotomy.parse(t)
        NestedDichotomy.train(nd, X, y, lambda **kw: clone(base))
        return np.mean(NestedDichotomy.predict_proba(nd, X, c).argmax(1) == y)

    return max((gen(n, labels=labels, seed=seed + i) for i in range(K)), key=score)