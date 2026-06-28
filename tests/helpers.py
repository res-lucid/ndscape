"""Shared test helpers (not fixtures — functions that tests import directly)."""


def order_tree(tree, n_classes):
    """Reorders tree splits into BFS order."""
    d = {tuple(sorted(L + R)): (L, R) for L, R in tree}
    out = []
    q = [d[tuple(range(n_classes))]]
    while q:
        split = q.pop(0)
        out.append(split)
        for child in split:
            if child in d:
                q.append(d[child])
    return tuple(out)
