"""Plot a tree-space embedding (the kind of figure used in the paper).

Both functions take `rows`, the list of dicts returned by `ndscape.analyze()`
(each needs "coord" and the metric you're plotting). Needs matplotlib for
`plot`, and bokeh as well for `plot_interactive`:

    pip install ndscape[plot]
"""

import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# red -> yellow -> green, same palette used for the paper's accuracy/variance plots
BRIGHT = LinearSegmentedColormap.from_list("bright_RYG", ["#ff2d2d", "#fff176", "#00e676"], N=100)


def plot(rows, metric="accuracy", path=None, ax=None):
    """Static scatter of the embedding, colored by `metric`. Marks the best tree with a black x.

    Pass `path` to save a PNG/PDF, or `ax` to draw onto an existing matplotlib axes.
    Returns (fig, ax).
    """
    import matplotlib.pyplot as plt

    coords = np.array([r["coord"] for r in rows])
    values = np.array([r[metric] for r in rows], dtype=float)
    best = int(np.argmin(values)) if metric == "logloss" else int(np.argmax(values))

    if ax is None:
        fig, ax = plt.subplots(figsize=(8.0, 5.2))
    else:
        fig = ax.figure

    sc = ax.scatter(coords[:, 0], coords[:, 1], c=values, cmap=BRIGHT, s=30, linewidths=0)
    fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02).set_label(metric)
    ax.scatter(coords[best, 0], coords[best, 1], marker="x", s=220, c="black", linewidths=2.2, zorder=12)
    ax.set_xlabel("MDS-1")
    ax.set_ylabel("MDS-2")
    ax.grid(True, lw=0.3, alpha=0.4)
    fig.tight_layout()

    if path is not None:
        fig.savefig(path, dpi=300, bbox_inches="tight")
    return fig, ax


def plot_interactive(rows, metric="accuracy", path="tree_space.html"):
    """Same plot as `plot`, but as an HTML page with pan/zoom/hover (accuracy + logloss).

    Needs bokeh: pip install ndscape[plot]. Returns the bokeh figure.
    """
    from bokeh.models import ColorBar, ColumnDataSource, HoverTool, LinearColorMapper
    from bokeh.plotting import figure, output_file, save
    from matplotlib.colors import to_hex

    coords = np.array([r["coord"] for r in rows])
    values = np.array([r[metric] for r in rows], dtype=float)
    palette = [to_hex(BRIGHT(i / 99)) for i in range(100)]
    mapper = LinearColorMapper(palette=palette, low=float(values.min()), high=float(values.max()))

    source = ColumnDataSource(dict(
        x=coords[:, 0], y=coords[:, 1], value=values,
        accuracy=[r["accuracy"] for r in rows],
        logloss=[r["logloss"] for r in rows],
    ))

    p = figure(width=800, height=520, x_axis_label="MDS-1", y_axis_label="MDS-2",
               tools="pan,wheel_zoom,box_zoom,reset,save")
    points = p.scatter("x", "y", source=source, size=6, line_color="black", line_width=0.4,
                        fill_color={"field": "value", "transform": mapper}, fill_alpha=0.9)
    p.add_tools(HoverTool(renderers=[points], tooltips=[
        ("accuracy", "@accuracy{0.000}"), ("logloss", "@logloss{0.000}"),
    ]))
    p.add_layout(ColorBar(color_mapper=mapper, label_standoff=8, title=metric), "right")

    output_file(path)
    save(p)
    return p
