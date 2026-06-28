"""ndscape: fit, score, and embed nested-dichotomy (ND) trees for multi-class classification."""

from .api import (
    ND,
    all_trees,
    analyze,
    embed_trees,
    fit,
    sample_trees,
    spatial_autocorrelation,
)
from .plot import plot, plot_interactive

__all__ = [
    "ND",
    "all_trees",
    "analyze",
    "embed_trees",
    "fit",
    "sample_trees",
    "spatial_autocorrelation",
    "plot",
    "plot_interactive",
]
