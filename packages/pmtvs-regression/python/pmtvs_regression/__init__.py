"""
pmtvs-regression — Pairwise regression primitives.

numpy in, number/array out.
"""

__version__ = "0.3.2"
BACKEND = "python"

# --- Public API ---
from pmtvs_regression.regression import (
    linear_regression,
    ratio,
    product,
    difference,
    sum_signals,
)

__all__ = [
    "__version__",
    "BACKEND",
    "linear_regression",
    "ratio",
    "product",
    "difference",
    "sum_signals",
]
