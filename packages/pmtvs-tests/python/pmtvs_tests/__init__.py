"""
pmtvs-tests — Statistical hypothesis testing primitives.
"""

__version__ = "0.1.0"
BACKEND = "python"

from pmtvs_tests.tests import (
    bootstrap_mean,
    bootstrap_confidence_interval,
    permutation_test,
    surrogate_test,
    adf_test,
    runs_test,
    mann_kendall_test,
)

__all__ = [
    "__version__",
    "BACKEND",
    "bootstrap_mean",
    "bootstrap_confidence_interval",
    "permutation_test",
    "surrogate_test",
    "adf_test",
    "runs_test",
    "mann_kendall_test",
]
