"""
pmtvs-tests — Statistical hypothesis testing primitives.
"""

__version__ = "0.2.0"
BACKEND = "python"

from pmtvs_tests.tests import (
    bootstrap_mean,
    bootstrap_confidence_interval,
    permutation_test,
    surrogate_test,
    adf_test,
    runs_test,
    mann_kendall_test,
    bootstrap_ci,
    bootstrap_std,
    block_bootstrap_ci,
    marchenko_pastur_test,
    arch_test,
)

from pmtvs_tests.hypothesis import (
    t_test,
    t_test_paired,
    t_test_independent,
    f_test,
    chi_squared_test,
    mannwhitney_test,
    kruskal_test,
    anova,
    shapiro_test,
    levene_test,
)

from pmtvs_tests.stationarity import (
    stationarity_test,
    trend,
    changepoints,
    kpss_test,
    phillips_perron_test,
)

__all__ = [
    "__version__",
    "BACKEND",
    # tests.py
    "bootstrap_mean",
    "bootstrap_confidence_interval",
    "permutation_test",
    "surrogate_test",
    "adf_test",
    "runs_test",
    "mann_kendall_test",
    "bootstrap_ci",
    "bootstrap_std",
    "block_bootstrap_ci",
    "marchenko_pastur_test",
    "arch_test",
    # hypothesis.py
    "t_test",
    "t_test_paired",
    "t_test_independent",
    "f_test",
    "chi_squared_test",
    "mannwhitney_test",
    "kruskal_test",
    "anova",
    "shapiro_test",
    "levene_test",
    # stationarity.py
    "stationarity_test",
    "trend",
    "changepoints",
    "kpss_test",
    "phillips_perron_test",
]
