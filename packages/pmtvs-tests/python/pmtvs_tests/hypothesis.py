"""
Parametric and Non-parametric Hypothesis Testing Functions
"""

import numpy as np
from typing import Tuple


def t_test(
    data: np.ndarray,
    popmean: float = 0.0
) -> Tuple[float, float]:
    """
    One-sample t-test.

    Parameters
    ----------
    data : np.ndarray
        Input data
    popmean : float
        Expected population mean

    Returns
    -------
    tuple
        (t_statistic, p_value)
    """
    data = np.asarray(data, dtype=np.float64).flatten()
    data = data[~np.isnan(data)]
    if len(data) < 2:
        return (np.nan, np.nan)
    from scipy.stats import ttest_1samp
    stat, p = ttest_1samp(data, popmean)
    return (float(stat), float(p))


def t_test_paired(
    x: np.ndarray,
    y: np.ndarray
) -> Tuple[float, float]:
    """
    Paired t-test.

    Parameters
    ----------
    x : np.ndarray
        First group
    y : np.ndarray
        Second group

    Returns
    -------
    tuple
        (t_statistic, p_value)
    """
    x = np.asarray(x, dtype=np.float64).flatten()
    y = np.asarray(y, dtype=np.float64).flatten()
    n = min(len(x), len(y))
    x, y = x[:n], y[:n]
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    if len(x) < 2:
        return (np.nan, np.nan)
    from scipy.stats import ttest_rel
    stat, p = ttest_rel(x, y)
    return (float(stat), float(p))


def t_test_independent(
    x: np.ndarray,
    y: np.ndarray,
    equal_var: bool = True
) -> Tuple[float, float]:
    """
    Independent two-sample t-test.

    Parameters
    ----------
    x : np.ndarray
        First group
    y : np.ndarray
        Second group
    equal_var : bool
        If True, assume equal variances (Student's t-test).
        If False, use Welch's t-test.

    Returns
    -------
    tuple
        (t_statistic, p_value)
    """
    x = np.asarray(x, dtype=np.float64).flatten()
    y = np.asarray(y, dtype=np.float64).flatten()
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    if len(x) < 2 or len(y) < 2:
        return (np.nan, np.nan)
    from scipy.stats import ttest_ind
    stat, p = ttest_ind(x, y, equal_var=equal_var)
    return (float(stat), float(p))


def f_test(
    x: np.ndarray,
    y: np.ndarray
) -> Tuple[float, float]:
    """
    F-test for equality of variances.

    Parameters
    ----------
    x : np.ndarray
        First group
    y : np.ndarray
        Second group

    Returns
    -------
    tuple
        (f_statistic, p_value)
    """
    x = np.asarray(x, dtype=np.float64).flatten()
    y = np.asarray(y, dtype=np.float64).flatten()
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    if len(x) < 2 or len(y) < 2:
        return (np.nan, np.nan)
    var_x = np.var(x, ddof=1)
    var_y = np.var(y, ddof=1)
    if var_y == 0:
        return (np.nan, np.nan)
    f_stat = var_x / var_y
    df1 = len(x) - 1
    df2 = len(y) - 1
    from scipy.stats import f as f_dist
    p_value = 2 * min(f_dist.cdf(f_stat, df1, df2), 1 - f_dist.cdf(f_stat, df1, df2))
    return (float(f_stat), float(p_value))


def chi_squared_test(
    observed: np.ndarray,
    expected: np.ndarray = None
) -> Tuple[float, float]:
    """
    Chi-squared goodness-of-fit test.

    Parameters
    ----------
    observed : np.ndarray
        Observed frequencies
    expected : np.ndarray, optional
        Expected frequencies. If None, assumes uniform.

    Returns
    -------
    tuple
        (chi2_statistic, p_value)
    """
    observed = np.asarray(observed, dtype=np.float64).flatten()
    observed = observed[~np.isnan(observed)]
    if len(observed) < 2:
        return (np.nan, np.nan)
    from scipy.stats import chisquare
    if expected is not None:
        expected = np.asarray(expected, dtype=np.float64).flatten()
        expected = expected[~np.isnan(expected)]
        n = min(len(observed), len(expected))
        observed, expected = observed[:n], expected[:n]
        stat, p = chisquare(observed, f_exp=expected)
    else:
        stat, p = chisquare(observed)
    return (float(stat), float(p))


def mannwhitney_test(
    x: np.ndarray,
    y: np.ndarray,
    alternative: str = "two-sided"
) -> Tuple[float, float]:
    """
    Mann-Whitney U test (non-parametric).

    Parameters
    ----------
    x : np.ndarray
        First group
    y : np.ndarray
        Second group
    alternative : str
        'two-sided', 'less', or 'greater'

    Returns
    -------
    tuple
        (u_statistic, p_value)
    """
    x = np.asarray(x, dtype=np.float64).flatten()
    y = np.asarray(y, dtype=np.float64).flatten()
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    if len(x) < 2 or len(y) < 2:
        return (np.nan, np.nan)
    from scipy.stats import mannwhitneyu
    stat, p = mannwhitneyu(x, y, alternative=alternative)
    return (float(stat), float(p))


def kruskal_test(
    *groups
) -> Tuple[float, float]:
    """
    Kruskal-Wallis H-test (non-parametric one-way ANOVA).

    Parameters
    ----------
    *groups : np.ndarray
        Two or more groups of data

    Returns
    -------
    tuple
        (h_statistic, p_value)
    """
    clean_groups = []
    for g in groups:
        g = np.asarray(g, dtype=np.float64).flatten()
        g = g[~np.isnan(g)]
        if len(g) < 2:
            return (np.nan, np.nan)
        clean_groups.append(g)
    if len(clean_groups) < 2:
        return (np.nan, np.nan)
    from scipy.stats import kruskal as kruskal_wallis
    stat, p = kruskal_wallis(*clean_groups)
    return (float(stat), float(p))


def anova(
    *groups
) -> Tuple[float, float]:
    """
    One-way ANOVA.

    Parameters
    ----------
    *groups : np.ndarray
        Two or more groups of data

    Returns
    -------
    tuple
        (f_statistic, p_value)
    """
    clean_groups = []
    for g in groups:
        g = np.asarray(g, dtype=np.float64).flatten()
        g = g[~np.isnan(g)]
        if len(g) < 2:
            return (np.nan, np.nan)
        clean_groups.append(g)
    if len(clean_groups) < 2:
        return (np.nan, np.nan)
    from scipy.stats import f_oneway
    stat, p = f_oneway(*clean_groups)
    return (float(stat), float(p))


def shapiro_test(
    data: np.ndarray
) -> Tuple[float, float]:
    """
    Shapiro-Wilk test for normality.

    Parameters
    ----------
    data : np.ndarray
        Input data

    Returns
    -------
    tuple
        (w_statistic, p_value)
    """
    data = np.asarray(data, dtype=np.float64).flatten()
    data = data[~np.isnan(data)]
    if len(data) < 3:
        return (np.nan, np.nan)
    from scipy.stats import shapiro as shapiro_wilk
    stat, p = shapiro_wilk(data)
    return (float(stat), float(p))


def levene_test(
    *groups,
    center: str = "median"
) -> Tuple[float, float]:
    """
    Levene's test for equality of variances.

    Parameters
    ----------
    *groups : np.ndarray
        Two or more groups of data
    center : str
        'mean', 'median', or 'trimmed'

    Returns
    -------
    tuple
        (w_statistic, p_value)
    """
    clean_groups = []
    for g in groups:
        g = np.asarray(g, dtype=np.float64).flatten()
        g = g[~np.isnan(g)]
        if len(g) < 2:
            return (np.nan, np.nan)
        clean_groups.append(g)
    if len(clean_groups) < 2:
        return (np.nan, np.nan)
    from scipy.stats import levene as levene_stat
    stat, p = levene_stat(*clean_groups, center=center)
    return (float(stat), float(p))
