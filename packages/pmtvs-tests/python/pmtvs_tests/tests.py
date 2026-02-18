"""
Statistical Hypothesis Testing Functions
"""

import numpy as np
from typing import Callable, Optional, Tuple


def bootstrap_mean(
    data: np.ndarray,
    n_bootstrap: int = 1000
) -> np.ndarray:
    """
    Compute bootstrap distribution of the mean.

    Parameters
    ----------
    data : np.ndarray
        Input data
    n_bootstrap : int
        Number of bootstrap samples

    Returns
    -------
    np.ndarray
        Bootstrap distribution of means
    """
    data = np.asarray(data).flatten()
    data = data[~np.isnan(data)]
    n = len(data)

    if n < 2:
        return np.array([np.nan])

    means = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        means[i] = np.mean(sample)

    return means


def bootstrap_confidence_interval(
    data: np.ndarray,
    statistic: Callable = np.mean,
    confidence: float = 0.95,
    n_bootstrap: int = 1000
) -> Tuple[float, float]:
    """
    Compute bootstrap confidence interval.

    Parameters
    ----------
    data : np.ndarray
        Input data
    statistic : callable
        Statistic function
    confidence : float
        Confidence level (e.g., 0.95)
    n_bootstrap : int
        Number of bootstrap samples

    Returns
    -------
    tuple
        (lower_bound, upper_bound)
    """
    data = np.asarray(data).flatten()
    data = data[~np.isnan(data)]
    n = len(data)

    if n < 2:
        return (np.nan, np.nan)

    stats = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        stats[i] = statistic(sample)

    alpha = 1 - confidence
    lower = np.percentile(stats, 100 * alpha / 2)
    upper = np.percentile(stats, 100 * (1 - alpha / 2))

    return (float(lower), float(upper))


def permutation_test(
    x: np.ndarray,
    y: np.ndarray,
    statistic: Callable = lambda a, b: np.mean(a) - np.mean(b),
    n_permutations: int = 1000
) -> Tuple[float, float]:
    """
    Perform permutation test for difference between groups.

    Parameters
    ----------
    x : np.ndarray
        First group
    y : np.ndarray
        Second group
    statistic : callable
        Test statistic function
    n_permutations : int
        Number of permutations

    Returns
    -------
    tuple
        (observed_statistic, p_value)
    """
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]

    if len(x) < 2 or len(y) < 2:
        return (np.nan, np.nan)

    observed = statistic(x, y)

    combined = np.concatenate([x, y])
    n_x = len(x)
    count = 0

    for _ in range(n_permutations):
        np.random.shuffle(combined)
        perm_x = combined[:n_x]
        perm_y = combined[n_x:]
        perm_stat = statistic(perm_x, perm_y)

        if abs(perm_stat) >= abs(observed):
            count += 1

    p_value = (count + 1) / (n_permutations + 1)

    return (float(observed), float(p_value))


def surrogate_test(
    signal: np.ndarray,
    statistic: Callable,
    n_surrogates: int = 100,
    method: str = "phase_shuffle"
) -> Tuple[float, float]:
    """
    Perform surrogate data test.

    Parameters
    ----------
    signal : np.ndarray
        Input signal
    statistic : callable
        Test statistic function
    n_surrogates : int
        Number of surrogates
    method : str
        Surrogate method: "phase_shuffle" or "random_shuffle"

    Returns
    -------
    tuple
        (observed_statistic, p_value)
    """
    signal = np.asarray(signal).flatten()
    signal = signal[~np.isnan(signal)]
    n = len(signal)

    if n < 4:
        return (np.nan, np.nan)

    observed = statistic(signal)
    count = 0

    for _ in range(n_surrogates):
        if method == "phase_shuffle":
            # Phase randomization (preserves power spectrum)
            fft = np.fft.rfft(signal)
            phases = np.angle(fft)
            random_phases = np.random.uniform(0, 2 * np.pi, len(phases))
            new_fft = np.abs(fft) * np.exp(1j * random_phases)
            surrogate = np.fft.irfft(new_fft, n=n)
        else:  # random_shuffle
            surrogate = np.random.permutation(signal)

        surrogate_stat = statistic(surrogate)

        if abs(surrogate_stat) >= abs(observed):
            count += 1

    p_value = (count + 1) / (n_surrogates + 1)

    return (float(observed), float(p_value))


def adf_test(
    signal: np.ndarray,
    max_lag: Optional[int] = None
) -> Tuple[float, float]:
    """
    Simplified Augmented Dickey-Fuller test for stationarity.

    Parameters
    ----------
    signal : np.ndarray
        Input signal
    max_lag : int, optional
        Maximum lag for autoregression

    Returns
    -------
    tuple
        (test_statistic, critical_value_5pct)
    """
    signal = np.asarray(signal).flatten()
    signal = signal[~np.isnan(signal)]
    n = len(signal)

    if n < 10:
        return (np.nan, np.nan)

    if max_lag is None:
        max_lag = int(np.floor(np.power(n - 1, 1/3)))

    # First difference
    diff = np.diff(signal)

    # Lagged level
    y_lag = signal[:-1]

    # Simple regression: diff = alpha + beta * y_lag + error
    X = np.column_stack([np.ones(len(diff)), y_lag])
    y = diff

    try:
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        residuals = y - X @ beta
        se = np.sqrt(np.sum(residuals**2) / (len(y) - 2))

        # Standard error of beta[1]
        XtX_inv = np.linalg.inv(X.T @ X)
        se_beta = se * np.sqrt(XtX_inv[1, 1])

        if se_beta > 0:
            t_stat = beta[1] / se_beta
        else:
            t_stat = np.nan

    except np.linalg.LinAlgError:
        t_stat = np.nan

    # Critical value at 5% (approximation for n > 100)
    critical_5pct = -2.86

    return (float(t_stat), critical_5pct)


def runs_test(signal: np.ndarray) -> Tuple[float, float]:
    """
    Runs test for randomness.

    Tests whether the sequence of above/below median values is random.

    Parameters
    ----------
    signal : np.ndarray
        Input signal

    Returns
    -------
    tuple
        (z_statistic, p_value)
    """
    signal = np.asarray(signal).flatten()
    signal = signal[~np.isnan(signal)]
    n = len(signal)

    if n < 10:
        return (np.nan, np.nan)

    median = np.median(signal)
    binary = (signal > median).astype(int)

    # Count runs
    runs = 1
    for i in range(1, n):
        if binary[i] != binary[i-1]:
            runs += 1

    # Expected runs under null
    n1 = np.sum(binary)
    n0 = n - n1

    if n1 == 0 or n0 == 0:
        return (np.nan, np.nan)

    expected_runs = (2 * n0 * n1) / n + 1
    var_runs = (2 * n0 * n1 * (2 * n0 * n1 - n)) / (n**2 * (n - 1))

    if var_runs <= 0:
        return (np.nan, np.nan)

    z = (runs - expected_runs) / np.sqrt(var_runs)

    # Two-tailed p-value (normal approximation)
    from math import erf
    p_value = 2 * (1 - 0.5 * (1 + erf(abs(z) / np.sqrt(2))))

    return (float(z), float(p_value))


def mann_kendall_test(signal: np.ndarray) -> Tuple[float, float]:
    """
    Mann-Kendall test for monotonic trend.

    Parameters
    ----------
    signal : np.ndarray
        Input signal

    Returns
    -------
    tuple
        (tau, p_value)
    """
    signal = np.asarray(signal).flatten()
    signal = signal[~np.isnan(signal)]
    n = len(signal)

    if n < 4:
        return (np.nan, np.nan)

    # Count concordant and discordant pairs
    s = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            diff = signal[j] - signal[i]
            if diff > 0:
                s += 1
            elif diff < 0:
                s -= 1

    # Kendall's tau
    tau = s / (n * (n - 1) / 2)

    # Variance of S (assuming no ties)
    var_s = n * (n - 1) * (2 * n + 5) / 18

    # Z-statistic
    if s > 0:
        z = (s - 1) / np.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / np.sqrt(var_s)
    else:
        z = 0

    # Two-tailed p-value
    from math import erf
    p_value = 2 * (1 - 0.5 * (1 + erf(abs(z) / np.sqrt(2))))

    return (float(tau), float(p_value))
