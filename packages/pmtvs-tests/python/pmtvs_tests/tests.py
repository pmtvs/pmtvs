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


def bootstrap_ci(
    data: np.ndarray,
    statistic: Callable = np.mean,
    confidence: float = 0.95,
    n_bootstrap: int = 1000
) -> Tuple[float, float]:
    """
    Compute bootstrap confidence interval (alias for bootstrap_confidence_interval).

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
    return bootstrap_confidence_interval(data, statistic, confidence, n_bootstrap)


def bootstrap_std(
    data: np.ndarray,
    n_bootstrap: int = 1000
) -> float:
    """
    Compute bootstrap standard error of the mean.

    Parameters
    ----------
    data : np.ndarray
        Input data
    n_bootstrap : int
        Number of bootstrap samples

    Returns
    -------
    float
        Bootstrap standard error
    """
    data = np.asarray(data).flatten()
    data = data[~np.isnan(data)]
    n = len(data)

    if n < 2:
        return np.nan

    means = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        means[i] = np.mean(sample)

    return float(np.std(means, ddof=1))


def block_bootstrap_ci(
    data: np.ndarray,
    statistic: Callable = np.mean,
    block_size: Optional[int] = None,
    confidence: float = 0.95,
    n_bootstrap: int = 1000
) -> Tuple[float, float]:
    """
    Block bootstrap confidence interval for dependent data.

    Parameters
    ----------
    data : np.ndarray
        Input data (time series)
    statistic : callable
        Statistic function
    block_size : int, optional
        Size of contiguous blocks. Defaults to int(sqrt(n)).
    confidence : float
        Confidence level
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

    if n < 4:
        return (np.nan, np.nan)

    if block_size is None:
        block_size = max(1, int(np.sqrt(n)))

    n_blocks = int(np.ceil(n / block_size))
    stats = np.zeros(n_bootstrap)

    for i in range(n_bootstrap):
        # Sample blocks with replacement
        blocks = []
        for _ in range(n_blocks):
            start = np.random.randint(0, n - block_size + 1)
            blocks.append(data[start:start + block_size])
        sample = np.concatenate(blocks)[:n]
        stats[i] = statistic(sample)

    alpha = 1 - confidence
    lower = np.percentile(stats, 100 * alpha / 2)
    upper = np.percentile(stats, 100 * (1 - alpha / 2))

    return (float(lower), float(upper))


def marchenko_pastur_test(
    data: np.ndarray,
    gamma: Optional[float] = None
) -> Tuple[float, float]:
    """
    Test whether eigenvalue distribution follows Marchenko-Pastur law.

    Useful for detecting signal vs. noise in correlation matrices.

    Parameters
    ----------
    data : np.ndarray
        Data matrix (n_samples x n_features)
    gamma : float, optional
        Ratio n_features / n_samples. Computed from data if None.

    Returns
    -------
    tuple
        (max_eigenvalue, mp_upper_bound) where mp_upper_bound is the
        theoretical maximum eigenvalue under null (pure noise)
    """
    data = np.asarray(data, dtype=np.float64)

    if data.ndim == 1:
        return (np.nan, np.nan)

    n, p = data.shape
    if n < 4 or p < 2:
        return (np.nan, np.nan)

    # Remove NaN rows
    mask = ~np.any(np.isnan(data), axis=1)
    data = data[mask]
    n = data.shape[0]

    if n < 4:
        return (np.nan, np.nan)

    if gamma is None:
        gamma = p / n

    # Standardize columns
    data = (data - np.mean(data, axis=0)) / (np.std(data, axis=0) + 1e-12)

    # Correlation matrix
    corr = data.T @ data / n

    eigenvalues = np.linalg.eigvalsh(corr)

    max_eig = float(np.max(eigenvalues))

    # Marchenko-Pastur upper bound
    mp_upper = float((1 + np.sqrt(gamma)) ** 2)

    return (max_eig, mp_upper)


def arch_test(
    signal: np.ndarray,
    nlags: int = 5
) -> Tuple[float, float]:
    """
    Engle's ARCH test for heteroscedasticity.

    Tests whether residuals exhibit autoregressive conditional
    heteroscedasticity.

    Parameters
    ----------
    signal : np.ndarray
        Input signal (typically residuals)
    nlags : int
        Number of lags

    Returns
    -------
    tuple
        (lm_statistic, p_value)
    """
    signal = np.asarray(signal, dtype=np.float64).flatten()
    signal = signal[~np.isnan(signal)]
    n = len(signal)

    if n < nlags + 10:
        return (np.nan, np.nan)

    # Demean and square
    resid = signal - np.mean(signal)
    sq_resid = resid ** 2

    # Regress squared residuals on lagged squared residuals
    y = sq_resid[nlags:]
    X = np.column_stack(
        [np.ones(len(y))] + [sq_resid[nlags - i - 1:n - i - 1] for i in range(nlags)]
    )

    try:
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        fitted = X @ beta
        ss_res = np.sum((y - fitted) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)

        if ss_tot <= 0:
            return (np.nan, np.nan)

        r_squared = 1 - ss_res / ss_tot
        lm_stat = float(len(y) * r_squared)

        # Chi-squared p-value
        from scipy.stats import chi2
        p_value = float(1 - chi2.cdf(lm_stat, nlags))

        return (lm_stat, p_value)
    except (np.linalg.LinAlgError, Exception):
        return (np.nan, np.nan)
