"""
Stationarity and Trend Analysis Functions
"""

import numpy as np
from typing import Optional, Tuple, List


def stationarity_test(
    signal: np.ndarray,
    window_size: Optional[int] = None
) -> Tuple[float, bool]:
    """
    Test stationarity by comparing rolling mean/variance across windows.

    Parameters
    ----------
    signal : np.ndarray
        Input signal
    window_size : int, optional
        Window size for splitting. Defaults to n // 4.

    Returns
    -------
    tuple
        (ratio, is_stationary) where ratio is max_var_change / mean_var
        and is_stationary is True if ratio < 1.0
    """
    signal = np.asarray(signal, dtype=np.float64).flatten()
    signal = signal[~np.isnan(signal)]
    n = len(signal)

    if n < 20:
        return (np.nan, False)

    if window_size is None:
        window_size = n // 4

    n_windows = n // window_size
    if n_windows < 2:
        return (np.nan, False)

    means = []
    variances = []
    for i in range(n_windows):
        chunk = signal[i * window_size:(i + 1) * window_size]
        means.append(np.mean(chunk))
        variances.append(np.var(chunk))

    means = np.array(means)
    variances = np.array(variances)

    mean_var = np.mean(variances)
    if mean_var < 1e-12:
        return (0.0, True)

    var_change = np.max(np.abs(np.diff(variances)))
    ratio = float(var_change / mean_var)

    return (ratio, ratio < 1.0)


def trend(
    signal: np.ndarray,
    method: str = "linear"
) -> Tuple[float, float]:
    """
    Estimate trend in signal.

    Parameters
    ----------
    signal : np.ndarray
        Input signal
    method : str
        'linear' for linear regression slope

    Returns
    -------
    tuple
        (slope, r_squared)
    """
    signal = np.asarray(signal, dtype=np.float64).flatten()
    signal = signal[~np.isnan(signal)]
    n = len(signal)

    if n < 4:
        return (np.nan, np.nan)

    t = np.arange(n, dtype=np.float64)

    if method == "linear":
        X = np.column_stack([np.ones(n), t])
        try:
            beta = np.linalg.lstsq(X, signal, rcond=None)[0]
            predicted = X @ beta
            ss_res = np.sum((signal - predicted) ** 2)
            ss_tot = np.sum((signal - np.mean(signal)) ** 2)
            r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
            return (float(beta[1]), float(r_squared))
        except np.linalg.LinAlgError:
            return (np.nan, np.nan)

    return (np.nan, np.nan)


def changepoints(
    signal: np.ndarray,
    n_bkps: int = 1,
    min_size: int = 10
) -> List[int]:
    """
    Detect changepoints using cumulative sum (CUSUM) approach.

    Parameters
    ----------
    signal : np.ndarray
        Input signal
    n_bkps : int
        Number of changepoints to detect
    min_size : int
        Minimum segment size

    Returns
    -------
    list
        List of changepoint indices
    """
    signal = np.asarray(signal, dtype=np.float64).flatten()
    signal = signal[~np.isnan(signal)]
    n = len(signal)

    if n < 2 * min_size:
        return []

    def _cost_diff(seg):
        """Cost reduction from splitting a segment."""
        best_idx = -1
        best_gain = 0.0
        seg_mean = np.mean(seg)
        total_cost = np.sum((seg - seg_mean) ** 2)

        for i in range(min_size, len(seg) - min_size):
            left = seg[:i]
            right = seg[i:]
            cost_left = np.sum((left - np.mean(left)) ** 2)
            cost_right = np.sum((right - np.mean(right)) ** 2)
            gain = total_cost - cost_left - cost_right
            if gain > best_gain:
                best_gain = gain
                best_idx = i
        return best_idx, best_gain

    # Greedy binary segmentation
    segments = [(0, n)]
    bkps = []

    for _ in range(n_bkps):
        best_seg_idx = -1
        best_split = -1
        best_gain = 0.0

        for seg_idx, (start, end) in enumerate(segments):
            seg = signal[start:end]
            if len(seg) < 2 * min_size:
                continue
            split_pos, gain = _cost_diff(seg)
            if gain > best_gain:
                best_gain = gain
                best_split = start + split_pos
                best_seg_idx = seg_idx

        if best_seg_idx < 0 or best_split < 0:
            break

        start, end = segments[best_seg_idx]
        segments[best_seg_idx] = (start, best_split)
        segments.insert(best_seg_idx + 1, (best_split, end))
        bkps.append(best_split)

    return sorted(bkps)


def kpss_test(
    signal: np.ndarray,
    regression: str = "c",
    nlags: Optional[int] = None
) -> Tuple[float, float]:
    """
    KPSS test for stationarity.

    Parameters
    ----------
    signal : np.ndarray
        Input signal
    regression : str
        'c' for level stationarity, 'ct' for trend stationarity
    nlags : int, optional
        Number of lags for Newey-West estimator

    Returns
    -------
    tuple
        (kpss_statistic, p_value)
    """
    signal = np.asarray(signal, dtype=np.float64).flatten()
    signal = signal[~np.isnan(signal)]
    n = len(signal)

    if n < 10:
        return (np.nan, np.nan)

    # Detrend
    t = np.arange(n, dtype=np.float64)
    if regression == "ct":
        X = np.column_stack([np.ones(n), t])
    else:
        X = np.ones((n, 1))

    try:
        beta = np.linalg.lstsq(X, signal, rcond=None)[0]
        residuals = signal - X @ beta
    except np.linalg.LinAlgError:
        return (np.nan, np.nan)

    if nlags is None:
        nlags = int(np.ceil(np.sqrt(n)))

    # Partial sums of residuals
    cumsum = np.cumsum(residuals)

    # Newey-West estimator of long-run variance
    gamma_0 = np.sum(residuals ** 2) / n
    nw_var = gamma_0
    for lag in range(1, nlags + 1):
        weight = 1 - lag / (nlags + 1)
        gamma_lag = np.sum(residuals[lag:] * residuals[:-lag]) / n
        nw_var += 2 * weight * gamma_lag

    if nw_var <= 0:
        return (np.nan, np.nan)

    kpss_stat = float(np.sum(cumsum ** 2) / (n ** 2 * nw_var))

    # Approximate p-value using critical value tables
    if regression == "ct":
        # Critical values: 10%=0.119, 5%=0.146, 2.5%=0.176, 1%=0.216
        crit_vals = [0.119, 0.146, 0.176, 0.216]
        p_vals = [0.10, 0.05, 0.025, 0.01]
    else:
        # Critical values: 10%=0.347, 5%=0.463, 2.5%=0.574, 1%=0.739
        crit_vals = [0.347, 0.463, 0.574, 0.739]
        p_vals = [0.10, 0.05, 0.025, 0.01]

    if kpss_stat < crit_vals[0]:
        p_value = 0.10  # p > 0.10
    elif kpss_stat > crit_vals[-1]:
        p_value = 0.01  # p < 0.01
    else:
        # Linear interpolation
        for i in range(len(crit_vals) - 1):
            if crit_vals[i] <= kpss_stat <= crit_vals[i + 1]:
                frac = (kpss_stat - crit_vals[i]) / (crit_vals[i + 1] - crit_vals[i])
                p_value = p_vals[i] - frac * (p_vals[i] - p_vals[i + 1])
                break
        else:
            p_value = 0.05

    return (kpss_stat, float(p_value))


def phillips_perron_test(
    signal: np.ndarray,
    nlags: Optional[int] = None
) -> Tuple[float, float]:
    """
    Phillips-Perron test for unit root.

    Parameters
    ----------
    signal : np.ndarray
        Input signal
    nlags : int, optional
        Number of lags for Newey-West correction

    Returns
    -------
    tuple
        (pp_statistic, critical_value_5pct)
    """
    signal = np.asarray(signal, dtype=np.float64).flatten()
    signal = signal[~np.isnan(signal)]
    n = len(signal)

    if n < 10:
        return (np.nan, np.nan)

    if nlags is None:
        nlags = int(np.ceil(np.power(n, 1/3)))

    # OLS regression: delta_y = alpha + beta * y_{t-1} + e
    diff = np.diff(signal)
    y_lag = signal[:-1]

    X = np.column_stack([np.ones(len(diff)), y_lag])
    y = diff

    try:
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        residuals = y - X @ beta
        se_sq = np.sum(residuals ** 2) / (len(y) - 2)

        # Newey-West long-run variance
        gamma_0 = np.sum(residuals ** 2) / len(residuals)
        lrv = gamma_0
        for lag in range(1, nlags + 1):
            weight = 1 - lag / (nlags + 1)
            gamma_lag = np.sum(residuals[lag:] * residuals[:-lag]) / len(residuals)
            lrv += 2 * weight * gamma_lag

        # Standard error of beta[1]
        XtX_inv = np.linalg.inv(X.T @ X)
        se_beta = np.sqrt(se_sq * XtX_inv[1, 1])

        if se_beta <= 0:
            return (np.nan, np.nan)

        # ADF t-stat
        t_adf = beta[1] / se_beta

        # PP correction
        n_obs = len(y)
        correction = (n_obs * (lrv - gamma_0) * se_beta) / (2 * np.sqrt(lrv) * np.sqrt(se_sq))
        pp_stat = float(t_adf * np.sqrt(gamma_0 / lrv) - correction)

    except np.linalg.LinAlgError:
        pp_stat = np.nan

    # Critical value at 5% (approximation)
    critical_5pct = -2.86

    return (pp_stat, critical_5pct)
