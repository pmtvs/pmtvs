"""
Stability Analysis

Functions for analyzing stability properties of dynamical systems:
fixed points, stability indices, bifurcations.
"""

import numpy as np
from typing import Optional, Tuple, List


def fixed_point_detection(
    trajectory: np.ndarray,
    threshold: float = 0.01,
    min_duration: int = 10
) -> List[Tuple[int, int, np.ndarray]]:
    """
    Detect fixed points (stationary regions) in a trajectory.

    Parameters
    ----------
    trajectory : np.ndarray
        State-space trajectory of shape (n_points, n_dims)
    threshold : float
        Maximum velocity to consider as "fixed"
    min_duration : int
        Minimum number of points in a fixed region

    Returns
    -------
    list
        List of (start_idx, end_idx, center_point) tuples
    """
    trajectory = np.asarray(trajectory)

    if trajectory.ndim == 1:
        trajectory = trajectory.reshape(-1, 1)

    n_points = len(trajectory)

    if n_points < min_duration:
        return []

    # Compute velocities
    velocities = np.linalg.norm(np.diff(trajectory, axis=0), axis=1)

    # Find regions where velocity is below threshold
    fixed_regions = []
    in_fixed_region = False
    region_start = 0

    for i, v in enumerate(velocities):
        if v < threshold:
            if not in_fixed_region:
                region_start = i
                in_fixed_region = True
        else:
            if in_fixed_region:
                if i - region_start >= min_duration:
                    center = np.mean(trajectory[region_start:i], axis=0)
                    fixed_regions.append((region_start, i, center))
                in_fixed_region = False

    # Check final region
    if in_fixed_region and n_points - region_start >= min_duration:
        center = np.mean(trajectory[region_start:], axis=0)
        fixed_regions.append((region_start, n_points, center))

    return fixed_regions


def stability_index(
    trajectory: np.ndarray,
    dt: float = 1.0
) -> float:
    """
    Compute local stability index.

    Measures how quickly perturbations decay or grow.
    Negative values indicate stable (convergent) dynamics.
    Positive values indicate unstable (divergent) dynamics.

    Parameters
    ----------
    trajectory : np.ndarray
        State-space trajectory
    dt : float
        Time step

    Returns
    -------
    float
        Stability index
    """
    trajectory = np.asarray(trajectory)

    if trajectory.ndim == 1:
        trajectory = trajectory.reshape(-1, 1)

    n_points = len(trajectory)

    if n_points < 10:
        return np.nan

    # Compute distance from centroid over time
    centroid = np.mean(trajectory, axis=0)
    distances = np.linalg.norm(trajectory - centroid, axis=1)

    if np.all(distances < 1e-10):
        return -np.inf  # Fixed point

    # Fit exponential decay/growth
    # distance(t) ~ exp(lambda * t)
    # log(distance) ~ lambda * t

    valid_mask = distances > 1e-10
    if np.sum(valid_mask) < 5:
        return np.nan

    log_dist = np.log(distances[valid_mask])
    times = np.arange(len(distances))[valid_mask] * dt

    # Linear regression
    x_mean = np.mean(times)
    y_mean = np.mean(log_dist)

    num = np.sum((times - x_mean) * (log_dist - y_mean))
    den = np.sum((times - x_mean) ** 2)

    if den == 0:
        return 0.0

    return float(num / den)


def jacobian_eigenvalues(
    trajectory: np.ndarray,
    at_index: int = -1,
    neighborhood_size: int = 10
) -> np.ndarray:
    """
    Estimate Jacobian eigenvalues at a point in the trajectory.

    Uses local linear approximation to estimate the Jacobian
    of the flow map.

    Parameters
    ----------
    trajectory : np.ndarray
        State-space trajectory
    at_index : int
        Index at which to estimate (default: last point)
    neighborhood_size : int
        Number of points to use for local approximation

    Returns
    -------
    np.ndarray
        Eigenvalues of estimated Jacobian (complex)
    """
    trajectory = np.asarray(trajectory)

    if trajectory.ndim == 1:
        trajectory = trajectory.reshape(-1, 1)

    n_points, n_dims = trajectory.shape

    if at_index < 0:
        at_index = n_points + at_index

    if n_points < neighborhood_size + 2:
        return np.full(n_dims, np.nan + 0j)

    # Get neighborhood
    start = max(0, at_index - neighborhood_size // 2)
    end = min(n_points - 1, start + neighborhood_size)

    if end - start < n_dims + 1:
        return np.full(n_dims, np.nan + 0j)

    # Current and next states
    X = trajectory[start:end]
    Y = trajectory[start + 1:end + 1]

    # Center data
    X_mean = np.mean(X, axis=0)
    Y_mean = np.mean(Y, axis=0)

    X_centered = X - X_mean
    Y_centered = Y - Y_mean

    # Solve for Jacobian: Y = J @ X (in least squares sense)
    try:
        J, residuals, rank, s = np.linalg.lstsq(X_centered, Y_centered, rcond=None)
        eigenvalues = np.linalg.eigvals(J.T)
        return eigenvalues
    except np.linalg.LinAlgError:
        return np.full(n_dims, np.nan + 0j)


def bifurcation_indicator(
    signal: np.ndarray,
    window_size: int = 100,
    overlap: float = 0.5
) -> np.ndarray:
    """
    Compute bifurcation indicator over time.

    Detects changes in dynamical behavior that may indicate
    bifurcations (qualitative changes in dynamics).

    Parameters
    ----------
    signal : np.ndarray
        Input time series
    window_size : int
        Size of analysis windows
    overlap : float
        Fraction of overlap between windows

    Returns
    -------
    np.ndarray
        Bifurcation indicator values for each window
    """
    signal = np.asarray(signal).flatten()
    signal = signal[~np.isnan(signal)]
    n = len(signal)

    if n < window_size * 2:
        return np.array([np.nan])

    stride = int(window_size * (1 - overlap))
    n_windows = (n - window_size) // stride + 1

    indicators = []

    for i in range(n_windows):
        start = i * stride
        end = start + window_size
        window = signal[start:end]

        # Compute multiple indicators
        # 1. Variance change
        var = np.var(window)

        # 2. Autocorrelation at lag 1
        if len(window) > 2:
            autocorr = np.corrcoef(window[:-1], window[1:])[0, 1]
        else:
            autocorr = 0

        # 3. Number of local extrema
        diff = np.diff(window)
        extrema = np.sum(diff[:-1] * diff[1:] < 0)
        extrema_rate = extrema / len(window)

        # Combine into single indicator
        # High indicator = potential bifurcation
        indicator = var * (1 + abs(autocorr)) * (1 + extrema_rate)
        indicators.append(indicator)

    return np.array(indicators)


def phase_space_contraction(
    trajectory: np.ndarray,
    dt: float = 1.0
) -> float:
    """
    Compute phase space contraction rate.

    For dissipative systems, this is related to the sum of
    Lyapunov exponents and measures information loss.

    Parameters
    ----------
    trajectory : np.ndarray
        State-space trajectory
    dt : float
        Time step

    Returns
    -------
    float
        Contraction rate (negative = contraction, positive = expansion)
    """
    trajectory = np.asarray(trajectory)

    if trajectory.ndim == 1:
        trajectory = trajectory.reshape(-1, 1)

    n_points, n_dims = trajectory.shape

    if n_points < 10:
        return np.nan

    # Estimate local divergence of flow
    # div(v) = trace(Jacobian)

    divergences = []

    for i in range(1, n_points - 1):
        # Estimate velocity at point i
        v_before = trajectory[i] - trajectory[i - 1]
        v_after = trajectory[i + 1] - trajectory[i]

        # Estimate Jacobian trace (divergence)
        # Using central difference
        if n_dims == 1:
            if abs(v_before[0]) > 1e-10:
                div = (v_after[0] - v_before[0]) / (2 * dt * v_before[0])
                divergences.append(div)
        else:
            # For multi-dimensional, compute trace of velocity gradient
            dv = v_after - v_before
            dx = trajectory[i] - trajectory[i - 1]

            if np.linalg.norm(dx) > 1e-10:
                # Approximate trace
                div = np.dot(dv, dx) / (np.dot(dx, dx) * 2 * dt)
                divergences.append(div)

    if len(divergences) == 0:
        return np.nan

    return float(np.mean(divergences))


def hilbert_stability(y: np.ndarray) -> dict:
    """
    Analyze instantaneous frequency and amplitude stability via Hilbert transform.

    Parameters
    ----------
    y : np.ndarray
        Input signal (1D).

    Returns
    -------
    dict
        inst_freq_mean, inst_freq_std, inst_freq_stability, inst_freq_kurtosis,
        inst_freq_skewness, inst_freq_range, inst_freq_drift, inst_amp_cv,
        inst_amp_trend, phase_coherence, am_fm_ratio.
    """
    from scipy.signal import hilbert

    nan_result = {
        'inst_freq_mean': np.nan, 'inst_freq_std': np.nan,
        'inst_freq_stability': np.nan, 'inst_freq_kurtosis': np.nan,
        'inst_freq_skewness': np.nan, 'inst_freq_range': np.nan,
        'inst_freq_drift': np.nan, 'inst_amp_cv': np.nan,
        'inst_amp_trend': np.nan, 'phase_coherence': np.nan,
        'am_fm_ratio': np.nan,
    }

    y = np.asarray(y, dtype=np.float64).ravel()
    y = y[np.isfinite(y)]
    if len(y) < 10:
        return nan_result

    analytic = hilbert(y)
    inst_amp = np.abs(analytic)
    phase = np.unwrap(np.angle(analytic))
    inst_freq = np.diff(phase) / (2 * np.pi)

    if len(inst_freq) < 3:
        return nan_result

    freq_mean = float(np.mean(inst_freq))
    freq_std = float(np.std(inst_freq))

    if abs(freq_mean) > 1e-12:
        freq_stability = float(np.clip(1.0 - (freq_std / abs(freq_mean)), 0.0, 1.0))
    else:
        freq_stability = 0.0

    if freq_std > 1e-12:
        freq_kurtosis = float(np.mean(((inst_freq - freq_mean) / freq_std) ** 4) - 3.0)
        freq_skewness = float(np.mean(((inst_freq - freq_mean) / freq_std) ** 3))
    else:
        freq_kurtosis = 0.0
        freq_skewness = 0.0

    freq_range = float(np.max(inst_freq) - np.min(inst_freq))
    t = np.arange(len(inst_freq), dtype=np.float64)
    coeffs = np.polyfit(t, inst_freq, 1)
    freq_drift = float(coeffs[0])

    amp_mean = float(np.mean(inst_amp))
    amp_std = float(np.std(inst_amp))
    amp_cv = float(amp_std / amp_mean) if amp_mean > 1e-12 else np.nan

    t_amp = np.arange(len(inst_amp), dtype=np.float64)
    amp_coeffs = np.polyfit(t_amp, inst_amp, 1)
    amp_trend = float(amp_coeffs[0])

    phase_diff = np.diff(phase)
    phase_coherence = float(np.mean(np.cos(phase_diff)))
    am_fm = float(amp_std / freq_std) if freq_std > 1e-12 else np.nan

    return {
        'inst_freq_mean': freq_mean, 'inst_freq_std': freq_std,
        'inst_freq_stability': freq_stability, 'inst_freq_kurtosis': freq_kurtosis,
        'inst_freq_skewness': freq_skewness, 'inst_freq_range': freq_range,
        'inst_freq_drift': freq_drift, 'inst_amp_cv': amp_cv,
        'inst_amp_trend': amp_trend, 'phase_coherence': phase_coherence,
        'am_fm_ratio': am_fm,
    }


def wavelet_stability(y: np.ndarray) -> dict:
    """
    Analyze wavelet energy distribution and stability.

    Parameters
    ----------
    y : np.ndarray
        Input signal (1D).

    Returns
    -------
    dict
        energy_low, energy_mid, energy_high, energy_ratio, entropy,
        concentration, dominant_scale, energy_drift, temporal_std, intermittency.
    """
    nan_result = {
        'energy_low': np.nan, 'energy_mid': np.nan, 'energy_high': np.nan,
        'energy_ratio': np.nan, 'entropy': np.nan, 'concentration': np.nan,
        'dominant_scale': np.nan, 'energy_drift': np.nan,
        'temporal_std': np.nan, 'intermittency': np.nan,
    }

    y = np.asarray(y, dtype=np.float64).ravel()
    y = y[np.isfinite(y)]
    if len(y) < 10:
        return nan_result

    # FFT-based band energy computation (no pywt dependency)
    fft_vals = np.fft.rfft(y)
    power = np.abs(fft_vals) ** 2
    n_bins = len(power)
    if n_bins < 3:
        return nan_result

    b1 = n_bins // 3
    b2 = 2 * n_bins // 3
    energy_low_val = float(np.sum(power[:b1]))
    energy_mid_val = float(np.sum(power[b1:b2]))
    energy_high_val = float(np.sum(power[b2:]))

    total_energy = energy_low_val + energy_mid_val + energy_high_val
    if total_energy < 1e-12:
        return nan_result

    e_low = float(energy_low_val / total_energy)
    e_mid = float(energy_mid_val / total_energy)
    e_high = float(energy_high_val / total_energy)
    e_ratio = float(energy_low_val / energy_high_val) if energy_high_val > 1e-12 else np.nan

    energies = np.array([e_low, e_mid, e_high])
    nonzero = energies[energies > 0]
    entropy = float(-np.sum(nonzero * np.log(nonzero)))
    concentration = float(np.max(energies) / np.sum(energies))
    dominant_scale = int(np.argmax(energies))

    window_size = max(len(y) // 10, 2)
    n_windows = len(y) - window_size + 1
    if n_windows < 2:
        return {
            'energy_low': e_low, 'energy_mid': e_mid, 'energy_high': e_high,
            'energy_ratio': e_ratio, 'entropy': entropy,
            'concentration': concentration, 'dominant_scale': dominant_scale,
            'energy_drift': 0.0, 'temporal_std': 0.0, 'intermittency': 0.0,
        }

    rolling_energy = np.array([np.sum(y[i:i + window_size] ** 2) for i in range(n_windows)])
    t = np.arange(n_windows, dtype=np.float64)
    coeffs = np.polyfit(t, rolling_energy, 1)
    energy_drift = float(coeffs[0])
    temporal_std = float(np.std(rolling_energy))
    re_std = float(np.std(rolling_energy))
    re_mean = float(np.mean(rolling_energy))
    intermittency = float(np.mean(((rolling_energy - re_mean) / re_std) ** 4) - 3.0) if re_std > 1e-12 else 0.0

    return {
        'energy_low': e_low, 'energy_mid': e_mid, 'energy_high': e_high,
        'energy_ratio': e_ratio, 'entropy': entropy,
        'concentration': concentration, 'dominant_scale': dominant_scale,
        'energy_drift': energy_drift, 'temporal_std': temporal_std,
        'intermittency': intermittency,
    }


def detect_collapse(
    effective_dim: np.ndarray,
    threshold_velocity: float = -0.1,
    sustained_fraction: float = 0.3,
    min_collapse_length: int = 5,
) -> dict:
    """
    Detect dimensional collapse in an effective dimension time series.

    Parameters
    ----------
    effective_dim : np.ndarray
        Effective dimension time series (1D).
    threshold_velocity : float
        Velocity threshold below which signal is considered collapsing.
    sustained_fraction : float
        Minimum fraction of series that must be collapsing.
    min_collapse_length : int
        Minimum consecutive samples below threshold.

    Returns
    -------
    dict
        collapse_onset_idx (-1 if none), collapse_onset_fraction (NaN if none).
    """
    _no_collapse = {"collapse_onset_idx": -1, "collapse_onset_fraction": np.nan}

    effective_dim = np.asarray(effective_dim, dtype=np.float64).ravel()
    effective_dim = effective_dim[~np.isnan(effective_dim)]
    n = len(effective_dim)

    if n < min_collapse_length + 1:
        return _no_collapse

    velocity = np.gradient(effective_dim)
    is_collapsing = velocity < threshold_velocity

    sustained_mask = np.zeros(n, dtype=bool)
    run_start = None
    run_length = 0

    for i in range(n):
        if is_collapsing[i]:
            if run_start is None:
                run_start = i
            run_length += 1
        else:
            if run_length >= min_collapse_length:
                sustained_mask[run_start:run_start + run_length] = True
            run_start = None
            run_length = 0

    if run_length >= min_collapse_length and run_start is not None:
        sustained_mask[run_start:run_start + run_length] = True

    total_collapsing = int(np.sum(sustained_mask))
    if total_collapsing / n < sustained_fraction:
        return _no_collapse

    onset_indices = np.where(sustained_mask)[0]
    if len(onset_indices) == 0:
        return _no_collapse

    onset_idx = int(onset_indices[0])
    return {
        "collapse_onset_idx": onset_idx,
        "collapse_onset_fraction": float(onset_idx) / float(n),
    }
