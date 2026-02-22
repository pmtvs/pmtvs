"""
Dynamic Mode Decomposition Functions

DMD for analyzing spatiotemporal dynamics in multivariate signals.
"""

import numpy as np
from typing import Tuple, Optional


def dynamic_mode_decomposition(
    signals: np.ndarray,
    dt: float = 1.0,
    rank: Optional[int] = None,
    exact: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Dynamic Mode Decomposition.

    Parameters
    ----------
    signals : np.ndarray
        Data matrix (n_samples x n_signals) with time series in columns.
    dt : float
        Time step between samples.
    rank : int, optional
        Truncation rank (default: min(n_samples - 1, n_signals)).
    exact : bool
        If True, compute exact DMD modes.

    Returns
    -------
    modes : np.ndarray
        DMD modes (n_signals x rank), spatial patterns.
    eigenvalues : np.ndarray
        DMD eigenvalues (rank,), complex growth/decay + frequency.
    dynamics : np.ndarray
        Time dynamics (rank x n_samples), temporal evolution.
    amplitudes : np.ndarray
        Mode amplitudes (rank,), initial mode contributions.

    Notes
    -----
    DMD approximates: x_{k+1} = A @ x_k where A is the best-fit
    linear operator.

    Eigenvalue interpretation:

    - |lambda| > 1: growing mode
    - |lambda| < 1: decaying mode
    - |lambda| = 1: neutral mode
    - angle(lambda): oscillation frequency (radians per time step)

    Continuous-time eigenvalue: omega = log(lambda) / dt

    - real(omega): growth rate
    - imag(omega): frequency (rad/s)
    """
    signals = np.asarray(signals)

    if signals.ndim == 1:
        signals = signals.reshape(-1, 1)

    # Filter non-finite values (rows with any inf/nan)
    finite_mask = np.all(np.isfinite(signals), axis=1)
    signals = signals[finite_mask]

    n_samples, n_signals = signals.shape

    if n_samples < 3:
        return (
            np.full((n_signals, 1), np.nan),
            np.array([np.nan + 0j]),
            np.full((1, n_samples), np.nan),
            np.array([np.nan])
        )

    # Build time-shifted matrices
    # X = [x_0, x_1, ..., x_{n-2}]
    # X' = [x_1, x_2, ..., x_{n-1}]
    X = signals[:-1, :].T        # (n_signals x n_samples-1)
    Xprime = signals[1:, :].T    # (n_signals x n_samples-1)

    # SVD of X
    if rank is None:
        rank = min(n_samples - 1, n_signals)

    try:
        U, S, Vh = np.linalg.svd(X, full_matrices=False)
    except np.linalg.LinAlgError:
        return (
            np.full((n_signals, rank), np.nan),
            np.full(rank, np.nan + 0j),
            np.full((rank, n_samples), np.nan),
            np.full(rank, np.nan)
        )

    # Truncate to rank
    r = min(rank, len(S))
    Ur = U[:, :r]
    Sr = S[:r]
    Vr = Vh[:r, :]

    # Build reduced operator: Atilde = Ur.T @ Xprime @ Vr.T @ inv(diag(Sr))
    Sr_inv = 1.0 / (Sr + 1e-10)
    Atilde = Ur.T @ Xprime @ Vr.T @ np.diag(Sr_inv)

    # Eigendecomposition of Atilde
    eigenvalues, W = np.linalg.eig(Atilde)

    # DMD modes
    if exact:
        # Exact DMD: Phi = Xprime @ Vr.T @ diag(Sr_inv) @ W
        modes = Xprime @ Vr.T @ np.diag(Sr_inv) @ W
    else:
        # Projected DMD: Phi = Ur @ W
        modes = Ur @ W

    # Initial amplitudes (least squares fit to first snapshot)
    x0 = signals[0, :]
    try:
        amplitudes = np.linalg.lstsq(modes, x0, rcond=None)[0]
    except np.linalg.LinAlgError:
        amplitudes = np.zeros(r, dtype=complex)

    # Time dynamics
    time_steps = np.arange(n_samples)
    dynamics = np.zeros((r, n_samples), dtype=complex)
    for i, (lamb, amp) in enumerate(zip(eigenvalues, amplitudes)):
        dynamics[i, :] = amp * (lamb ** time_steps)

    return modes, eigenvalues, dynamics, amplitudes


def dmd_frequencies(
    eigenvalues: np.ndarray,
    dt: float = 1.0
) -> np.ndarray:
    """
    Extract continuous-time frequencies from DMD eigenvalues.

    Parameters
    ----------
    eigenvalues : np.ndarray
        DMD eigenvalues (complex).
    dt : float
        Time step.

    Returns
    -------
    np.ndarray
        Frequencies in Hz.

    Notes
    -----
    omega = log(lambda) / dt
    frequency = imag(omega) / (2 * pi)
    """
    eigenvalues = np.asarray(eigenvalues)
    omega = np.log(eigenvalues + 1e-10) / dt
    frequencies = np.imag(omega) / (2 * np.pi)
    return np.abs(frequencies)


def dmd_growth_rates(
    eigenvalues: np.ndarray,
    dt: float = 1.0
) -> np.ndarray:
    """
    Extract growth rates from DMD eigenvalues.

    Parameters
    ----------
    eigenvalues : np.ndarray
        DMD eigenvalues (complex).
    dt : float
        Time step.

    Returns
    -------
    np.ndarray
        Growth rates (positive = growing, negative = decaying).

    Notes
    -----
    omega = log(lambda) / dt
    growth_rate = real(omega)
    """
    eigenvalues = np.asarray(eigenvalues)
    omega = np.log(eigenvalues + 1e-10) / dt
    return np.real(omega)


def dmd_decompose(y: np.ndarray, delay: int = 10, dt: float = 1.0) -> dict:
    """
    DMD summary for a 1D signal via delay embedding.

    Delay-embeds the signal, runs DMD, and extracts summary statistics.

    Parameters
    ----------
    y : np.ndarray
        Input signal (1D).
    delay : int
        Embedding dimension for delay matrix.
    dt : float
        Time step.

    Returns
    -------
    dict
        dominant_freq, growth_rate, is_stable, n_modes.
    """
    nan_result = {
        'dominant_freq': np.nan, 'growth_rate': np.nan,
        'is_stable': np.nan, 'n_modes': np.nan,
    }

    y = np.asarray(y, dtype=np.float64).ravel()
    y = y[np.isfinite(y)]
    n = len(y)

    if n < delay + 2:
        return nan_result

    # Build delay-embedded matrix (n_samples x delay)
    n_rows = n - delay + 1
    embedded = np.empty((n_rows, delay))
    for i in range(n_rows):
        embedded[i, :] = y[i:i + delay]

    try:
        modes, eigenvalues, dynamics, amplitudes = dynamic_mode_decomposition(
            embedded, dt=dt
        )
    except Exception:
        return nan_result

    if np.any(np.isnan(eigenvalues)):
        return nan_result

    freqs = dmd_frequencies(eigenvalues, dt=dt)
    growth = dmd_growth_rates(eigenvalues, dt=dt)

    # Dominant mode by amplitude
    amp_mag = np.abs(amplitudes)
    if len(amp_mag) == 0 or np.all(np.isnan(amp_mag)):
        return nan_result

    dom_idx = int(np.argmax(amp_mag))
    dominant_freq = float(freqs[dom_idx])
    growth_rate = float(growth[dom_idx])
    is_stable = float(np.all(np.abs(eigenvalues) <= 1.0 + 1e-10))
    n_modes = int(len(eigenvalues))

    return {
        'dominant_freq': dominant_freq, 'growth_rate': growth_rate,
        'is_stable': is_stable, 'n_modes': float(n_modes),
    }
