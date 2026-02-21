"""
Lyapunov divergence curve utilities.

Shared by both lyapunov_rosenstein and largest_lyapunov_exponent.
The core function detects the initial linear region of a Rosenstein
divergence curve and fits the slope only to that region.
"""

import numpy as np
from typing import Tuple


def fit_linear_region(
    divergence_curve: np.ndarray,
    iterations: np.ndarray = None,
    min_pts: int = 4,
    r2_threshold: float = 0.98,
) -> Tuple[float, int, int, float]:
    """
    Detect and fit the initial linear region of a Rosenstein divergence curve.

    The Rosenstein method produces a curve of mean log-divergence vs iteration.
    The largest Lyapunov exponent is the slope of the initial linear portion.
    After a few iterations, the curve saturates (neighbors diverge to attractor
    diameter) and the slope approaches zero. Fitting the full curve averages
    the true exponent with this saturation, systematically underestimating it.

    This function grows the fit window from iteration 0, stopping when the
    linear fit quality (R²) drops below the threshold. This captures the
    steepest initial portion before curvature sets in.

    Parameters
    ----------
    divergence_curve : array
        Mean log-divergence at each iteration step. Output of Rosenstein's
        algorithm (the second element of lyapunov_rosenstein return tuple).
    iterations : array, optional
        Iteration indices corresponding to divergence_curve.
        Default: np.arange(len(divergence_curve)).
    min_pts : int
        Minimum number of points to fit. Default 4.
    r2_threshold : float
        Stop growing the window when R² drops below this. Default 0.98.
        Higher values (0.99) give steeper slopes from fewer points.
        Lower values (0.95) give more stable but potentially lower estimates.

    Returns
    -------
    slope : float
        Estimated Lyapunov exponent (slope of linear region).
    fit_start : int
        Start index of the fit region (always 0).
    fit_end : int
        End index (exclusive) of the fit region.
    r_squared : float
        R² of the linear fit in the selected region.
    """
    div = np.asarray(divergence_curve, dtype=np.float64)

    if iterations is None:
        iters = np.arange(len(div), dtype=np.float64)
    else:
        iters = np.asarray(iterations, dtype=np.float64)

    # Remove NaN/inf entries
    mask = np.isfinite(div)
    div = div[mask]
    iters = iters[mask]
    n = len(div)

    if n < 2:
        return np.nan, 0, 0, 0.0

    if n < min_pts:
        # Not enough points for min_pts window — fit what we have
        slope = np.polyfit(iters, div, 1)[0]
        return float(slope), 0, n, 0.0

    # Grow window from min_pts, keep extending while R² stays above threshold
    best_slope = None
    best_end = min_pts
    best_r2 = 0.0

    for end in range(min_pts, n + 1):
        x = iters[:end]
        y = div[:end]
        coeffs = np.polyfit(x, y, 1)
        slope = coeffs[0]
        fitted = np.polyval(coeffs, x)
        ss_res = np.sum((y - fitted) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-20 else 0.0

        if r2 >= r2_threshold:
            best_slope = slope
            best_end = end
            best_r2 = r2
        else:
            # R² dropped below threshold — linear region ended
            break

    if best_slope is None:
        # Never found acceptable linearity — use minimum window
        x = iters[:min_pts]
        y = div[:min_pts]
        best_slope = np.polyfit(x, y, 1)[0]
        best_end = min_pts
        best_r2 = 0.0

    return float(best_slope), 0, best_end, float(best_r2)
