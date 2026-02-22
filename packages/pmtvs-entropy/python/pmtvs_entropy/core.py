"""
Entropy Primitives

Sample entropy, permutation entropy, approximate entropy.
These measure signal complexity and regularity.
"""

import numpy as np
from typing import Optional
from math import factorial

from pmtvs_entropy._dispatch import use_rust


def sample_entropy(
    signal: np.ndarray,
    m: int = 2,
    r: Optional[float] = None
) -> float:
    """
    Compute sample entropy.

    Parameters
    ----------
    signal : np.ndarray
        Input signal
    m : int
        Embedding dimension
    r : float, optional
        Tolerance (default: 0.2 * std)

    Returns
    -------
    float
        Sample entropy (higher = more complex/random)
    """
    # Input normalization and edge guards (before any dispatch)
    signal = np.asarray(signal, dtype=np.float64).flatten()
    signal = signal[~np.isnan(signal)]
    n = len(signal)

    if r is None:
        r = 0.2 * np.std(signal)

    if r == 0 or n < m + 2:
        return np.nan

    # Check for Rust backend
    if use_rust('sample_entropy'):
        from pmtvs_entropy import _get_rust
        rust_fn = _get_rust('sample_entropy')
        if rust_fn is not None:
            return rust_fn(signal, m, r)

    # Python implementation

    def count_matches(dim):
        count = 0
        templates = np.array([signal[i:i+dim] for i in range(n - dim)])

        for i in range(len(templates)):
            for j in range(i + 1, len(templates)):
                if np.max(np.abs(templates[i] - templates[j])) < r:
                    count += 1
        return count

    B = count_matches(m)
    A = count_matches(m + 1)

    if B == 0:
        return np.nan

    return -np.log(A / B) if A > 0 else np.nan


def permutation_entropy(
    signal: np.ndarray,
    order: int = 3,
    delay: int = 1,
    normalize: bool = True
) -> float:
    """
    Compute permutation entropy.

    Parameters
    ----------
    signal : np.ndarray
        Input signal
    order : int
        Embedding dimension (pattern length)
    delay : int
        Time delay
    normalize : bool
        If True, normalize to [0, 1]

    Returns
    -------
    float
        Permutation entropy
    """
    signal = np.asarray(signal).flatten()
    n = len(signal)

    if n < order * delay:
        return np.nan

    # Build ordinal patterns
    n_patterns = n - (order - 1) * delay
    patterns = np.zeros((n_patterns, order))

    for i in range(n_patterns):
        for j in range(order):
            patterns[i, j] = signal[i + j * delay]

    # Get permutation indices
    perms = np.argsort(patterns, axis=1)

    # Count unique permutations
    perm_strings = [''.join(map(str, p)) for p in perms]
    unique, counts = np.unique(perm_strings, return_counts=True)

    # Compute entropy
    probs = counts / counts.sum()
    entropy = -np.sum(probs * np.log2(probs))

    if normalize:
        max_entropy = np.log2(factorial(order))
        if max_entropy > 0:
            entropy = entropy / max_entropy

    return float(entropy)


def approximate_entropy(
    signal: np.ndarray,
    m: int = 2,
    r: Optional[float] = None
) -> float:
    """
    Compute approximate entropy.

    Parameters
    ----------
    signal : np.ndarray
        Input signal
    m : int
        Embedding dimension
    r : float, optional
        Tolerance (default: 0.2 * std)

    Returns
    -------
    float
        Approximate entropy
    """
    signal = np.asarray(signal).flatten()
    n = len(signal)

    if r is None:
        r = 0.2 * np.std(signal)

    if r == 0 or n < m + 2:
        return np.nan

    def phi(dim):
        templates = np.array([signal[i:i+dim] for i in range(n - dim + 1)])
        C = np.zeros(len(templates))

        for i in range(len(templates)):
            for j in range(len(templates)):
                if np.max(np.abs(templates[i] - templates[j])) < r:
                    C[i] += 1

        C = C / len(templates)
        C[C == 0] = np.finfo(float).eps  # Avoid log(0)
        return np.mean(np.log(C))

    return float(phi(m) - phi(m + 1))


def multiscale_entropy(
    signal: np.ndarray,
    m: int = 2,
    r: Optional[float] = None,
    max_scale: int = 20
) -> np.ndarray:
    """
    Compute multiscale entropy (MSE).

    Coarse-grains the signal at multiple scales and computes sample
    entropy at each scale.

    Parameters
    ----------
    signal : np.ndarray
        Input signal
    m : int
        Embedding dimension
    r : float, optional
        Tolerance (default: 0.2 * std of original signal)
    max_scale : int
        Maximum scale factor

    Returns
    -------
    np.ndarray
        Sample entropy values at each scale (1 to max_scale)
    """
    signal = np.asarray(signal, dtype=np.float64).flatten()
    signal = signal[~np.isnan(signal)]
    n = len(signal)

    if r is None:
        r = 0.2 * np.std(signal)

    max_scale = min(max_scale, n // (m + 2))
    if max_scale < 1:
        return np.array([np.nan])

    mse = np.zeros(max_scale)

    for scale in range(1, max_scale + 1):
        # Coarse-grain
        n_coarse = n // scale
        coarse = np.array([
            np.mean(signal[i * scale:(i + 1) * scale])
            for i in range(n_coarse)
        ])
        mse[scale - 1] = sample_entropy(coarse, m=m, r=r)

    return mse


def lempel_ziv_complexity(
    signal: np.ndarray,
    threshold: Optional[float] = None
) -> float:
    """
    Compute Lempel-Ziv complexity.

    Binarizes the signal and counts the number of distinct subsequences.

    Parameters
    ----------
    signal : np.ndarray
        Input signal
    threshold : float, optional
        Binarization threshold (default: median)

    Returns
    -------
    float
        Normalized Lempel-Ziv complexity in [0, 1]
    """
    signal = np.asarray(signal, dtype=np.float64).flatten()
    signal = signal[~np.isnan(signal)]
    n = len(signal)

    if n < 2:
        return np.nan

    if threshold is None:
        threshold = np.median(signal)

    # Binarize
    binary = ''.join('1' if x > threshold else '0' for x in signal)

    # Lempel-Ziv parsing
    i = 0
    c = 1  # complexity counter
    l = 1  # current prefix length

    while i + l <= n:
        substring = binary[i:i + l]
        prior = binary[:i + l - 1]

        if substring in prior:
            l += 1
        else:
            c += 1
            i += l
            l = 1

    # Normalize by theoretical upper bound
    if n > 0:
        b = n / np.log2(n) if n > 1 else 1
        return float(c / b)
    return np.nan


def entropy_rate(
    signal: np.ndarray,
    m_max: int = 10,
    r: Optional[float] = None
) -> float:
    """
    Estimate entropy rate from sample entropy convergence.

    Computes sample entropy at increasing embedding dimensions and
    estimates the asymptotic rate.

    Parameters
    ----------
    signal : np.ndarray
        Input signal
    m_max : int
        Maximum embedding dimension to try
    r : float, optional
        Tolerance (default: 0.2 * std)

    Returns
    -------
    float
        Estimated entropy rate
    """
    signal = np.asarray(signal, dtype=np.float64).flatten()
    signal = signal[~np.isnan(signal)]
    n = len(signal)

    if r is None:
        r = 0.2 * np.std(signal)

    if n < m_max + 2 or r == 0:
        return np.nan

    se_values = []
    for m in range(1, m_max + 1):
        if n < m + 2:
            break
        se = sample_entropy(signal, m=m, r=r)
        if np.isfinite(se):
            se_values.append(se)
        else:
            break

    if len(se_values) < 2:
        return np.nan

    # Entropy rate approximated by SE at highest valid m
    return float(se_values[-1])
