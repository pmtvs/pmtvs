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
    # Check for Rust backend
    if use_rust('sample_entropy'):
        from pmtvs_entropy import _get_rust
        rust_fn = _get_rust('sample_entropy')
        if rust_fn is not None:
            if r is None:
                r = 0.2 * np.std(signal)
            return rust_fn(signal, m, r)

    # Python implementation
    signal = np.asarray(signal).flatten()
    n = len(signal)

    if r is None:
        r = 0.2 * np.std(signal)

    if r == 0 or n < m + 2:
        return np.nan

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
