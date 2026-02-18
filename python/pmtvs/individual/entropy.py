"""
Entropy Primitives

Sample entropy, permutation entropy, approximate entropy.
These measure signal complexity and regularity.
"""

import numpy as np
from typing import Optional
from math import factorial

from pmtvs._dispatch import dispatch


def _sample_entropy_py(
    signal: np.ndarray,
    m: int = 2,
    r: Optional[float] = None
) -> float:
    """
    Compute sample entropy (Python implementation).

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


def _permutation_entropy_py(
    signal: np.ndarray,
    order: int = 3,
    delay: int = 1,
    normalize: bool = True
) -> float:
    """
    Compute permutation entropy (Python implementation).

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


def _approximate_entropy_py(
    signal: np.ndarray,
    m: int = 2,
    r: Optional[float] = None
) -> float:
    """
    Compute approximate entropy (Python implementation).

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


# Dispatch wrappers
sample_entropy = dispatch("sample_entropy", _sample_entropy_py)
permutation_entropy = dispatch("permutation_entropy", _permutation_entropy_py)
approximate_entropy = dispatch("approximate_entropy", _approximate_entropy_py)
