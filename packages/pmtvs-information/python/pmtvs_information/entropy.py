"""Entropy variant primitives."""
import numpy as np


def _estimate_probabilities(data, bins=None):
    data = np.asarray(data, dtype=np.float64).flatten()
    data = data[~np.isnan(data)]
    if len(data) < 2:
        return np.array([])
    if bins is None:
        bins = max(10, int(np.sqrt(len(data))))
    counts, _ = np.histogram(data, bins=bins)
    probs = counts / counts.sum()
    return probs[probs > 0]


def shannon_entropy(data, bins=None, base=2):
    """Shannon entropy H(X) = -sum(p * log(p))."""
    p = _estimate_probabilities(data, bins)
    if len(p) == 0:
        return np.nan
    return float(-np.sum(p * np.log(p)) / np.log(base))


def renyi_entropy(data, alpha=2.0, bins=None, base=2):
    """Renyi entropy of order alpha."""
    p = _estimate_probabilities(data, bins)
    if len(p) == 0:
        return np.nan
    if alpha == 1.0:
        return shannon_entropy(data, bins, base)
    return float(np.log(np.sum(p ** alpha)) / ((1 - alpha) * np.log(base)))


def tsallis_entropy(data, q=2.0, bins=None):
    """Tsallis entropy."""
    p = _estimate_probabilities(data, bins)
    if len(p) == 0:
        return np.nan
    if q == 1.0:
        return float(-np.sum(p * np.log(p)))
    return float((1 - np.sum(p ** q)) / (q - 1))
