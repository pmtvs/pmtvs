"""Divergence and distance between distributions."""
import numpy as np


def _to_distribution(data, bins=None):
    data = np.asarray(data, dtype=np.float64).flatten()
    data = data[~np.isnan(data)]
    if len(data) < 2:
        return np.array([])
    if bins is None:
        bins = max(10, int(np.sqrt(len(data))))
    counts, _ = np.histogram(data, bins=bins)
    p = counts / counts.sum()
    return p


def cross_entropy(p, q, bins=None, base=2):
    """Cross entropy H(P, Q) = -sum(p * log(q))."""
    p_dist = _to_distribution(p, bins) if np.ndim(p) == 1 and len(p) > 20 else np.asarray(p, dtype=np.float64)
    q_dist = _to_distribution(q, bins) if np.ndim(q) == 1 and len(q) > 20 else np.asarray(q, dtype=np.float64)
    if len(p_dist) == 0 or len(q_dist) == 0:
        return np.nan
    min_len = min(len(p_dist), len(q_dist))
    p_dist, q_dist = p_dist[:min_len], q_dist[:min_len]
    p_dist = p_dist / (p_dist.sum() + 1e-12)
    q_dist = q_dist / (q_dist.sum() + 1e-12)
    q_dist = np.maximum(q_dist, 1e-12)
    mask = p_dist > 0
    return float(-np.sum(p_dist[mask] * np.log(q_dist[mask])) / np.log(base))


def hellinger_distance(p, q, bins=None):
    """Hellinger distance between distributions."""
    p_dist = _to_distribution(p, bins) if np.ndim(p) == 1 and len(p) > 20 else np.asarray(p, dtype=np.float64)
    q_dist = _to_distribution(q, bins) if np.ndim(q) == 1 and len(q) > 20 else np.asarray(q, dtype=np.float64)
    if len(p_dist) == 0 or len(q_dist) == 0:
        return np.nan
    min_len = min(len(p_dist), len(q_dist))
    p_dist, q_dist = p_dist[:min_len], q_dist[:min_len]
    p_dist = p_dist / (p_dist.sum() + 1e-12)
    q_dist = q_dist / (q_dist.sum() + 1e-12)
    return float(np.sqrt(np.sum((np.sqrt(p_dist) - np.sqrt(q_dist)) ** 2)) / np.sqrt(2))


def total_variation_distance(p, q, bins=None):
    """Total variation distance TV(P,Q) = 0.5 * sum(|p-q|)."""
    p_dist = _to_distribution(p, bins) if np.ndim(p) == 1 and len(p) > 20 else np.asarray(p, dtype=np.float64)
    q_dist = _to_distribution(q, bins) if np.ndim(q) == 1 and len(q) > 20 else np.asarray(q, dtype=np.float64)
    if len(p_dist) == 0 or len(q_dist) == 0:
        return np.nan
    min_len = min(len(p_dist), len(q_dist))
    p_dist, q_dist = p_dist[:min_len], q_dist[:min_len]
    p_dist = p_dist / (p_dist.sum() + 1e-12)
    q_dist = q_dist / (q_dist.sum() + 1e-12)
    return float(0.5 * np.sum(np.abs(p_dist - q_dist)))
