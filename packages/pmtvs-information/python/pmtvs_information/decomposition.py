"""Partial Information Decomposition."""
import numpy as np
from typing import Dict


def _mutual_information(x, y, n_bins, base=2):
    x = np.asarray(x, dtype=np.float64).flatten()
    y = np.asarray(y, dtype=np.float64).flatten()
    n = min(len(x), len(y))
    x, y = x[:n], y[:n]
    hist_xy, _, _ = np.histogram2d(x, y, bins=n_bins)
    p_xy = hist_xy / hist_xy.sum()
    p_x = np.sum(p_xy, axis=1)
    p_y = np.sum(p_xy, axis=0)
    mi = 0.0
    for i in range(n_bins):
        for j in range(n_bins):
            if p_xy[i, j] > 0 and p_x[i] > 0 and p_y[j] > 0:
                mi += p_xy[i, j] * np.log(p_xy[i, j] / (p_x[i] * p_y[j]))
    return float(mi / np.log(base))


def _joint_mutual_information(s1, s2, target, n_bins, base=2):
    s1 = np.asarray(s1, dtype=np.float64).flatten()
    s2 = np.asarray(s2, dtype=np.float64).flatten()
    target = np.asarray(target, dtype=np.float64).flatten()
    n = min(len(s1), len(s2), len(target))
    s1, s2, target = s1[:n], s2[:n], target[:n]
    joint_source = s1 * n_bins + s2
    return _mutual_information(joint_source, target, n_bins, base)


def partial_information_decomposition(source_1, source_2, target, n_bins=8) -> Dict:
    """PID: decompose I(S1,S2;T) into redundancy, unique1, unique2, synergy."""
    mi_s1_t = _mutual_information(source_1, target, n_bins)
    mi_s2_t = _mutual_information(source_2, target, n_bins)
    mi_joint = _joint_mutual_information(source_1, source_2, target, n_bins)
    red = min(mi_s1_t, mi_s2_t)
    unique_1 = mi_s1_t - red
    unique_2 = mi_s2_t - red
    syn = mi_joint - unique_1 - unique_2 - red
    return {
        'redundancy': float(max(0, red)),
        'unique_1': float(max(0, unique_1)),
        'unique_2': float(max(0, unique_2)),
        'synergy': float(max(0, syn)),
        'total_mi': float(mi_joint),
    }


def redundancy(sources, target, n_bins=8) -> float:
    """Redundant information (min MI)."""
    mis = [_mutual_information(s, target, n_bins) for s in sources]
    return float(min(mis)) if mis else np.nan


def synergy(sources, target, n_bins=8) -> float:
    """Synergistic information."""
    if len(sources) < 2:
        return np.nan
    pid = partial_information_decomposition(sources[0], sources[1], target, n_bins)
    return float(pid['synergy'])


def information_atoms(sources, target, n_bins=8) -> Dict:
    """Compute all information atoms."""
    if len(sources) < 2:
        return {'redundancy': np.nan, 'synergy': np.nan, 'total_mi': np.nan, 'emergence_ratio': np.nan}
    pid = partial_information_decomposition(sources[0], sources[1], target, n_bins)
    total = pid['total_mi']
    emergence = pid['synergy'] / total if total > 0 else 0.0
    return {
        'redundancy': pid['redundancy'],
        'synergy': pid['synergy'],
        'total_mi': total,
        'emergence_ratio': float(emergence),
    }
