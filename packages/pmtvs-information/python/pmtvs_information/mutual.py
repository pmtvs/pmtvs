"""Mutual information variants."""
import numpy as np


def _joint_entropy(x, y, bins, base=2):
    hist, _, _ = np.histogram2d(x, y, bins=bins)
    p = hist / hist.sum()
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)) / np.log(base))


def _entropy(x, bins, base=2):
    counts, _ = np.histogram(x, bins=bins)
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)) / np.log(base))


def conditional_mutual_information(x, y, z, bins=None, base=2):
    """I(X;Y|Z) = H(X,Z) + H(Y,Z) - H(X,Y,Z) - H(Z)."""
    x = np.asarray(x, dtype=np.float64).flatten()
    y = np.asarray(y, dtype=np.float64).flatten()
    z = np.asarray(z, dtype=np.float64).flatten()
    n = min(len(x), len(y), len(z))
    x, y, z = x[:n], y[:n], z[:n]
    if bins is None:
        bins = max(8, int(n ** (1/3)))
    h_xz = _joint_entropy(x, z, bins, base)
    h_yz = _joint_entropy(y, z, bins, base)
    h_z = _entropy(z, bins, base)
    xyz = np.column_stack([x, y, z])
    hist, _ = np.histogramdd(xyz, bins=bins)
    p = hist / hist.sum()
    p = p[p > 0]
    h_xyz = float(-np.sum(p * np.log(p)) / np.log(base))
    return float(h_xz + h_yz - h_xyz - h_z)


def multivariate_mutual_information(variables, bins=None, base=2):
    """Co-information for 3+ variables."""
    variables = [np.asarray(v, dtype=np.float64).flatten() for v in variables]
    n = min(len(v) for v in variables)
    variables = [v[:n] for v in variables]
    if bins is None:
        bins = max(8, int(n ** (1/3)))
    k = len(variables)
    if k < 2:
        return np.nan
    if k == 2:
        h_x = _entropy(variables[0], bins, base)
        h_y = _entropy(variables[1], bins, base)
        h_xy = _joint_entropy(variables[0], variables[1], bins, base)
        return float(h_x + h_y - h_xy)
    # For 3 variables: I(X;Y;Z) = I(X;Y) - I(X;Y|Z)
    h_x = _entropy(variables[0], bins, base)
    h_y = _entropy(variables[1], bins, base)
    h_xy = _joint_entropy(variables[0], variables[1], bins, base)
    mi_xy = h_x + h_y - h_xy
    cmi = conditional_mutual_information(variables[0], variables[1], variables[2], bins, base)
    return float(mi_xy - cmi)


def total_correlation(variables, bins=None, base=2):
    """TC = sum(H(Xi)) - H(X1,...,Xn)."""
    variables = [np.asarray(v, dtype=np.float64).flatten() for v in variables]
    n = min(len(v) for v in variables)
    variables = [v[:n] for v in variables]
    if bins is None:
        bins = max(8, int(n ** (1/3)))
    sum_h = sum(_entropy(v, bins, base) for v in variables)
    data = np.column_stack(variables)
    hist, _ = np.histogramdd(data, bins=bins)
    p = hist / hist.sum()
    p = p[p > 0]
    h_joint = float(-np.sum(p * np.log(p)) / np.log(base))
    return float(sum_h - h_joint)


def interaction_information(variables, bins=None, base=2):
    """Alias for multivariate mutual information (co-information)."""
    return multivariate_mutual_information(variables, bins, base)


def dual_total_correlation(variables, bins=None, base=2):
    """DTC = H(X1,...,Xn) - sum(H(Xi|X_{-i}))."""
    variables = [np.asarray(v, dtype=np.float64).flatten() for v in variables]
    n = min(len(v) for v in variables)
    variables = [v[:n] for v in variables]
    if bins is None:
        bins = max(8, int(n ** (1/3)))
    data = np.column_stack(variables)
    hist, _ = np.histogramdd(data, bins=bins)
    p = hist / hist.sum()
    p = p[p > 0]
    h_joint = float(-np.sum(p * np.log(p)) / np.log(base))
    sum_cond = 0.0
    k = len(variables)
    for i in range(k):
        others = [variables[j] for j in range(k) if j != i]
        if len(others) == 1:
            h_others = _entropy(others[0], bins, base)
        else:
            others_data = np.column_stack(others)
            hist_o, _ = np.histogramdd(others_data, bins=bins)
            p_o = hist_o / hist_o.sum()
            p_o = p_o[p_o > 0]
            h_others = float(-np.sum(p_o * np.log(p_o)) / np.log(base))
        h_xi = _entropy(variables[i], bins, base)
        sum_cond += h_joint - h_others
    return float(h_joint - sum_cond)
