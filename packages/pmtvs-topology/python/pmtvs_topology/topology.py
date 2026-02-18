"""
Topological Data Analysis Functions
"""

import numpy as np
from typing import List, Tuple, Optional


def distance_matrix(points: np.ndarray, metric: str = "euclidean") -> np.ndarray:
    """
    Compute pairwise distance matrix.

    Parameters
    ----------
    points : np.ndarray
        Point cloud (n_points, n_dims)
    metric : str
        Distance metric

    Returns
    -------
    np.ndarray
        Distance matrix
    """
    points = np.asarray(points)
    if points.ndim == 1:
        points = points.reshape(-1, 1)

    n = len(points)
    dist = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            if metric == "euclidean":
                d = np.linalg.norm(points[i] - points[j])
            elif metric == "manhattan":
                d = np.sum(np.abs(points[i] - points[j]))
            else:
                d = np.linalg.norm(points[i] - points[j])
            dist[i, j] = d
            dist[j, i] = d

    return dist


def persistent_homology_0d(
    points: np.ndarray
) -> List[Tuple[float, float]]:
    """
    Compute 0-dimensional persistent homology (connected components).

    Uses single-linkage clustering to compute birth-death pairs.

    Parameters
    ----------
    points : np.ndarray
        Point cloud

    Returns
    -------
    list
        List of (birth, death) tuples for each feature
    """
    points = np.asarray(points)
    if points.ndim == 1:
        points = points.reshape(-1, 1)

    n = len(points)
    if n < 2:
        return [(0.0, np.inf)]

    dist = distance_matrix(points)

    # Union-Find for single-linkage clustering
    parent = list(range(n))
    rank = [0] * n
    birth = [0.0] * n

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y, death_time):
        px, py = find(x), find(y)
        if px == py:
            return None

        # Younger component dies
        if rank[px] < rank[py]:
            px, py = py, px
        parent[py] = px
        if rank[px] == rank[py]:
            rank[px] += 1

        return (birth[py], death_time)

    # Get all edges sorted by distance
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((dist[i, j], i, j))
    edges.sort()

    # Process edges
    persistence = []
    for d, i, j in edges:
        result = union(i, j, d)
        if result is not None:
            persistence.append(result)

    # Add one infinite persistence (final component)
    persistence.append((0.0, np.inf))

    return persistence


def betti_numbers(
    persistence: List[Tuple[float, float]],
    threshold: float
) -> int:
    """
    Compute Betti number at a given threshold.

    Parameters
    ----------
    persistence : list
        Persistence pairs (birth, death)
    threshold : float
        Filtration threshold

    Returns
    -------
    int
        Number of features alive at threshold
    """
    return sum(1 for b, d in persistence if b <= threshold < d)


def persistence_entropy(
    persistence: List[Tuple[float, float]]
) -> float:
    """
    Compute persistence entropy.

    Parameters
    ----------
    persistence : list
        Persistence pairs (birth, death)

    Returns
    -------
    float
        Shannon entropy of persistence lengths
    """
    # Filter out infinite persistence
    lifetimes = [d - b for b, d in persistence if np.isfinite(d) and d > b]

    if len(lifetimes) == 0:
        return 0.0

    total = sum(lifetimes)
    if total == 0:
        return 0.0

    p = np.array(lifetimes) / total
    return float(-np.sum(p * np.log(p + 1e-12)))


def persistence_landscape(
    persistence: List[Tuple[float, float]],
    k: int = 1,
    resolution: int = 100
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute k-th persistence landscape.

    Parameters
    ----------
    persistence : list
        Persistence pairs
    k : int
        Landscape index (1-indexed)
    resolution : int
        Number of sample points

    Returns
    -------
    tuple
        (x_values, landscape_values)
    """
    # Filter finite persistence
    pairs = [(b, d) for b, d in persistence if np.isfinite(d)]

    if len(pairs) == 0:
        return np.array([0, 1]), np.array([0, 0])

    # Determine range
    all_values = [b for b, d in pairs] + [d for b, d in pairs]
    x_min = min(all_values)
    x_max = max(all_values)

    if x_min == x_max:
        return np.array([x_min, x_max + 1]), np.array([0, 0])

    x = np.linspace(x_min, x_max, resolution)
    landscapes = []

    for t in x:
        # Tent function values at t
        values = []
        for b, d in pairs:
            if b <= t <= (b + d) / 2:
                values.append(t - b)
            elif (b + d) / 2 <= t <= d:
                values.append(d - t)
            else:
                values.append(0)

        # k-th largest
        values.sort(reverse=True)
        if k <= len(values):
            landscapes.append(values[k - 1])
        else:
            landscapes.append(0)

    return x, np.array(landscapes)


def bottleneck_distance(
    persistence1: List[Tuple[float, float]],
    persistence2: List[Tuple[float, float]]
) -> float:
    """
    Compute approximate bottleneck distance between persistence diagrams.

    Parameters
    ----------
    persistence1 : list
        First persistence diagram
    persistence2 : list
        Second persistence diagram

    Returns
    -------
    float
        Bottleneck distance (approximation)
    """
    # Filter finite pairs
    p1 = [(b, d) for b, d in persistence1 if np.isfinite(d)]
    p2 = [(b, d) for b, d in persistence2 if np.isfinite(d)]

    if len(p1) == 0 and len(p2) == 0:
        return 0.0

    # Simple approximation: match by persistence
    pers1 = sorted([d - b for b, d in p1], reverse=True)
    pers2 = sorted([d - b for b, d in p2], reverse=True)

    # Pad shorter list with zeros (diagonal matches)
    max_len = max(len(pers1), len(pers2))
    pers1.extend([0] * (max_len - len(pers1)))
    pers2.extend([0] * (max_len - len(pers2)))

    # Bottleneck is max difference
    return float(max(abs(a - b) / 2 for a, b in zip(pers1, pers2)))
