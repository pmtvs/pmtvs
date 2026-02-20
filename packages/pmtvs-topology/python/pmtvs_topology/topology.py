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


def persistent_homology_1d(
    points: np.ndarray,
    max_edge_length: float = None,
    max_points: int = 200,
) -> List[Tuple[float, float]]:
    """
    Compute 1-dimensional persistent homology (loops/cycles).

    Uses Vietoris-Rips filtration with simplices up to dimension 2.

    Parameters
    ----------
    points : np.ndarray
        Point cloud (n_points, n_dims). Must be at least 2D.
    max_edge_length : float, optional
        Maximum edge length in the Rips complex. Auto-detected if None
        using the 30th percentile of pairwise distances.
    max_points : int
        Maximum points before subsampling (default 200).

    Returns
    -------
    list
        List of (birth, death) tuples for H1 features (loops).
        Birth = filtration scale where loop appears.
        Death = scale where loop is filled by a triangle (np.inf if never).
    """
    points = np.asarray(points)
    if points.ndim == 1:
        points = points.reshape(-1, 1)

    n = len(points)
    if n < 3:
        return []

    # Subsample if too many points
    if n > max_points:
        idx = np.random.choice(n, max_points, replace=False)
        points = points[idx]
        n = max_points

    dist = distance_matrix(points)

    # Auto-detect max_edge_length
    if max_edge_length is None:
        upper_tri = dist[np.triu_indices(n, k=1)]
        max_edge_length = float(np.percentile(upper_tri, 30))

    # Build sorted edges within threshold
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if dist[i, j] <= max_edge_length:
                edges.append((dist[i, j], i, j))
    edges.sort()

    if not edges:
        return []

    # Map edge pair -> index in sorted edge list
    edge_to_idx = {}
    for idx, (d, i, j) in enumerate(edges):
        edge_to_idx[(i, j)] = idx

    # Union-Find to identify cycle-creating edges (H1 births)
    parent = list(range(n))
    uf_rank = [0] * n

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        px, py = find(x), find(y)
        if px == py:
            return False
        if uf_rank[px] < uf_rank[py]:
            px, py = py, px
        parent[py] = px
        if uf_rank[px] == uf_rank[py]:
            uf_rank[px] += 1
        return True

    cycle_edge_indices = set()
    h1_births = {}

    for idx, (d, i, j) in enumerate(edges):
        if not union(i, j):
            # Both endpoints already connected -> creates a 1-cycle
            cycle_edge_indices.add(idx)
            h1_births[idx] = d

    if not cycle_edge_indices:
        return []

    # Build triangles within threshold, sorted by max edge length
    triangles = []
    # Build adjacency for fast triangle enumeration
    adj = [set() for _ in range(n)]
    for d, i, j in edges:
        adj[i].add(j)
        adj[j].add(i)

    for i in range(n):
        for j in adj[i]:
            if j <= i:
                continue
            for k in adj[i]:
                if k <= j:
                    continue
                if k in adj[j]:
                    filt_val = max(dist[i, j], dist[i, k], dist[j, k])
                    triangles.append((filt_val, i, j, k))
    triangles.sort()

    # Persistence reduction: triangles kill H1 features
    killed = set()
    reduced_columns = {}  # pivot_edge_idx -> reduced boundary chain
    persistence_pairs = []

    for filt_val, i, j, k in triangles:
        # Boundary of triangle (i,j,k) in Z/2 = {(i,j), (i,k), (j,k)}
        boundary = set()
        for a, b in [(i, j), (i, k), (j, k)]:
            key = (min(a, b), max(a, b))
            if key in edge_to_idx:
                edge_idx = edge_to_idx[key]
                boundary.symmetric_difference_update({edge_idx})

        # Column reduction against previously stored columns
        while boundary:
            pivot = max(boundary)
            if pivot in reduced_columns:
                boundary = boundary.symmetric_difference(reduced_columns[pivot])
            else:
                break

        if boundary:
            pivot = max(boundary)
            reduced_columns[pivot] = boundary
            # If pivot is a cycle-creating edge, this triangle kills that H1 feature
            if pivot in cycle_edge_indices and pivot not in killed:
                killed.add(pivot)
                persistence_pairs.append((h1_births[pivot], filt_val))

    # Features never killed persist to infinity
    for idx in cycle_edge_indices:
        if idx not in killed:
            persistence_pairs.append((h1_births[idx], np.inf))

    persistence_pairs.sort()
    return persistence_pairs


def persistent_homology(
    points: np.ndarray,
    max_edge_length: float = None,
    max_points: int = 200,
) -> dict:
    """
    Compute persistent homology in dimensions 0 and 1.

    Convenience wrapper that returns both H0 (connected components)
    and H1 (loops) persistence diagrams.

    Parameters
    ----------
    points : np.ndarray
        Point cloud (n_points, n_dims).
    max_edge_length : float, optional
        Maximum edge length for H1 Rips filtration.
    max_points : int
        Maximum points before subsampling for H1.

    Returns
    -------
    dict
        {'h0': [(birth, death), ...], 'h1': [(birth, death), ...]}
    """
    h0 = persistent_homology_0d(points)
    h1 = persistent_homology_1d(
        points, max_edge_length=max_edge_length, max_points=max_points
    )
    return {'h0': h0, 'h1': h1}


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


def wasserstein_distance(
    persistence1: List[Tuple[float, float]],
    persistence2: List[Tuple[float, float]],
    p: int = 1
) -> float:
    """
    Compute p-Wasserstein distance between persistence diagrams.

    Parameters
    ----------
    persistence1 : list
        First persistence diagram as (birth, death) pairs
    persistence2 : list
        Second persistence diagram as (birth, death) pairs
    p : int
        Order of the Wasserstein distance (default: 1)

    Returns
    -------
    float
        Wasserstein distance
    """
    # Filter finite pairs
    p1 = [(b, d) for b, d in persistence1 if np.isfinite(d)]
    p2 = [(b, d) for b, d in persistence2 if np.isfinite(d)]

    if len(p1) == 0 and len(p2) == 0:
        return 0.0

    # Persistence values sorted descending
    pers1 = sorted([d - b for b, d in p1], reverse=True)
    pers2 = sorted([d - b for b, d in p2], reverse=True)

    # Pad shorter list with zeros (diagonal matches)
    max_len = max(len(pers1), len(pers2))
    pers1.extend([0] * (max_len - len(pers1)))
    pers2.extend([0] * (max_len - len(pers2)))

    # p-Wasserstein: (sum |a_i - b_i|^p)^(1/p)
    cost = sum(abs(a - b) ** p for a, b in zip(pers1, pers2))
    return float(cost ** (1.0 / p))
