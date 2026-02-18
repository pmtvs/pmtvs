"""
Network Analysis Functions
"""

import numpy as np
from typing import List


def degree_centrality(adjacency: np.ndarray) -> np.ndarray:
    """
    Compute degree centrality for each node.

    Parameters
    ----------
    adjacency : np.ndarray
        Adjacency matrix (binary or weighted)

    Returns
    -------
    np.ndarray
        Degree centrality values (normalized)
    """
    adjacency = np.asarray(adjacency)
    n = len(adjacency)
    if n <= 1:
        return np.array([1.0] * n)

    degrees = np.sum(adjacency > 0, axis=1)
    return degrees / (n - 1)


def betweenness_centrality(adjacency: np.ndarray) -> np.ndarray:
    """
    Compute betweenness centrality using Brandes' algorithm (simplified).

    Parameters
    ----------
    adjacency : np.ndarray
        Adjacency matrix

    Returns
    -------
    np.ndarray
        Betweenness centrality values
    """
    adjacency = np.asarray(adjacency)
    n = len(adjacency)
    bc = np.zeros(n)

    if n <= 2:
        return bc

    for s in range(n):
        # BFS from s
        dist = np.full(n, np.inf)
        dist[s] = 0
        pred = [[] for _ in range(n)]
        sigma = np.zeros(n)
        sigma[s] = 1

        queue = [s]
        stack = []

        while queue:
            v = queue.pop(0)
            stack.append(v)

            for w in range(n):
                if adjacency[v, w] > 0 and v != w:
                    if dist[w] == np.inf:
                        dist[w] = dist[v] + 1
                        queue.append(w)
                    if dist[w] == dist[v] + 1:
                        sigma[w] += sigma[v]
                        pred[w].append(v)

        delta = np.zeros(n)
        while stack:
            w = stack.pop()
            for v in pred[w]:
                if sigma[w] > 0:
                    delta[v] += (sigma[v] / sigma[w]) * (1 + delta[w])
            if w != s:
                bc[w] += delta[w]

    # Normalize
    if n > 2:
        bc /= ((n - 1) * (n - 2))

    return bc


def closeness_centrality(adjacency: np.ndarray) -> np.ndarray:
    """
    Compute closeness centrality.

    Parameters
    ----------
    adjacency : np.ndarray
        Adjacency matrix

    Returns
    -------
    np.ndarray
        Closeness centrality values
    """
    adjacency = np.asarray(adjacency)
    n = len(adjacency)
    cc = np.zeros(n)

    if n <= 1:
        return np.ones(n)

    for i in range(n):
        # BFS for shortest paths
        dist = np.full(n, np.inf)
        dist[i] = 0
        queue = [i]

        while queue:
            v = queue.pop(0)
            for w in range(n):
                if adjacency[v, w] > 0 and dist[w] == np.inf:
                    dist[w] = dist[v] + 1
                    queue.append(w)

        reachable = dist[np.isfinite(dist)]
        if len(reachable) > 1:
            cc[i] = (len(reachable) - 1) / np.sum(dist[np.isfinite(dist)])

    return cc


def clustering_coefficient(adjacency: np.ndarray) -> np.ndarray:
    """
    Compute local clustering coefficient for each node.

    Parameters
    ----------
    adjacency : np.ndarray
        Adjacency matrix

    Returns
    -------
    np.ndarray
        Clustering coefficients
    """
    adjacency = np.asarray(adjacency)
    adjacency = (adjacency > 0).astype(float)  # Binarize
    n = len(adjacency)
    cc = np.zeros(n)

    for i in range(n):
        neighbors = np.where(adjacency[i] > 0)[0]
        k = len(neighbors)

        if k < 2:
            cc[i] = 0
            continue

        # Count edges among neighbors
        edges = 0
        for j in range(len(neighbors)):
            for l in range(j + 1, len(neighbors)):
                if adjacency[neighbors[j], neighbors[l]] > 0:
                    edges += 1

        cc[i] = 2 * edges / (k * (k - 1))

    return cc


def average_path_length(adjacency: np.ndarray) -> float:
    """
    Compute average shortest path length.

    Parameters
    ----------
    adjacency : np.ndarray
        Adjacency matrix

    Returns
    -------
    float
        Average path length (inf if disconnected)
    """
    adjacency = np.asarray(adjacency)
    n = len(adjacency)

    if n <= 1:
        return 0.0

    total_dist = 0
    count = 0

    for i in range(n):
        dist = np.full(n, np.inf)
        dist[i] = 0
        queue = [i]

        while queue:
            v = queue.pop(0)
            for w in range(n):
                if adjacency[v, w] > 0 and dist[w] == np.inf:
                    dist[w] = dist[v] + 1
                    queue.append(w)

        for j in range(n):
            if i != j and np.isfinite(dist[j]):
                total_dist += dist[j]
                count += 1

    if count == 0:
        return np.inf

    return float(total_dist / count)


def density(adjacency: np.ndarray) -> float:
    """
    Compute graph density.

    Parameters
    ----------
    adjacency : np.ndarray
        Adjacency matrix

    Returns
    -------
    float
        Density in [0, 1]
    """
    adjacency = np.asarray(adjacency)
    n = len(adjacency)

    if n <= 1:
        return 0.0

    edges = np.sum(adjacency > 0)
    # Assuming undirected, count each edge once
    edges = edges / 2  # If symmetric
    max_edges = n * (n - 1) / 2

    return float(edges / max_edges) if max_edges > 0 else 0.0


def connected_components(adjacency: np.ndarray) -> List[List[int]]:
    """
    Find connected components.

    Parameters
    ----------
    adjacency : np.ndarray
        Adjacency matrix

    Returns
    -------
    list
        List of components (each is list of node indices)
    """
    adjacency = np.asarray(adjacency)
    n = len(adjacency)
    visited = [False] * n
    components = []

    for start in range(n):
        if visited[start]:
            continue

        component = []
        queue = [start]

        while queue:
            v = queue.pop(0)
            if visited[v]:
                continue
            visited[v] = True
            component.append(v)

            for w in range(n):
                if adjacency[v, w] > 0 and not visited[w]:
                    queue.append(w)

        components.append(component)

    return components


def adjacency_from_correlation(
    correlation_matrix: np.ndarray,
    threshold: float = 0.5
) -> np.ndarray:
    """
    Create adjacency matrix from correlation matrix.

    Parameters
    ----------
    correlation_matrix : np.ndarray
        Correlation matrix
    threshold : float
        Minimum absolute correlation for an edge

    Returns
    -------
    np.ndarray
        Binary adjacency matrix
    """
    corr = np.asarray(correlation_matrix)
    adj = (np.abs(corr) >= threshold).astype(int)
    np.fill_diagonal(adj, 0)
    return adj
