"""
pmtvs-network — Network analysis primitives.
"""

__version__ = "0.3.3"
BACKEND = "python"

from pmtvs_network.network import (
    degree_centrality,
    betweenness_centrality,
    closeness_centrality,
    eigenvector_centrality,
    clustering_coefficient,
    average_path_length,
    density,
    connected_components,
    adjacency_from_correlation,
)

from pmtvs_network.community import (
    modularity,
    community_detection,
)

__all__ = [
    "__version__",
    "BACKEND",
    "degree_centrality",
    "betweenness_centrality",
    "closeness_centrality",
    "eigenvector_centrality",
    "clustering_coefficient",
    "average_path_length",
    "density",
    "connected_components",
    "adjacency_from_correlation",
    "modularity",
    "community_detection",
]
