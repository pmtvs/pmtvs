"""
pmtvs-topology — Topological data analysis primitives.
"""

__version__ = "0.3.1"
BACKEND = "python"

from pmtvs_topology.topology import (
    distance_matrix,
    persistent_homology_0d,
    betti_numbers,
    persistence_entropy,
    persistence_landscape,
    bottleneck_distance,
    wasserstein_distance,
)

__all__ = [
    "__version__",
    "BACKEND",
    "distance_matrix",
    "persistent_homology_0d",
    "betti_numbers",
    "persistence_entropy",
    "persistence_landscape",
    "bottleneck_distance",
    "wasserstein_distance",
]
