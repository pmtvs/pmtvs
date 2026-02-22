"""
pmtvs-topology — Topological data analysis primitives.
"""

__version__ = "0.3.3"
BACKEND = "python"

from pmtvs_topology.topology import (
    distance_matrix,
    persistent_homology_0d,
    persistent_homology_1d,
    persistent_homology,
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
    "persistent_homology_1d",
    "persistent_homology",
    "betti_numbers",
    "persistence_entropy",
    "persistence_landscape",
    "bottleneck_distance",
    "wasserstein_distance",
]
