"""
pmtvs-matrix — Matrix analysis primitives.
"""

__version__ = "0.1.0"
BACKEND = "python"

from pmtvs_matrix.matrix import (
    covariance_matrix,
    correlation_matrix,
    eigendecomposition,
    svd_decomposition,
    matrix_rank,
    condition_number,
    effective_rank,
    graph_laplacian,
)

__all__ = [
    "__version__",
    "BACKEND",
    "covariance_matrix",
    "correlation_matrix",
    "eigendecomposition",
    "svd_decomposition",
    "matrix_rank",
    "condition_number",
    "effective_rank",
    "graph_laplacian",
]
