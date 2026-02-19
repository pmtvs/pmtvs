"""
pmtvs-matrix — Matrix analysis primitives.
"""

__version__ = "0.2.0"
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

from pmtvs_matrix.geometry import (
    effective_dimension,
    participation_ratio,
    alignment_metric,
    eigenvalue_spread,
    matrix_entropy,
    geometric_mean_eigenvalue,
    explained_variance_ratio,
    cumulative_variance_ratio,
)

from pmtvs_matrix.dmd import (
    dynamic_mode_decomposition,
    dmd_frequencies,
    dmd_growth_rates,
)

from pmtvs_matrix.information import (
    mutual_information_matrix,
    transfer_entropy_matrix,
    granger_matrix,
)

__all__ = [
    "__version__",
    "BACKEND",
    # matrix.py
    "covariance_matrix",
    "correlation_matrix",
    "eigendecomposition",
    "svd_decomposition",
    "matrix_rank",
    "condition_number",
    "effective_rank",
    "graph_laplacian",
    # geometry.py
    "effective_dimension",
    "participation_ratio",
    "alignment_metric",
    "eigenvalue_spread",
    "matrix_entropy",
    "geometric_mean_eigenvalue",
    "explained_variance_ratio",
    "cumulative_variance_ratio",
    # dmd.py
    "dynamic_mode_decomposition",
    "dmd_frequencies",
    "dmd_growth_rates",
    # information.py
    "mutual_information_matrix",
    "transfer_entropy_matrix",
    "granger_matrix",
]
