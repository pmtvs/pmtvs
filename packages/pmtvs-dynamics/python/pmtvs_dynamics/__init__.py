"""
pmtvs-dynamics — Dynamical systems analysis primitives.

numpy in, number/array out.
"""

__version__ = "0.3.3"
BACKEND = "python"

# --- Public API ---
from pmtvs_dynamics.lyapunov import (
    ftle,
    largest_lyapunov_exponent,
    lyapunov_spectrum,
    lyapunov_rosenstein,
    lyapunov_kantz,
    estimate_embedding_dim_cao,
    estimate_tau_ami,
    ftle_local_linearization,
    ftle_direct_perturbation,
)

from pmtvs_dynamics.recurrence import (
    recurrence_matrix,
    recurrence_rate,
    determinism,
    laminarity,
    trapping_time,
    entropy_recurrence,
    max_diagonal_line,
    divergence_rqa,
    determinism_from_signal,
    rqa_metrics,
)

from pmtvs_dynamics.attractor import (
    correlation_dimension,
    attractor_reconstruction,
    kaplan_yorke_dimension,
)

from pmtvs_dynamics.stability import (
    fixed_point_detection,
    stability_index,
    jacobian_eigenvalues,
    bifurcation_indicator,
    phase_space_contraction,
    hilbert_stability,
    wavelet_stability,
    detect_collapse,
)

from pmtvs_dynamics.saddle import (
    estimate_jacobian_local,
    classify_jacobian_eigenvalues,
    detect_saddle_points,
    compute_separatrix_distance,
    compute_basin_stability,
)

from pmtvs_dynamics.sensitivity import (
    compute_variable_sensitivity,
    compute_directional_sensitivity,
    compute_sensitivity_evolution,
    detect_sensitivity_transitions,
    compute_influence_matrix,
)

from pmtvs_dynamics.dimension import (
    correlation_integral,
    information_dimension,
)

from pmtvs_dynamics.domain import (
    basin_stability,
    cycle_counting,
    local_outlier_factor,
    time_constant,
)

__all__ = [
    "__version__",
    "BACKEND",
    # Lyapunov
    "ftle",
    "largest_lyapunov_exponent",
    "lyapunov_spectrum",
    "lyapunov_rosenstein",
    "lyapunov_kantz",
    "estimate_embedding_dim_cao",
    "estimate_tau_ami",
    "ftle_local_linearization",
    "ftle_direct_perturbation",
    # Recurrence
    "recurrence_matrix",
    "recurrence_rate",
    "determinism",
    "laminarity",
    "trapping_time",
    "entropy_recurrence",
    "max_diagonal_line",
    "divergence_rqa",
    "determinism_from_signal",
    "rqa_metrics",
    # Attractor
    "correlation_dimension",
    "attractor_reconstruction",
    "kaplan_yorke_dimension",
    # Stability
    "fixed_point_detection",
    "stability_index",
    "jacobian_eigenvalues",
    "bifurcation_indicator",
    "phase_space_contraction",
    "hilbert_stability",
    "wavelet_stability",
    "detect_collapse",
    # Saddle
    "estimate_jacobian_local",
    "classify_jacobian_eigenvalues",
    "detect_saddle_points",
    "compute_separatrix_distance",
    "compute_basin_stability",
    # Sensitivity
    "compute_variable_sensitivity",
    "compute_directional_sensitivity",
    "compute_sensitivity_evolution",
    "detect_sensitivity_transitions",
    "compute_influence_matrix",
    # Dimension
    "correlation_integral",
    "information_dimension",
    # Domain
    "basin_stability",
    "cycle_counting",
    "local_outlier_factor",
    "time_constant",
]
