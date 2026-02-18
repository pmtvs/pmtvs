"""
pmtvs-dynamics — Dynamical systems analysis primitives.

numpy in, number/array out.
"""

__version__ = "0.1.0"
BACKEND = "python"

# --- Public API ---
from pmtvs_dynamics.lyapunov import (
    ftle,
    largest_lyapunov_exponent,
    lyapunov_spectrum,
)

from pmtvs_dynamics.recurrence import (
    recurrence_matrix,
    recurrence_rate,
    determinism,
    laminarity,
    trapping_time,
    entropy_recurrence,
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
)

__all__ = [
    "__version__",
    "BACKEND",
    # Lyapunov
    "ftle",
    "largest_lyapunov_exponent",
    "lyapunov_spectrum",
    # Recurrence
    "recurrence_matrix",
    "recurrence_rate",
    "determinism",
    "laminarity",
    "trapping_time",
    "entropy_recurrence",
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
]
