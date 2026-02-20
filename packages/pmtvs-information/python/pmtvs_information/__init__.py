"""
pmtvs-information — Information theory primitives.
"""

__version__ = "0.3.1"
BACKEND = "python"

from pmtvs_information.information import (
    mutual_information,
    transfer_entropy,
    conditional_entropy,
    joint_entropy,
    kl_divergence,
    js_divergence,
    information_gain,
)

from pmtvs_information.entropy import (
    shannon_entropy,
    renyi_entropy,
    tsallis_entropy,
)

from pmtvs_information.divergence import (
    cross_entropy,
    hellinger_distance,
    total_variation_distance,
)

from pmtvs_information.mutual import (
    conditional_mutual_information,
    multivariate_mutual_information,
    total_correlation,
    interaction_information,
    dual_total_correlation,
)

from pmtvs_information.decomposition import (
    partial_information_decomposition,
    redundancy,
    synergy,
    information_atoms,
)

from pmtvs_information.causality import (
    granger_causality,
    convergent_cross_mapping,
    phase_coupling,
    information_flow,
)

__all__ = [
    "__version__",
    "BACKEND",
    # information.py
    "mutual_information",
    "transfer_entropy",
    "conditional_entropy",
    "joint_entropy",
    "kl_divergence",
    "js_divergence",
    "information_gain",
    # entropy.py
    "shannon_entropy",
    "renyi_entropy",
    "tsallis_entropy",
    # divergence.py
    "cross_entropy",
    "hellinger_distance",
    "total_variation_distance",
    # mutual.py
    "conditional_mutual_information",
    "multivariate_mutual_information",
    "total_correlation",
    "interaction_information",
    "dual_total_correlation",
    # decomposition.py
    "partial_information_decomposition",
    "redundancy",
    "synergy",
    "information_atoms",
    # causality.py
    "granger_causality",
    "convergent_cross_mapping",
    "phase_coupling",
    "information_flow",
]
