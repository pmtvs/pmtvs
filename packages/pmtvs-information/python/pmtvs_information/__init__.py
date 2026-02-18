"""
pmtvs-information — Information theory primitives.
"""

__version__ = "0.1.0"
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

__all__ = [
    "__version__",
    "BACKEND",
    "mutual_information",
    "transfer_entropy",
    "conditional_entropy",
    "joint_entropy",
    "kl_divergence",
    "js_divergence",
    "information_gain",
]
