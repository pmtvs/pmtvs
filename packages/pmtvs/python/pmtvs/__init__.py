"""pmtvs — all signal analysis primitives in one import."""

__version__ = "0.3.1"

# Re-export everything from sub-packages
from pmtvs_entropy import *
from pmtvs_fractal import *
from pmtvs_statistics import *
from pmtvs_correlation import *
from pmtvs_distance import *
from pmtvs_embedding import *
from pmtvs_dynamics import *
from pmtvs_spectral import *
from pmtvs_matrix import *
from pmtvs_topology import *
from pmtvs_network import *
from pmtvs_information import *
from pmtvs_tests import *
from pmtvs_regression import *

# --- Aliases for backward compatibility ---
# Distance
from pmtvs_distance import dtw_distance as dynamic_time_warping
# Embedding
from pmtvs_embedding import delay_embedding as time_delay_embedding
from pmtvs_embedding import mutual_information_delay as optimal_delay
from pmtvs_embedding import optimal_embedding_dimension as optimal_dimension
# Dynamics
from pmtvs_dynamics import estimate_embedding_dim_cao as cao_embedding_analysis
from pmtvs_dynamics import entropy_recurrence as entropy_rqa
# Spectral
from pmtvs_spectral import power_spectral_density as psd
# Network
from pmtvs_network import density as network_density
from pmtvs_network import betweenness_centrality as centrality_betweenness
from pmtvs_network import eigenvector_centrality as centrality_eigenvector
# Topology
from pmtvs_topology import persistent_homology_0d as persistence_diagram
