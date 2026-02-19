# pmtvs-dynamics

Dynamical systems analysis primitives.

## Installation

```bash
pip install pmtvs-dynamics
```

## Functions

### Lyapunov Analysis
- `ftle(trajectory, dt, method)` - Finite-Time Lyapunov Exponent
- `largest_lyapunov_exponent(signal, dim, tau)` - Rosenstein method
- `lyapunov_spectrum(trajectory, dt, n_exponents)` - Full spectrum
- `lyapunov_rosenstein(signal, dimension, delay)` - Rosenstein with divergence curve
- `lyapunov_kantz(signal, dimension, delay)` - Kantz method with multi-epsilon
- `ftle_local_linearization(trajectory, time_horizon)` - FTLE via local Jacobian
- `ftle_direct_perturbation(signal, dimension)` - FTLE via perturbation

### Embedding Estimation
- `estimate_embedding_dim_cao(signal, max_dim, tau)` - Cao's method
- `estimate_tau_ami(signal, max_tau, n_bins)` - Average mutual information delay

### Recurrence Analysis
- `recurrence_matrix(trajectory, threshold)` - Recurrence plot
- `recurrence_rate(R)` - Fraction of recurrence points
- `determinism(R)` - Diagonal line structure
- `laminarity(R)` - Vertical line structure
- `trapping_time(R)` - Average vertical line length
- `entropy_recurrence(R)` - Line length entropy
- `max_diagonal_line(R)` - Longest diagonal line
- `divergence_rqa(R)` - Inverse of max diagonal line
- `determinism_from_signal(signal)` - DET directly from signal
- `rqa_metrics(signal)` - Full RQA metrics dictionary

### Attractor Analysis
- `correlation_dimension(trajectory)` - Grassberger-Procaccia
- `correlation_integral(embedded, r)` - Correlation integral at radius r
- `information_dimension(signal)` - Information dimension
- `attractor_reconstruction(signal, dim, tau)` - Delay embedding
- `kaplan_yorke_dimension(spectrum)` - From Lyapunov exponents

### Stability Analysis
- `fixed_point_detection(trajectory)` - Find stationary regions
- `stability_index(trajectory)` - Local stability measure
- `jacobian_eigenvalues(trajectory)` - Local Jacobian eigenvalues
- `bifurcation_indicator(signal)` - Detect bifurcations
- `phase_space_contraction(trajectory)` - Flow divergence
- `hilbert_stability(y)` - Instantaneous frequency stability
- `wavelet_stability(y)` - Wavelet-based stability
- `detect_collapse(effective_dim)` - Detect dimensional collapse

### Saddle Point Analysis
- `estimate_jacobian_local(trajectory, point_idx)` - Local Jacobian estimation
- `classify_jacobian_eigenvalues(jacobian)` - Eigenvalue classification
- `detect_saddle_points(trajectory)` - Find saddle points
- `compute_separatrix_distance(trajectory, saddle_indices)` - Distance to separatrix
- `compute_basin_stability(trajectory, saddle_score)` - Basin stability

### Sensitivity Analysis
- `compute_variable_sensitivity(trajectory)` - Per-variable sensitivity
- `compute_directional_sensitivity(trajectory, direction)` - Directional sensitivity
- `compute_sensitivity_evolution(sensitivity)` - Sensitivity over time
- `detect_sensitivity_transitions(sensitivity, rank)` - Regime transitions
- `compute_influence_matrix(trajectory)` - Variable influence matrix

### Domain Analysis
- `basin_stability(y)` - Basin stability measure
- `cycle_counting(y)` - Cycle statistics
- `local_outlier_factor(y)` - Local outlier factor
- `time_constant(y)` - Characteristic time constant

## Backend

Pure Python implementation (no Rust acceleration for this package).
