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

### Recurrence Analysis
- `recurrence_matrix(trajectory, threshold)` - Recurrence plot
- `recurrence_rate(R)` - Fraction of recurrence points
- `determinism(R)` - Diagonal line structure
- `laminarity(R)` - Vertical line structure
- `trapping_time(R)` - Average vertical line length
- `entropy_recurrence(R)` - Line length entropy

### Attractor Analysis
- `correlation_dimension(trajectory)` - Grassberger-Procaccia
- `attractor_reconstruction(signal, dim, tau)` - Delay embedding
- `kaplan_yorke_dimension(spectrum)` - From Lyapunov exponents

### Stability Analysis
- `fixed_point_detection(trajectory)` - Find stationary regions
- `stability_index(trajectory)` - Local stability measure
- `jacobian_eigenvalues(trajectory)` - Local Jacobian eigenvalues
- `bifurcation_indicator(signal)` - Detect bifurcations
- `phase_space_contraction(trajectory)` - Flow divergence

## Backend

Pure Python implementation (no Rust acceleration for this package).
