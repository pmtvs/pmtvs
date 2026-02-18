# pmtvs-embedding

Time-delay embedding for dynamical systems analysis.

## Installation

```bash
pip install pmtvs-embedding
```

## Functions

- `delay_embedding(signal, dim, tau)` - Construct time-delay embedding matrix
- `optimal_embedding_dimension(signal, tau, max_dim, threshold)` - Cao's method
- `mutual_information_delay(signal, max_lag, n_bins)` - Find optimal delay
- `false_nearest_neighbors(signal, tau, max_dim)` - FNN method

## Rust Acceleration

1 of 4 functions has Rust implementation (~12x speedup for delay_embedding).
Disable with `PMTVS_USE_RUST=0`.
