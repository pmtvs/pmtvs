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

## License

PolyForm Strict 1.0.0 with Additional Terms.

- **Students & individual researchers:** Free. Cite us.
- **Funded research labs (grants > $100K):** Academic Research License required. [Contact us](mailto:licensing@pmtvs.dev).
- **Commercial use:** Commercial License required. [Contact us](mailto:licensing@pmtvs.dev).

See [LICENSE](LICENSE) for full terms.
