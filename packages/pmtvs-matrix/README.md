# pmtvs-matrix

Matrix analysis primitives.

## Installation

```bash
pip install pmtvs-matrix
```

## Functions

### Core Matrix Operations
- `covariance_matrix(data)` - Covariance matrix
- `correlation_matrix(data)` - Correlation matrix
- `eigendecomposition(matrix)` - Eigenvalues and eigenvectors
- `svd_decomposition(matrix)` - Singular value decomposition
- `matrix_rank(matrix)` - Matrix rank
- `condition_number(matrix)` - Condition number
- `effective_rank(matrix)` - Shannon entropy-based rank
- `graph_laplacian(adjacency)` - Graph Laplacian

### Eigenvalue Geometry
- `effective_dimension(eigenvalues)` - Participation ratio / entropy dimension
- `participation_ratio(eigenvalues)` - Participation ratio
- `alignment_metric(eigenvalues)` - Distribution alignment (cosine / KL)
- `eigenvalue_spread(eigenvalues)` - Coefficient of variation
- `matrix_entropy(matrix)` - Shannon entropy of eigenvalues
- `geometric_mean_eigenvalue(eigenvalues)` - Geometric mean
- `explained_variance_ratio(eigenvalues)` - Per-component variance
- `cumulative_variance_ratio(eigenvalues)` - Cumulative variance

### Dynamic Mode Decomposition
- `dynamic_mode_decomposition(signals)` - Full DMD
- `dmd_frequencies(eigenvalues, dt)` - DMD frequencies in Hz
- `dmd_growth_rates(eigenvalues, dt)` - DMD growth/decay rates

### Matrix Information Theory
- `mutual_information_matrix(signals)` - Pairwise MI matrix
- `transfer_entropy_matrix(signals)` - Directed TE matrix
- `granger_matrix(signals)` - Granger causality F-stats and p-values

## Backend

Pure Python implementation.

## License

PolyForm Strict 1.0.0 with Additional Terms.

- **Students & individual researchers:** Free. Cite us.
- **Funded research labs (grants > $100K):** Academic Research License required. [Contact us](mailto:licensing@pmtvs.dev).
- **Commercial use:** Commercial License required. [Contact us](mailto:licensing@pmtvs.dev).

See [LICENSE](LICENSE) for full terms.
