# pmtvs-information

Information theory primitives.

## Installation

```bash
pip install pmtvs-information
```

## Functions

### Core Information Measures
- `mutual_information(x, y)` - I(X;Y)
- `transfer_entropy(source, target)` - Information flow
- `conditional_entropy(x, y)` - H(X|Y)
- `joint_entropy(x, y)` - H(X,Y)
- `kl_divergence(p, q)` - Kullback-Leibler divergence
- `js_divergence(p, q)` - Jensen-Shannon divergence
- `information_gain(x, y)` - IG(X;Y)

### Entropy Variants
- `shannon_entropy(data)` - Shannon entropy H(X)
- `renyi_entropy(data, alpha)` - Renyi entropy of order alpha
- `tsallis_entropy(data, q)` - Tsallis non-extensive entropy

### Divergence Measures
- `cross_entropy(p, q)` - Cross entropy H(P, Q)
- `hellinger_distance(p, q)` - Hellinger distance
- `total_variation_distance(p, q)` - Total variation distance

### Multivariate Information
- `conditional_mutual_information(x, y, z)` - I(X;Y|Z)
- `multivariate_mutual_information(variables)` - Co-information
- `total_correlation(variables)` - Total correlation
- `interaction_information(variables)` - Interaction information
- `dual_total_correlation(variables)` - Dual total correlation

### Information Decomposition
- `partial_information_decomposition(s1, s2, target)` - PID
- `redundancy(sources, target)` - Redundant information
- `synergy(sources, target)` - Synergistic information
- `information_atoms(sources, target)` - All information atoms

### Causality
- `granger_causality(source, target)` - Granger causality test
- `convergent_cross_mapping(sig_a, sig_b)` - CCM for nonlinear causality
- `phase_coupling(signal1, signal2)` - Phase-locking value

## Backend

Pure Python implementation. Requires scipy >= 1.7.

## License

PolyForm Noncommercial 1.0.0 with Additional Terms.

- **Students & individual researchers:** Free. Cite us.
- **Funded research labs (grants > $100K):** Academic Research License required. [Contact us](mailto:licensing@pmtvs.dev).
- **Commercial use:** Commercial License required. [Contact us](mailto:licensing@pmtvs.dev).

See [LICENSE](LICENSE) for full terms.
