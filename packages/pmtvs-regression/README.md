# pmtvs-regression

Pairwise regression and signal arithmetic primitives.

## Installation

```bash
pip install pmtvs-regression
```

## Functions

- `linear_regression(sig_a, sig_b)` - OLS linear regression (slope, intercept, R², std error)
- `ratio(sig_a, sig_b)` - Element-wise ratio with epsilon protection
- `product(sig_a, sig_b)` - Element-wise product
- `difference(sig_a, sig_b)` - Element-wise difference
- `sum_signals(sig_a, sig_b)` - Element-wise sum

## Backend

Pure Python implementation.

## License

PolyForm Noncommercial 1.0.0 with Additional Terms.

- **Students & individual researchers:** Free. Cite us.
- **Funded research labs (grants > $100K):** Academic Research License required. [Contact us](mailto:licensing@pmtvs.dev).
- **Commercial use:** Commercial License required. [Contact us](mailto:licensing@pmtvs.dev).

See [LICENSE](LICENSE) for full terms.
