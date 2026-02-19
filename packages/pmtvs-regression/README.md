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
