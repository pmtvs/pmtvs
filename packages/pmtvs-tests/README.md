# pmtvs-tests

Statistical hypothesis testing primitives.

## Functions

- `bootstrap_mean(data)` - Bootstrap distribution of mean
- `bootstrap_confidence_interval(data, statistic)` - Bootstrap CI
- `permutation_test(x, y)` - Two-sample permutation test
- `surrogate_test(signal, statistic)` - Surrogate data test
- `adf_test(signal)` - Augmented Dickey-Fuller (stationarity)
- `runs_test(signal)` - Runs test (randomness)
- `mann_kendall_test(signal)` - Trend test
