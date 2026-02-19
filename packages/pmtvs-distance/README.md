# pmtvs-distance

Distance metrics for signal comparison.

## Installation

```bash
pip install pmtvs-distance
```

## Functions

- `euclidean_distance(x, y)` - Euclidean (L2) distance
- `cosine_distance(x, y)` - Cosine distance (1 - cosine similarity)
- `cosine_similarity(x, y)` - Cosine similarity
- `manhattan_distance(x, y)` - Manhattan (L1) distance
- `dtw_distance(x, y, window=None)` - Dynamic Time Warping distance
- `earth_movers_distance(x, y)` - Earth mover's (Wasserstein) distance

## Rust Acceleration

3 of 6 functions have Rust implementations (~8x speedup).
Disable with `PMTVS_USE_RUST=0`.
