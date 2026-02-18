# pmtvs-spectral

Spectral analysis primitives.

## Installation

```bash
pip install pmtvs-spectral
```

## Functions

- `power_spectral_density(signal, fs)` - Welch's method PSD
- `dominant_frequency(signal, fs)` - Peak frequency
- `spectral_entropy(signal, fs)` - Flatness measure
- `spectral_centroid(signal, fs)` - Center of mass
- `spectral_bandwidth(signal, fs)` - Spread around centroid
- `spectral_rolloff(signal, fs)` - Energy threshold frequency
- `spectral_flatness(signal, fs)` - Wiener entropy
- `harmonic_ratio(signal, fs)` - Harmonic-to-noise ratio
- `total_harmonic_distortion(signal, fs)` - THD

## Backend

Pure Python implementation.
