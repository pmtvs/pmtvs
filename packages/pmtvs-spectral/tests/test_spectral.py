"""Tests for pmtvs-spectral."""
import numpy as np
import pytest

from pmtvs_spectral import (
    power_spectral_density,
    dominant_frequency,
    spectral_entropy,
    spectral_centroid,
    spectral_bandwidth,
    spectral_rolloff,
    spectral_flatness,
    harmonic_ratio,
    total_harmonic_distortion,
)


class TestPowerSpectralDensity:
    """Tests for PSD."""

    def test_returns_tuple(self):
        """Should return (freqs, psd) tuple."""
        signal = np.random.randn(512)
        freqs, psd = power_spectral_density(signal)
        assert len(freqs) == len(psd)

    def test_psd_non_negative(self):
        """PSD should be non-negative."""
        signal = np.random.randn(512)
        freqs, psd = power_spectral_density(signal)
        assert np.all(psd >= 0)


class TestDominantFrequency:
    """Tests for dominant frequency."""

    def test_sinusoid(self):
        """Sinusoid should have dominant frequency at signal frequency."""
        fs = 100
        f0 = 10
        t = np.arange(1000) / fs
        signal = np.sin(2 * np.pi * f0 * t)

        dom_freq = dominant_frequency(signal, fs=fs)
        assert abs(dom_freq - f0) < 1  # Within 1 Hz


class TestSpectralEntropy:
    """Tests for spectral entropy."""

    def test_bounded(self):
        """Spectral entropy should be in [0, 1] when normalized."""
        signal = np.random.randn(512)
        se = spectral_entropy(signal, normalize=True)
        assert 0 <= se <= 1

    def test_noise_high_entropy(self):
        """White noise should have high spectral entropy."""
        np.random.seed(42)
        signal = np.random.randn(1024)
        se = spectral_entropy(signal, normalize=True)
        assert se > 0.8

    def test_sinusoid_low_entropy(self):
        """Pure sinusoid should have low spectral entropy."""
        t = np.linspace(0, 1, 1024)
        signal = np.sin(2 * np.pi * 10 * t)
        se = spectral_entropy(signal, normalize=True)
        assert se < 0.5


class TestSpectralCentroid:
    """Tests for spectral centroid."""

    def test_returns_float(self):
        """Should return float."""
        signal = np.random.randn(512)
        sc = spectral_centroid(signal, fs=100)
        assert isinstance(sc, float)

    def test_positive(self):
        """Centroid should be positive."""
        signal = np.random.randn(512)
        sc = spectral_centroid(signal, fs=100)
        assert sc > 0


class TestSpectralBandwidth:
    """Tests for spectral bandwidth."""

    def test_returns_float(self):
        """Should return float."""
        signal = np.random.randn(512)
        sb = spectral_bandwidth(signal, fs=100)
        assert isinstance(sb, float)

    def test_non_negative(self):
        """Bandwidth should be non-negative."""
        signal = np.random.randn(512)
        sb = spectral_bandwidth(signal, fs=100)
        assert sb >= 0


class TestSpectralRolloff:
    """Tests for spectral rolloff."""

    def test_bounded(self):
        """Rolloff should be within Nyquist."""
        fs = 100
        signal = np.random.randn(512)
        sr = spectral_rolloff(signal, fs=fs)
        assert 0 <= sr <= fs / 2


class TestSpectralFlatness:
    """Tests for spectral flatness."""

    def test_bounded(self):
        """Flatness should be in [0, 1]."""
        signal = np.random.randn(512)
        sf = spectral_flatness(signal)
        assert 0 <= sf <= 1

    def test_noise_high_flatness(self):
        """White noise should have high flatness."""
        np.random.seed(42)
        signal = np.random.randn(1024)
        sf = spectral_flatness(signal)
        assert sf > 0.5


class TestHarmonicRatio:
    """Tests for harmonic ratio."""

    def test_returns_float(self):
        """Should return float."""
        t = np.linspace(0, 1, 1024)
        signal = np.sin(2 * np.pi * 10 * t)
        hr = harmonic_ratio(signal, fs=1024)
        assert isinstance(hr, float)


class TestTotalHarmonicDistortion:
    """Tests for THD."""

    def test_pure_sinusoid_low_thd(self):
        """Pure sinusoid should have low THD."""
        t = np.linspace(0, 1, 4096)
        signal = np.sin(2 * np.pi * 100 * t)
        thd = total_harmonic_distortion(signal, fs=4096)
        # Pure sinusoid should have very low THD
        assert np.isfinite(thd) and thd < 0.5

    def test_square_wave_high_thd(self):
        """Square wave should have high THD."""
        t = np.linspace(0, 1, 4096)
        signal = np.sign(np.sin(2 * np.pi * 100 * t))
        thd = total_harmonic_distortion(signal, fs=4096)
        assert np.isfinite(thd) and thd > 0.1
