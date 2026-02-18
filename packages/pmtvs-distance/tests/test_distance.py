"""Tests for pmtvs-distance."""
import numpy as np
import pytest

from pmtvs_distance import (
    euclidean_distance,
    cosine_distance,
    manhattan_distance,
    dtw_distance,
)


class TestEuclideanDistance:
    """Tests for Euclidean distance."""

    def test_identical_signals(self):
        """Identical signals should have distance 0."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert euclidean_distance(x, x) == pytest.approx(0.0)

    def test_known_distance(self):
        """Test known distance."""
        x = np.array([0.0, 0.0, 0.0])
        y = np.array([1.0, 0.0, 0.0])
        assert euclidean_distance(x, y) == pytest.approx(1.0)

    def test_pythagorean(self):
        """Test 3-4-5 triangle."""
        x = np.array([0.0, 0.0])
        y = np.array([3.0, 4.0])
        assert euclidean_distance(x, y) == pytest.approx(5.0)

    def test_different_lengths(self):
        """Different length signals should return NaN."""
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([1.0, 2.0])
        assert np.isnan(euclidean_distance(x, y))


class TestCosineDistance:
    """Tests for cosine distance."""

    def test_identical_signals(self):
        """Identical signals should have distance 0."""
        x = np.array([1.0, 2.0, 3.0])
        assert cosine_distance(x, x) == pytest.approx(0.0)

    def test_opposite_signals(self):
        """Opposite signals should have distance 2."""
        x = np.array([1.0, 0.0, 0.0])
        y = np.array([-1.0, 0.0, 0.0])
        assert cosine_distance(x, y) == pytest.approx(2.0)

    def test_orthogonal_signals(self):
        """Orthogonal signals should have distance 1."""
        x = np.array([1.0, 0.0])
        y = np.array([0.0, 1.0])
        assert cosine_distance(x, y) == pytest.approx(1.0)

    def test_bounded(self):
        """Cosine distance should be in [0, 2]."""
        np.random.seed(42)
        for _ in range(10):
            x = np.random.randn(100)
            y = np.random.randn(100)
            d = cosine_distance(x, y)
            assert 0 <= d <= 2

    def test_different_lengths(self):
        """Different length signals should return NaN."""
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([1.0, 2.0])
        assert np.isnan(cosine_distance(x, y))


class TestManhattanDistance:
    """Tests for Manhattan distance."""

    def test_identical_signals(self):
        """Identical signals should have distance 0."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert manhattan_distance(x, x) == pytest.approx(0.0)

    def test_known_distance(self):
        """Test known distance."""
        x = np.array([0.0, 0.0])
        y = np.array([3.0, 4.0])
        assert manhattan_distance(x, y) == pytest.approx(7.0)

    def test_unit_differences(self):
        """Test with unit differences."""
        x = np.array([1.0, 1.0, 1.0])
        y = np.array([2.0, 2.0, 2.0])
        assert manhattan_distance(x, y) == pytest.approx(3.0)

    def test_different_lengths(self):
        """Different length signals should return NaN."""
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([1.0, 2.0])
        assert np.isnan(manhattan_distance(x, y))


class TestDTWDistance:
    """Tests for Dynamic Time Warping distance."""

    def test_identical_signals(self):
        """Identical signals should have distance 0."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert dtw_distance(x, x) == pytest.approx(0.0)

    def test_shifted_signal(self):
        """Shifted signal should have small DTW distance."""
        x = np.array([0.0, 1.0, 2.0, 1.0, 0.0])
        y = np.array([0.0, 0.0, 1.0, 2.0, 1.0, 0.0])

        # DTW should find good alignment for shifted signals
        d = dtw_distance(x, y)
        assert np.isfinite(d)
        assert d >= 0

    def test_different_lengths(self):
        """DTW should handle different length signals."""
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        d = dtw_distance(x, y)
        assert np.isfinite(d)

    def test_with_window(self):
        """Test with Sakoe-Chiba band constraint."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([1.5, 2.5, 3.5, 4.5, 5.5])

        d_no_window = dtw_distance(x, y)
        d_with_window = dtw_distance(x, y, window=2)

        assert np.isfinite(d_no_window)
        assert np.isfinite(d_with_window)
        # With window, should still find good alignment for similar signals
        assert d_with_window >= d_no_window - 0.01  # Allow small numerical diff

    def test_empty_signal(self):
        """Empty signal should return NaN."""
        x = np.array([])
        y = np.array([1.0, 2.0])
        assert np.isnan(dtw_distance(x, y))
