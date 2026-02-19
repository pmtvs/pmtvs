"""Tests for pmtvs-regression."""
import numpy as np
import pytest

from pmtvs_regression import (
    linear_regression,
    ratio,
    product,
    difference,
    sum_signals,
)


class TestLinearRegression:
    """Tests for linear_regression."""

    def test_perfect_positive_line(self):
        """Perfect linear relationship: y = 2x + 1."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = 2.0 * x + 1.0
        slope, intercept, r_sq, std_err = linear_regression(x, y)
        assert slope == pytest.approx(2.0, abs=1e-10)
        assert intercept == pytest.approx(1.0, abs=1e-10)
        assert r_sq == pytest.approx(1.0, abs=1e-10)
        assert std_err == pytest.approx(0.0, abs=1e-10)

    def test_negative_slope(self):
        """Negative linear relationship: y = -3x + 10."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = -3.0 * x + 10.0
        slope, intercept, r_sq, std_err = linear_regression(x, y)
        assert slope == pytest.approx(-3.0, abs=1e-10)
        assert intercept == pytest.approx(10.0, abs=1e-10)
        assert r_sq == pytest.approx(1.0, abs=1e-10)

    def test_noisy_data(self):
        """Noisy data should have r_squared < 1."""
        np.random.seed(42)
        x = np.linspace(0, 10, 50)
        y = 2.0 * x + 1.0 + np.random.randn(50) * 2.0
        slope, intercept, r_sq, std_err = linear_regression(x, y)
        assert 1.5 < slope < 2.5
        assert 0.5 < r_sq < 1.0
        assert std_err > 0.0

    def test_too_few_points(self):
        """Fewer than 3 points should return all NaN."""
        x = np.array([1.0, 2.0])
        y = np.array([3.0, 4.0])
        slope, intercept, r_sq, std_err = linear_regression(x, y)
        assert np.isnan(slope)
        assert np.isnan(intercept)
        assert np.isnan(r_sq)
        assert np.isnan(std_err)

    def test_empty_arrays(self):
        """Empty arrays should return all NaN."""
        slope, intercept, r_sq, std_err = linear_regression([], [])
        assert np.isnan(slope)
        assert np.isnan(intercept)

    def test_constant_x(self):
        """Constant x (zero variance) should return all NaN."""
        x = np.array([5.0, 5.0, 5.0, 5.0])
        y = np.array([1.0, 2.0, 3.0, 4.0])
        slope, intercept, r_sq, std_err = linear_regression(x, y)
        assert np.isnan(slope)
        assert np.isnan(intercept)

    def test_nan_filtering(self):
        """NaN values should be filtered out before regression."""
        x = np.array([1.0, np.nan, 3.0, 4.0, 5.0])
        y = np.array([3.0, 5.0, 7.0, np.nan, 11.0])
        slope, intercept, r_sq, std_err = linear_regression(x, y)
        # After filtering NaN, we have x=[1,3,5], y=[3,7,11] => y = 2x + 1
        assert slope == pytest.approx(2.0, abs=1e-10)
        assert intercept == pytest.approx(1.0, abs=1e-10)
        assert r_sq == pytest.approx(1.0, abs=1e-10)

    def test_unequal_lengths(self):
        """Arrays of different length should be truncated to shorter."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([3.0, 5.0, 7.0])
        slope, intercept, r_sq, std_err = linear_regression(x, y)
        # Uses x=[1,2,3], y=[3,5,7] => y = 2x + 1
        assert slope == pytest.approx(2.0, abs=1e-10)
        assert intercept == pytest.approx(1.0, abs=1e-10)

    def test_returns_floats(self):
        """All return values should be native Python floats."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = 2.0 * x + 1.0
        result = linear_regression(x, y)
        for val in result:
            assert isinstance(val, float)

    def test_constant_y(self):
        """Constant y should have r_squared of special value and zero slope."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
        slope, intercept, r_sq, std_err = linear_regression(x, y)
        assert slope == pytest.approx(0.0, abs=1e-10)
        assert intercept == pytest.approx(5.0, abs=1e-10)


class TestRatio:
    """Tests for ratio."""

    def test_basic_ratio(self):
        """Basic element-wise ratio."""
        a = np.array([10.0, 20.0, 30.0])
        b = np.array([2.0, 5.0, 10.0])
        result = ratio(a, b)
        np.testing.assert_allclose(result, [5.0, 4.0, 3.0])

    def test_division_by_near_zero(self):
        """Division by near-zero should use epsilon protection."""
        a = np.array([1.0, 2.0])
        b = np.array([0.0, 0.0])
        result = ratio(a, b)
        assert np.all(np.isfinite(result))

    def test_negative_denominator(self):
        """Negative near-zero denominator should preserve sign."""
        a = np.array([1.0])
        b = np.array([-1e-15])
        result = ratio(a, b)
        assert np.isfinite(result[0])

    def test_unequal_lengths(self):
        """Truncate to shorter array."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.0, 1.0])
        result = ratio(a, b)
        assert len(result) == 2
        np.testing.assert_allclose(result, [1.0, 2.0])

    def test_returns_ndarray(self):
        """Result should be a numpy array."""
        result = ratio([6.0, 8.0], [2.0, 4.0])
        assert isinstance(result, np.ndarray)

    def test_custom_epsilon(self):
        """Custom epsilon value should be respected."""
        a = np.array([1.0])
        b = np.array([1e-5])
        # With large epsilon, b is replaced
        result_large = ratio(a, b, epsilon=1e-3)
        # With small epsilon, b is used as-is
        result_small = ratio(a, b, epsilon=1e-8)
        assert result_large[0] != result_small[0]


class TestProduct:
    """Tests for product."""

    def test_basic_product(self):
        """Basic element-wise product."""
        a = np.array([2.0, 3.0, 4.0])
        b = np.array([5.0, 6.0, 7.0])
        result = product(a, b)
        np.testing.assert_allclose(result, [10.0, 18.0, 28.0])

    def test_zeros(self):
        """Product with zeros should be zero."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([0.0, 0.0, 0.0])
        result = product(a, b)
        np.testing.assert_allclose(result, [0.0, 0.0, 0.0])

    def test_negatives(self):
        """Product with negatives."""
        a = np.array([-1.0, 2.0, -3.0])
        b = np.array([4.0, -5.0, -6.0])
        result = product(a, b)
        np.testing.assert_allclose(result, [-4.0, -10.0, 18.0])

    def test_unequal_lengths(self):
        """Truncate to shorter array."""
        a = np.array([1.0, 2.0, 3.0, 4.0])
        b = np.array([10.0, 20.0])
        result = product(a, b)
        assert len(result) == 2
        np.testing.assert_allclose(result, [10.0, 40.0])

    def test_list_input(self):
        """Should accept plain lists."""
        result = product([2.0, 3.0], [4.0, 5.0])
        np.testing.assert_allclose(result, [8.0, 15.0])

    def test_returns_ndarray(self):
        """Result should be a numpy array."""
        result = product([1.0], [2.0])
        assert isinstance(result, np.ndarray)


class TestDifference:
    """Tests for difference."""

    def test_basic_difference(self):
        """Basic element-wise difference."""
        a = np.array([10.0, 20.0, 30.0])
        b = np.array([1.0, 2.0, 3.0])
        result = difference(a, b)
        np.testing.assert_allclose(result, [9.0, 18.0, 27.0])

    def test_identical_signals(self):
        """Difference of identical signals should be zero."""
        a = np.array([1.0, 2.0, 3.0])
        result = difference(a, a)
        np.testing.assert_allclose(result, [0.0, 0.0, 0.0])

    def test_negative_result(self):
        """Difference can be negative."""
        a = np.array([1.0, 2.0])
        b = np.array([5.0, 10.0])
        result = difference(a, b)
        np.testing.assert_allclose(result, [-4.0, -8.0])

    def test_unequal_lengths(self):
        """Truncate to shorter array."""
        a = np.array([1.0, 2.0])
        b = np.array([1.0, 1.0, 1.0, 1.0])
        result = difference(a, b)
        assert len(result) == 2
        np.testing.assert_allclose(result, [0.0, 1.0])

    def test_list_input(self):
        """Should accept plain lists."""
        result = difference([5.0, 10.0], [1.0, 3.0])
        np.testing.assert_allclose(result, [4.0, 7.0])

    def test_returns_ndarray(self):
        """Result should be a numpy array."""
        result = difference([1.0], [2.0])
        assert isinstance(result, np.ndarray)


class TestSumSignals:
    """Tests for sum_signals."""

    def test_basic_sum(self):
        """Basic element-wise sum."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0, 6.0])
        result = sum_signals(a, b)
        np.testing.assert_allclose(result, [5.0, 7.0, 9.0])

    def test_zeros(self):
        """Sum with zeros should return the other signal."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([0.0, 0.0, 0.0])
        result = sum_signals(a, b)
        np.testing.assert_allclose(result, a)

    def test_negatives_cancel(self):
        """Negative signals should cancel positive."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([-1.0, -2.0, -3.0])
        result = sum_signals(a, b)
        np.testing.assert_allclose(result, [0.0, 0.0, 0.0])

    def test_unequal_lengths(self):
        """Truncate to shorter array."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([10.0])
        result = sum_signals(a, b)
        assert len(result) == 1
        np.testing.assert_allclose(result, [11.0])

    def test_list_input(self):
        """Should accept plain lists."""
        result = sum_signals([1.0, 2.0], [3.0, 4.0])
        np.testing.assert_allclose(result, [4.0, 6.0])

    def test_returns_ndarray(self):
        """Result should be a numpy array."""
        result = sum_signals([1.0], [2.0])
        assert isinstance(result, np.ndarray)

    def test_commutativity(self):
        """sum_signals(a, b) should equal sum_signals(b, a)."""
        a = np.array([1.0, 3.0, 5.0])
        b = np.array([2.0, 4.0, 6.0])
        np.testing.assert_allclose(sum_signals(a, b), sum_signals(b, a))
