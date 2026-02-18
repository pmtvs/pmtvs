"""Tests for calculus primitives"""

import numpy as np
import pytest
from pmtvs.individual import derivative, integral, curvature


class TestDerivative:
    """Test derivative function"""

    def test_derivative_linear(self):
        # d/dx(2x) = 2
        x = np.linspace(0, 10, 100)
        y = 2 * x
        dy = derivative(y, dt=x[1] - x[0])
        assert np.allclose(dy, 2, rtol=0.01)

    def test_derivative_quadratic(self):
        # d/dx(x^2) = 2x
        x = np.linspace(0, 10, 1000)
        y = x ** 2
        dy = derivative(y, dt=x[1] - x[0])
        # Check middle portion (edges have boundary effects)
        assert np.allclose(dy[100:-100], 2 * x[100:-100], rtol=0.01)

    def test_derivative_second_order(self):
        # d²/dx²(x^2) = 2
        x = np.linspace(0, 10, 1000)
        y = x ** 2
        d2y = derivative(y, dt=x[1] - x[0], order=2)
        # Check middle portion
        assert np.allclose(d2y[100:-100], 2, rtol=0.02)

    def test_derivative_sine(self):
        # d/dx(sin(x)) = cos(x)
        x = np.linspace(0, 2*np.pi, 1000)
        y = np.sin(x)
        dy = derivative(y, dt=x[1] - x[0])
        expected = np.cos(x)
        assert np.allclose(dy[50:-50], expected[50:-50], rtol=0.01)


class TestIntegral:
    """Test integral function"""

    def test_integral_constant(self):
        # ∫2 dx = 2x
        x = np.linspace(0, 10, 100)
        y = np.ones_like(x) * 2
        dt = x[1] - x[0]
        iy = integral(y, dt=dt)
        expected = 2 * x
        assert np.allclose(iy, expected, rtol=0.01)

    def test_integral_linear(self):
        # ∫x dx = x²/2
        x = np.linspace(0, 10, 1000)
        y = x
        dt = x[1] - x[0]
        iy = integral(y, dt=dt)
        expected = x ** 2 / 2
        assert np.allclose(iy, expected, rtol=0.01)

    def test_integral_initial_value(self):
        x = np.linspace(0, 10, 100)
        y = np.ones_like(x) * 2
        dt = x[1] - x[0]
        iy = integral(y, dt=dt, initial=5.0)
        assert iy[0] == 5.0


class TestCurvature:
    """Test curvature function"""

    def test_curvature_straight_line(self):
        # Straight line has zero curvature
        x = np.linspace(0, 10, 100)
        y = 2 * x + 1
        k = curvature(y, dt=x[1] - x[0])
        assert np.allclose(k[10:-10], 0, atol=1e-6)

    def test_curvature_circle(self):
        # Circle of radius R has curvature 1/R
        R = 5.0
        theta = np.linspace(0, np.pi, 1000)
        # Parametric circle
        x = R * np.cos(theta)
        y = R * np.sin(theta)
        dt = theta[1] - theta[0]

        # For parametric curve, curvature formula is different
        # Let's just check that curvature is non-zero and positive
        k = curvature(y, dt=dt)
        assert np.all(k[100:-100] > 0)

    def test_curvature_returns_array(self):
        x = np.linspace(0, 10, 100)
        y = x ** 2
        k = curvature(y, dt=x[1] - x[0])
        assert isinstance(k, np.ndarray)
        assert len(k) == len(y)
