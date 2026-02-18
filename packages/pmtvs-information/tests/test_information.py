"""Tests for pmtvs-information."""
import numpy as np
import pytest

from pmtvs_information import (
    mutual_information,
    transfer_entropy,
    conditional_entropy,
    joint_entropy,
    kl_divergence,
    js_divergence,
    information_gain,
)


class TestMutualInformation:
    def test_non_negative(self):
        np.random.seed(42)
        x = np.random.randn(500)
        y = np.random.randn(500)
        mi = mutual_information(x, y)
        assert mi >= 0

    def test_identical_signals(self):
        np.random.seed(42)
        x = np.random.randn(500)
        mi = mutual_information(x, x)
        assert mi > 0  # Should be positive for identical signals

    def test_correlated_higher(self):
        np.random.seed(42)
        x = np.random.randn(500)
        y = x + 0.1 * np.random.randn(500)  # Correlated
        z = np.random.randn(500)  # Uncorrelated

        mi_corr = mutual_information(x, y)
        mi_uncorr = mutual_information(x, z)

        assert mi_corr > mi_uncorr


class TestTransferEntropy:
    def test_non_negative(self):
        np.random.seed(42)
        x = np.random.randn(500)
        y = np.random.randn(500)
        te = transfer_entropy(x, y)
        assert te >= 0

    def test_causal_signal(self):
        np.random.seed(42)
        x = np.random.randn(500)
        # y depends on past x
        y = np.zeros(500)
        y[1:] = 0.8 * x[:-1] + 0.2 * np.random.randn(499)

        te_xy = transfer_entropy(x, y, lag=1)
        te_yx = transfer_entropy(y, x, lag=1)

        # Transfer entropy should be higher x->y
        assert te_xy > te_yx - 0.1  # Allow some tolerance


class TestConditionalEntropy:
    def test_less_than_joint(self):
        np.random.seed(42)
        x = np.random.randn(500)
        y = np.random.randn(500)
        ce = conditional_entropy(x, y)
        je = joint_entropy(x, y)
        assert ce <= je + 0.01  # H(X|Y) <= H(X,Y)


class TestJointEntropy:
    def test_positive(self):
        np.random.seed(42)
        x = np.random.randn(500)
        y = np.random.randn(500)
        je = joint_entropy(x, y)
        assert je > 0


class TestKLDivergence:
    def test_same_distribution(self):
        np.random.seed(42)
        x = np.random.randn(500)
        kl = kl_divergence(x, x)
        assert kl == pytest.approx(0.0, abs=0.01)

    def test_different_distributions(self):
        np.random.seed(42)
        x = np.random.randn(500)
        y = np.random.randn(500) + 5  # Shifted
        kl = kl_divergence(x, y)
        assert kl > 0


class TestJSDivergence:
    def test_bounded(self):
        np.random.seed(42)
        x = np.random.randn(500)
        y = np.random.randn(500) + 5
        js = js_divergence(x, y)
        assert 0 <= js <= 1

    def test_symmetric(self):
        np.random.seed(42)
        x = np.random.randn(500)
        y = np.random.randn(500)
        js_xy = js_divergence(x, y)
        js_yx = js_divergence(y, x)
        assert js_xy == pytest.approx(js_yx, abs=0.01)


class TestInformationGain:
    def test_returns_float(self):
        np.random.seed(42)
        x = np.random.randn(500)
        y = np.random.randn(500)
        ig = information_gain(x, y)
        assert isinstance(ig, float)
