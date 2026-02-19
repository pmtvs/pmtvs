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
    shannon_entropy,
    renyi_entropy,
    tsallis_entropy,
    cross_entropy,
    hellinger_distance,
    total_variation_distance,
    conditional_mutual_information,
    multivariate_mutual_information,
    total_correlation,
    interaction_information,
    dual_total_correlation,
    partial_information_decomposition,
    redundancy,
    synergy,
    information_atoms,
    granger_causality,
    convergent_cross_mapping,
    phase_coupling,
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


# --- Tests for entropy.py ---


class TestShannonEntropy:
    def test_positive(self):
        np.random.seed(42)
        data = np.random.randn(500)
        h = shannon_entropy(data)
        assert h > 0

    def test_short_data(self):
        h = shannon_entropy([1.0])
        assert np.isnan(h)

    def test_base_change(self):
        np.random.seed(42)
        data = np.random.randn(500)
        h2 = shannon_entropy(data, base=2)
        he = shannon_entropy(data, base=np.e)
        # H_e = H_2 * ln(2) / ln(e) = H_2 * ln(2)
        assert he == pytest.approx(h2 * np.log(2), rel=0.01)


class TestRenyiEntropy:
    def test_positive(self):
        np.random.seed(42)
        data = np.random.randn(500)
        h = renyi_entropy(data, alpha=2.0)
        assert h > 0

    def test_alpha_one_equals_shannon(self):
        np.random.seed(42)
        data = np.random.randn(500)
        h_r = renyi_entropy(data, alpha=1.0)
        h_s = shannon_entropy(data)
        assert h_r == pytest.approx(h_s, rel=0.01)

    def test_short_data(self):
        h = renyi_entropy([1.0])
        assert np.isnan(h)


class TestTsallisEntropy:
    def test_positive(self):
        np.random.seed(42)
        data = np.random.randn(500)
        h = tsallis_entropy(data, q=2.0)
        assert h > 0

    def test_short_data(self):
        h = tsallis_entropy([1.0])
        assert np.isnan(h)


# --- Tests for divergence.py ---


class TestCrossEntropy:
    def test_finite(self):
        np.random.seed(42)
        p = np.random.randn(500)
        q = np.random.randn(500)
        ce = cross_entropy(p, q)
        assert np.isfinite(ce)

    def test_same_distribution(self):
        np.random.seed(42)
        p = np.random.randn(500)
        ce = cross_entropy(p, p)
        assert np.isfinite(ce)


class TestHellingerDistance:
    def test_bounded(self):
        np.random.seed(42)
        p = np.random.randn(500)
        q = np.random.randn(500) + 5
        hd = hellinger_distance(p, q)
        assert 0 <= hd <= 1

    def test_same_distribution_near_zero(self):
        np.random.seed(42)
        p = np.random.randn(500)
        hd = hellinger_distance(p, p)
        assert hd == pytest.approx(0.0, abs=0.01)


class TestTotalVariationDistance:
    def test_bounded(self):
        np.random.seed(42)
        p = np.random.randn(500)
        q = np.random.randn(500) + 5
        tv = total_variation_distance(p, q)
        assert 0 <= tv <= 1

    def test_same_distribution_near_zero(self):
        np.random.seed(42)
        p = np.random.randn(500)
        tv = total_variation_distance(p, p)
        assert tv == pytest.approx(0.0, abs=0.01)


# --- Tests for mutual.py ---


class TestConditionalMutualInformation:
    def test_finite(self):
        np.random.seed(42)
        x = np.random.randn(500)
        y = np.random.randn(500)
        z = np.random.randn(500)
        cmi = conditional_mutual_information(x, y, z)
        assert np.isfinite(cmi)

    def test_correlated(self):
        np.random.seed(42)
        x = np.random.randn(500)
        y = x + 0.1 * np.random.randn(500)
        z = np.random.randn(500)
        cmi = conditional_mutual_information(x, y, z)
        assert cmi > 0


class TestMultivariateMutualInformation:
    def test_two_variables(self):
        np.random.seed(42)
        x = np.random.randn(500)
        y = x + 0.1 * np.random.randn(500)
        mmi = multivariate_mutual_information([x, y])
        assert mmi > 0

    def test_three_variables(self):
        np.random.seed(42)
        x = np.random.randn(500)
        y = np.random.randn(500)
        z = np.random.randn(500)
        mmi = multivariate_mutual_information([x, y, z])
        assert np.isfinite(mmi)


class TestTotalCorrelation:
    def test_non_negative(self):
        np.random.seed(42)
        x = np.random.randn(500)
        y = np.random.randn(500)
        z = np.random.randn(500)
        tc = total_correlation([x, y, z])
        assert tc >= -0.1  # Allow small numerical error


class TestInteractionInformation:
    def test_returns_float(self):
        np.random.seed(42)
        x = np.random.randn(500)
        y = np.random.randn(500)
        z = np.random.randn(500)
        ii = interaction_information([x, y, z])
        assert isinstance(ii, float)


class TestDualTotalCorrelation:
    def test_returns_float(self):
        np.random.seed(42)
        x = np.random.randn(500)
        y = np.random.randn(500)
        z = np.random.randn(500)
        dtc = dual_total_correlation([x, y, z])
        assert isinstance(dtc, float)


# --- Tests for decomposition.py ---


class TestPartialInformationDecomposition:
    def test_returns_dict(self):
        np.random.seed(42)
        s1 = np.random.randn(500)
        s2 = np.random.randn(500)
        t = s1 + s2 + 0.1 * np.random.randn(500)
        pid = partial_information_decomposition(s1, s2, t)
        assert 'redundancy' in pid
        assert 'unique_1' in pid
        assert 'unique_2' in pid
        assert 'synergy' in pid
        assert 'total_mi' in pid

    def test_non_negative_components(self):
        np.random.seed(42)
        s1 = np.random.randn(500)
        s2 = np.random.randn(500)
        t = s1 + s2
        pid = partial_information_decomposition(s1, s2, t)
        assert pid['redundancy'] >= 0
        assert pid['unique_1'] >= 0
        assert pid['unique_2'] >= 0
        assert pid['synergy'] >= 0


class TestRedundancy:
    def test_returns_float(self):
        np.random.seed(42)
        s1 = np.random.randn(500)
        s2 = np.random.randn(500)
        t = s1 + 0.1 * np.random.randn(500)
        r = redundancy([s1, s2], t)
        assert isinstance(r, float)
        assert r >= 0


class TestSynergy:
    def test_returns_float(self):
        np.random.seed(42)
        s1 = np.random.randn(500)
        s2 = np.random.randn(500)
        t = s1 + s2
        s = synergy([s1, s2], t)
        assert isinstance(s, float)
        assert s >= 0


class TestInformationAtoms:
    def test_returns_dict(self):
        np.random.seed(42)
        s1 = np.random.randn(500)
        s2 = np.random.randn(500)
        t = s1 + s2
        atoms = information_atoms([s1, s2], t)
        assert 'redundancy' in atoms
        assert 'synergy' in atoms
        assert 'total_mi' in atoms
        assert 'emergence_ratio' in atoms


# --- Tests for causality.py ---


class TestGrangerCausality:
    def test_causal_signal(self):
        np.random.seed(42)
        x = np.random.randn(200)
        y = np.zeros(200)
        y[1:] = 0.8 * x[:-1] + 0.2 * np.random.randn(199)
        f_stat, p_val = granger_causality(x, y, max_lag=3)
        assert np.isfinite(f_stat)
        assert p_val < 0.05

    def test_no_causality(self):
        np.random.seed(42)
        x = np.random.randn(200)
        y = np.random.randn(200)
        f_stat, p_val = granger_causality(x, y, max_lag=3)
        assert np.isfinite(f_stat)
        assert p_val > 0.01


class TestConvergentCrossMapping:
    def test_returns_tuple(self):
        np.random.seed(42)
        x = np.random.randn(100)
        y = np.random.randn(100)
        rho_a, rho_b = convergent_cross_mapping(x, y)
        assert isinstance(rho_a, float)
        assert isinstance(rho_b, float)

    def test_short_signal(self):
        rho_a, rho_b = convergent_cross_mapping([1, 2], [3, 4])
        assert np.isnan(rho_a)
        assert np.isnan(rho_b)


class TestPhaseCoupling:
    def test_identical_signals(self):
        np.random.seed(42)
        t = np.linspace(0, 10, 500)
        s = np.sin(2 * np.pi * 5 * t)
        plv = phase_coupling(s, s)
        assert plv == pytest.approx(1.0, abs=0.05)

    def test_bounded(self):
        np.random.seed(42)
        s1 = np.random.randn(500)
        s2 = np.random.randn(500)
        plv = phase_coupling(s1, s2)
        assert 0 <= plv <= 1
