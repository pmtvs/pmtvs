"""Tests for pmtvs-topology."""
import numpy as np
import pytest

from pmtvs_topology import (
    distance_matrix,
    persistent_homology_0d,
    betti_numbers,
    persistence_entropy,
    persistence_landscape,
    bottleneck_distance,
)


class TestDistanceMatrix:
    def test_shape(self):
        points = np.random.randn(10, 3)
        dist = distance_matrix(points)
        assert dist.shape == (10, 10)

    def test_symmetric(self):
        points = np.random.randn(10, 3)
        dist = distance_matrix(points)
        assert np.allclose(dist, dist.T)

    def test_zero_diagonal(self):
        points = np.random.randn(10, 3)
        dist = distance_matrix(points)
        assert np.allclose(np.diag(dist), 0)


class TestPersistentHomology:
    def test_returns_list(self):
        points = np.random.randn(20, 2)
        ph = persistent_homology_0d(points)
        assert isinstance(ph, list)
        assert len(ph) > 0

    def test_one_cluster(self):
        # Tight cluster should have one long-lived component
        points = np.random.randn(20, 2) * 0.01
        ph = persistent_homology_0d(points)
        # At least one infinite persistence
        infinite = [p for p in ph if p[1] == np.inf]
        assert len(infinite) == 1


class TestBettiNumbers:
    def test_counts(self):
        # Two separate clusters
        points = np.array([[0, 0], [0.1, 0], [10, 0], [10.1, 0]])
        ph = persistent_homology_0d(points)

        # At small threshold, should be 4 components
        assert betti_numbers(ph, 0.05) >= 2

        # At large threshold, should be 1 component
        assert betti_numbers(ph, 15) == 1


class TestPersistenceEntropy:
    def test_non_negative(self):
        points = np.random.randn(20, 2)
        ph = persistent_homology_0d(points)
        ent = persistence_entropy(ph)
        assert ent >= 0


class TestPersistenceLandscape:
    def test_returns_arrays(self):
        points = np.random.randn(20, 2)
        ph = persistent_homology_0d(points)
        x, y = persistence_landscape(ph, k=1)
        assert len(x) == len(y)


class TestBottleneckDistance:
    def test_same_diagram(self):
        ph = [(0, 1), (0, 2), (0, 3)]
        assert bottleneck_distance(ph, ph) == pytest.approx(0.0)

    def test_different_diagrams(self):
        ph1 = [(0, 1), (0, 2)]
        ph2 = [(0, 3), (0, 4)]
        d = bottleneck_distance(ph1, ph2)
        assert d > 0
