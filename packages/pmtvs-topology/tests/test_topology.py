"""Tests for pmtvs-topology."""
import numpy as np
import pytest

from pmtvs_topology import (
    distance_matrix,
    persistent_homology_0d,
    persistent_homology_1d,
    persistent_homology,
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


class TestPersistentHomology1D:
    def test_circle_has_one_loop(self):
        """Unit circle: exactly one persistent H1 feature (the loop)."""
        np.random.seed(42)
        t = np.linspace(0, 2 * np.pi, 200, endpoint=False)
        circle = np.column_stack([np.cos(t), np.sin(t)])
        circle += np.random.randn(*circle.shape) * 0.05
        h1 = persistent_homology_1d(circle)
        persistent = [p for p in h1 if (p[1] - p[0] > 0.1 if np.isfinite(p[1]) else True)]
        assert len(persistent) == 1, f"Circle should have exactly 1 persistent loop, got {len(persistent)}"

    def test_circle_loop_born_at_neighbor_scale(self):
        """Circle loop birth should be near the nearest-neighbor distance."""
        np.random.seed(42)
        t = np.linspace(0, 2 * np.pi, 100, endpoint=False)
        circle = np.column_stack([np.cos(t), np.sin(t)])
        h1 = persistent_homology_1d(circle)
        persistent = [p for p in h1 if (p[1] - p[0] > 0.1 if np.isfinite(p[1]) else True)]
        assert len(persistent) >= 1
        # Birth ~ 2*pi/100 ≈ 0.063. Should be small relative to diameter (2.0).
        birth = persistent[0][0]
        assert birth < 0.2, f"Loop birth {birth:.3f} should be near neighbor scale, not diameter"

    def test_scattered_no_persistent_loops(self):
        """Random scattered points should have no persistent H1 features."""
        np.random.seed(42)
        scattered = np.random.randn(50, 3) * 10
        h1 = persistent_homology_1d(scattered)
        persistent = [p for p in h1 if (p[1] - p[0] > 5.0 if np.isfinite(p[1]) else True)]
        assert len(persistent) == 0

    def test_two_concentric_circles_two_loops(self):
        """Two concentric circles should have at least 2 persistent loops."""
        np.random.seed(42)
        t = np.linspace(0, 2 * np.pi, 80, endpoint=False)
        inner = np.column_stack([np.cos(t), np.sin(t)])
        outer = np.column_stack([3 * np.cos(t), 3 * np.sin(t)])
        points = np.vstack([inner, outer])
        h1 = persistent_homology_1d(points, max_edge_length=2.0)
        persistent = [p for p in h1 if (p[1] - p[0] > 0.1 if np.isfinite(p[1]) else True)]
        assert len(persistent) >= 2, f"Two circles should have >= 2 loops, got {len(persistent)}"

    def test_collinear_points_no_loops(self):
        """Points on a line have no loops (Betti_1 = 0)."""
        points = np.column_stack([np.linspace(0, 10, 50), np.zeros(50)])
        h1 = persistent_homology_1d(points)
        persistent = [p for p in h1 if (p[1] - p[0] > 0.5 if np.isfinite(p[1]) else True)]
        assert len(persistent) == 0, f"Line should have 0 loops, got {len(persistent)}"


class TestPersistentHomologyUnified:
    def test_circle_h0_one_component_h1_one_loop(self):
        """Circle: H0 has one inf-persistence component, H1 has one persistent loop."""
        np.random.seed(42)
        t = np.linspace(0, 2 * np.pi, 100, endpoint=False)
        circle = np.column_stack([np.cos(t), np.sin(t)])
        result = persistent_homology(circle)
        # H0: exactly one component with infinite persistence
        inf_h0 = [p for p in result['h0'] if p[1] == np.inf]
        assert len(inf_h0) == 1
        # H1: at least one persistent loop
        persistent_h1 = [p for p in result['h1']
                         if (p[1] - p[0] > 0.1 if np.isfinite(p[1]) else True)]
        assert len(persistent_h1) >= 1


class TestBottleneckDistance:
    def test_same_diagram(self):
        ph = [(0, 1), (0, 2), (0, 3)]
        assert bottleneck_distance(ph, ph) == pytest.approx(0.0)

    def test_different_diagrams(self):
        ph1 = [(0, 1), (0, 2)]
        ph2 = [(0, 3), (0, 4)]
        d = bottleneck_distance(ph1, ph2)
        assert d > 0
