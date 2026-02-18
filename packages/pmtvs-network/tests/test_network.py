"""Tests for pmtvs-network."""
import numpy as np
import pytest

from pmtvs_network import (
    degree_centrality,
    betweenness_centrality,
    closeness_centrality,
    clustering_coefficient,
    average_path_length,
    density,
    connected_components,
    adjacency_from_correlation,
)


class TestDegreeCentrality:
    def test_bounded(self):
        adj = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
        dc = degree_centrality(adj)
        assert np.all(dc >= 0) and np.all(dc <= 1)

    def test_complete_graph(self):
        # Complete graph: all nodes have degree n-1
        adj = np.ones((4, 4)) - np.eye(4)
        dc = degree_centrality(adj)
        assert np.allclose(dc, 1.0)


class TestBetweennessCentrality:
    def test_non_negative(self):
        adj = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
        bc = betweenness_centrality(adj)
        assert np.all(bc >= 0)

    def test_star_graph(self):
        # Star graph: center has highest betweenness
        adj = np.array([
            [0, 1, 1, 1],
            [1, 0, 0, 0],
            [1, 0, 0, 0],
            [1, 0, 0, 0],
        ])
        bc = betweenness_centrality(adj)
        assert np.argmax(bc) == 0


class TestClosenessCentrality:
    def test_non_negative(self):
        adj = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
        cc = closeness_centrality(adj)
        assert np.all(cc >= 0)


class TestClusteringCoefficient:
    def test_bounded(self):
        adj = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
        cc = clustering_coefficient(adj)
        assert np.all(cc >= 0) and np.all(cc <= 1)

    def test_triangle(self):
        # Triangle: all nodes have clustering 1
        adj = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
        cc = clustering_coefficient(adj)
        assert np.allclose(cc, 1.0)


class TestAveragePathLength:
    def test_complete_graph(self):
        # Complete graph: all paths have length 1
        adj = np.ones((4, 4)) - np.eye(4)
        apl = average_path_length(adj)
        assert apl == pytest.approx(1.0)


class TestDensity:
    def test_complete_graph(self):
        adj = np.ones((4, 4)) - np.eye(4)
        d = density(adj)
        assert d == pytest.approx(1.0)

    def test_empty_graph(self):
        adj = np.zeros((4, 4))
        d = density(adj)
        assert d == pytest.approx(0.0)


class TestConnectedComponents:
    def test_single_component(self):
        adj = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
        cc = connected_components(adj)
        assert len(cc) == 1

    def test_two_components(self):
        adj = np.array([
            [0, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
        ])
        cc = connected_components(adj)
        assert len(cc) == 2


class TestAdjacencyFromCorrelation:
    def test_diagonal_zero(self):
        corr = np.array([[1, 0.8, 0.3], [0.8, 1, 0.1], [0.3, 0.1, 1]])
        adj = adjacency_from_correlation(corr, threshold=0.5)
        assert np.all(np.diag(adj) == 0)

    def test_threshold(self):
        corr = np.array([[1, 0.8, 0.3], [0.8, 1, 0.1], [0.3, 0.1, 1]])
        adj = adjacency_from_correlation(corr, threshold=0.5)
        # Only 0-1 edge should exist
        assert adj[0, 1] == 1
        assert adj[0, 2] == 0
