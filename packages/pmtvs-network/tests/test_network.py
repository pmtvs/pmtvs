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
    modularity,
    community_detection,
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


class TestModularity:
    """Tests for modularity scoring."""

    def test_perfect_partition(self):
        """Two disconnected cliques with correct labels should have high modularity."""
        np.random.seed(42)
        # Two disconnected cliques of size 3
        adj = np.array([
            [0, 1, 1, 0, 0, 0],
            [1, 0, 1, 0, 0, 0],
            [1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1],
            [0, 0, 0, 1, 0, 1],
            [0, 0, 0, 1, 1, 0],
        ], dtype=float)
        communities = np.array([0, 0, 0, 1, 1, 1])
        Q = modularity(adj, communities)
        assert Q > 0.3

    def test_single_community(self):
        """All nodes in one community should give Q = 0."""
        adj = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=float)
        communities = np.array([0, 0, 0])
        Q = modularity(adj, communities)
        assert Q == pytest.approx(0.0, abs=1e-10)

    def test_empty_graph(self):
        """Empty adjacency matrix should return NaN."""
        adj = np.array([]).reshape(0, 0)
        communities = np.array([])
        assert np.isnan(modularity(adj, communities))

    def test_no_edges(self):
        """Graph with no edges should return 0."""
        adj = np.zeros((4, 4))
        communities = np.array([0, 0, 1, 1])
        Q = modularity(adj, communities)
        assert Q == pytest.approx(0.0)

    def test_resolution_parameter(self):
        """Higher resolution should produce lower modularity for same partition."""
        np.random.seed(42)
        adj = np.array([
            [0, 1, 1, 0, 0, 0],
            [1, 0, 1, 0, 0, 0],
            [1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1],
            [0, 0, 0, 1, 0, 1],
            [0, 0, 0, 1, 1, 0],
        ], dtype=float)
        communities = np.array([0, 0, 0, 1, 1, 1])
        Q_low = modularity(adj, communities, resolution=0.5)
        Q_high = modularity(adj, communities, resolution=2.0)
        assert Q_low > Q_high


class TestCommunityDetection:
    """Tests for community detection."""

    def test_two_cliques_louvain(self):
        """Louvain should separate two disconnected cliques."""
        np.random.seed(42)
        adj = np.array([
            [0, 1, 1, 0, 0, 0],
            [1, 0, 1, 0, 0, 0],
            [1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1],
            [0, 0, 0, 1, 0, 1],
            [0, 0, 0, 1, 1, 0],
        ], dtype=float)
        communities, mod = community_detection(adj, method='louvain')
        # Should find 2 communities
        assert len(np.unique(communities)) == 2
        # First 3 nodes should be in same community
        assert communities[0] == communities[1] == communities[2]
        # Last 3 nodes should be in same community
        assert communities[3] == communities[4] == communities[5]
        # The two groups should be different
        assert communities[0] != communities[3]
        # Modularity should be positive
        assert mod > 0.3

    def test_returns_correct_shape(self):
        """Community labels should have same length as number of nodes."""
        np.random.seed(42)
        adj = np.ones((5, 5)) - np.eye(5)
        communities, mod = community_detection(adj, method='louvain')
        assert len(communities) == 5

    def test_empty_graph(self):
        """Empty graph should return empty communities."""
        adj = np.array([]).reshape(0, 0)
        communities, mod = community_detection(adj, method='louvain')
        assert len(communities) == 0
        assert mod == 0.0

    def test_spectral_method(self):
        """Spectral clustering should produce valid partition."""
        np.random.seed(42)
        adj = np.array([
            [0, 1, 1, 0, 0, 0],
            [1, 0, 1, 0, 0, 0],
            [1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1],
            [0, 0, 0, 1, 0, 1],
            [0, 0, 0, 1, 1, 0],
        ], dtype=float)
        communities, mod = community_detection(adj, method='spectral', n_communities=2)
        assert len(communities) == 6
        assert len(np.unique(communities)) == 2

    def test_label_propagation_method(self):
        """Label propagation should produce valid partition."""
        np.random.seed(42)
        adj = np.array([
            [0, 1, 1, 0, 0, 0],
            [1, 0, 1, 0, 0, 0],
            [1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1],
            [0, 0, 0, 1, 0, 1],
            [0, 0, 0, 1, 1, 0],
        ], dtype=float)
        communities, mod = community_detection(adj, method='label_propagation')
        assert len(communities) == 6
        # Should find at least 2 communities for disconnected graph
        assert len(np.unique(communities)) >= 2

    def test_invalid_method(self):
        """Invalid method should raise ValueError."""
        adj = np.array([[0, 1], [1, 0]], dtype=float)
        with pytest.raises(ValueError, match="Unknown method"):
            community_detection(adj, method='invalid_method')

    def test_modularity_is_float(self):
        """Returned modularity should be a float."""
        np.random.seed(42)
        adj = np.ones((4, 4)) - np.eye(4)
        _, mod = community_detection(adj, method='louvain')
        assert isinstance(mod, float)
