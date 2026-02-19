# pmtvs-network

Network analysis primitives.

## Installation

```bash
pip install pmtvs-network
```

## Functions

### Centrality
- `degree_centrality(adjacency)` - Degree centrality
- `betweenness_centrality(adjacency)` - Betweenness centrality
- `closeness_centrality(adjacency)` - Closeness centrality

### Structure
- `clustering_coefficient(adjacency)` - Local clustering
- `average_path_length(adjacency)` - Mean shortest path
- `density(adjacency)` - Graph density
- `connected_components(adjacency)` - Find components
- `adjacency_from_correlation(corr, threshold)` - Build network from correlation

### Community Detection
- `modularity(adjacency, communities)` - Newman-Girvan modularity
- `community_detection(adjacency, method)` - Louvain, spectral, or label propagation

## Backend

Pure Python implementation.
