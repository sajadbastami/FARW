import random
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec


class FARW:
    """
    Feature-Aware Random Walk (FARW) for node representation learning.

    This implementation follows the main idea of the paper:
    - combine graph topology with node attribute similarity
    - guide random walks using cosine similarity
    - learn node embeddings using Word2Vec over generated walks
    """

    def __init__(
        self,
        graph: nx.Graph,
        features: Dict[str, np.ndarray],
        walk_length: int = 40,
        num_walks: int = 10,
        embedding_dim: int = 128,
        window_size: int = 5,
        workers: int = 1,
        seed: int = 42,
        min_count: int = 0,
        epochs: int = 5,
    ):
        self.graph = graph
        self.features = features
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.workers = workers
        self.seed = seed
        self.min_count = min_count
        self.epochs = epochs

        random.seed(seed)
        np.random.seed(seed)

        self.nodes = list(self.graph.nodes())
        self.embedding_model: Optional[Word2Vec] = None
        self.walks: List[List[str]] = []

    def _cosine(self, u: str, v: str) -> float:
        """Cosine similarity between feature vectors of two nodes."""
        fu = self.features[u].reshape(1, -1)
        fv = self.features[v].reshape(1, -1)
        return float(cosine_similarity(fu, fv)[0, 0])

    def _choose_next(self, current: str, neighbors: List[str]) -> str:
        """
        Choose the next node based on cosine similarity of features.
        If all similarities are invalid, fall back to uniform random choice.
        """
        if not neighbors:
            return current

        sims = np.array([self._cosine(current, nbr) for nbr in neighbors], dtype=float)

        # Make similarities non-negative for sampling
        sims = np.maximum(sims, 0.0)

        if sims.sum() <= 0:
            return random.choice(neighbors)

        probs = sims / sims.sum()
        return np.random.choice(neighbors, p=probs)

    def generate_walk(self, start_node: str) -> List[str]:
        """Generate one feature-aware random walk starting from start_node."""
        walk = [start_node]

        while len(walk) < self.walk_length:
            current = walk[-1]
            neighbors = list(self.graph.neighbors(current))

            if not neighbors:
                break

            nxt = self._choose_next(current, neighbors)
            walk.append(nxt)

        return walk

    def generate_walks(self) -> List[List[str]]:
        """Generate multiple random walks for all nodes."""
        walks = []
        nodes = self.nodes[:]

        for _ in range(self.num_walks):
            random.shuffle(nodes)
            for node in nodes:
                walks.append(self.generate_walk(node))

        self.walks = walks
        return walks

    def fit(self) -> Word2Vec:
        """Train Word2Vec on generated walks and return the trained model."""
        if not self.walks:
            self.generate_walks()

        self.embedding_model = Word2Vec(
            sentences=self.walks,
            vector_size=self.embedding_dim,
            window=self.window_size,
            min_count=self.min_count,
            sg=1,  # Skip-gram
            workers=self.workers,
            seed=self.seed,
            epochs=self.epochs,
        )
        return self.embedding_model

    def get_embedding(self, node: str) -> np.ndarray:
        """Return the embedding vector of a node."""
        if self.embedding_model is None:
            raise ValueError("Model is not trained yet. Call fit() first.")

        return self.embedding_model.wv[str(node)]

    def get_embeddings(self) -> Dict[str, np.ndarray]:
        """Return embeddings for all nodes."""
        if self.embedding_model is None:
            raise ValueError("Model is not trained yet. Call fit() first.")

        return {node: self.embedding_model.wv[str(node)] for node in self.graph.nodes()}


def build_graph_from_edges(edge_list: List[Tuple[str, str]]) -> nx.Graph:
    """Utility function to build an undirected graph from edge list."""
    g = nx.Graph()
    g.add_edges_from(edge_list)
    return g


if __name__ == "__main__":
    # Example graph
    edges = [
        ("1", "2"),
        ("1", "3"),
        ("2", "3"),
        ("2", "4"),
        ("3", "5"),
        ("4", "5"),
    ]

    # Example node features
    features = {
        "1": np.array([1.0, 0.1, 0.0]),
        "2": np.array([0.9, 0.2, 0.1]),
        "3": np.array([0.8, 0.1, 0.2]),
        "4": np.array([0.1, 0.9, 0.8]),
        "5": np.array([0.2, 0.8, 0.7]),
    }

    graph = build_graph_from_edges(edges)

    farw = FARW(
        graph=graph,
        features=features,
        walk_length=10,
        num_walks=5,
        embedding_dim=64,
        window_size=3,
        workers=1,
        seed=42,
        epochs=10,
    )

    farw.generate_walks()
    farw.fit()

    print("Generated walks:")
    for w in farw.walks[:5]:
        print(w)

    print("\nEmbedding for node '1':")
    print(farw.get_embedding("1"))
