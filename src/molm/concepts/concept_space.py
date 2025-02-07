"""
Module for managing concept vectors and their relationships using ConceptNet.
"""
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np
import requests
from numpy.typing import NDArray

from ..embeddings.nomic_embedder import NomicEmbedder


class ConceptSpace:
    """A class to manage concept vectors and their relationships."""

    CONCEPTNET_API = "http://api.conceptnet.io"
    RELEVANT_RELATIONS = {
        "RelatedTo",
        "IsA",
        "PartOf",
        "HasA",
        "UsedFor",
        "CapableOf",
        "AtLocation",
        "Causes",
        "HasProperty",
    }

    def __init__(
        self,
        embedder: NomicEmbedder,
        max_concepts: int = 1000,
        min_weight: float = 0.1,
    ) -> None:
        """Initialize the ConceptSpace.

        Args:
            embedder: NomicEmbedder instance for generating concept embeddings.
            max_concepts: Maximum number of base concepts to maintain.
            min_weight: Minimum weight for concept relationships.
        """
        self.embedder = embedder
        self.max_concepts = max_concepts
        self.min_weight = min_weight
        
        self.concept_graph = nx.DiGraph()
        self.concept_embeddings: Dict[str, NDArray[np.float32]] = {}
        
    def _get_conceptnet_edges(self, concept: str) -> List[Tuple[str, str, float]]:
        """Get related concepts and their relationships from ConceptNet.

        Args:
            concept: The concept to query.

        Returns:
            List of (source, target, weight) tuples.
        """
        edges = []
        url = f"{self.CONCEPTNET_API}/c/en/{concept}?limit=1000"
        
        try:
            response = requests.get(url)
            if response.status_code != 200:
                return edges

            data = response.json()
            for edge in data.get("edges", []):
                rel = edge.get("rel", {}).get("label")
                if rel not in self.RELEVANT_RELATIONS:
                    continue

                start = edge.get("start", {}).get("label", "").lower()
                end = edge.get("end", {}).get("label", "").lower()
                weight = edge.get("weight", 0.0)

                if weight >= self.min_weight and start and end:
                    edges.append((start, end, weight))

        except requests.RequestException:
            return edges

        return edges

    def add_concept(self, concept: str) -> None:
        """Add a concept and its relationships to the concept space.

        Args:
            concept: The concept to add.
        """
        if concept in self.concept_embeddings:
            return

        # Get concept embedding
        embedding = self.embedder.get_embedding(concept)
        self.concept_embeddings[concept] = embedding

        # Get related concepts from ConceptNet
        edges = self._get_conceptnet_edges(concept)
        for source, target, weight in edges:
            self.concept_graph.add_edge(source, target, weight=weight)

            # Get embeddings for related concepts if needed
            for node in (source, target):
                if node not in self.concept_embeddings:
                    self.concept_embeddings[node] = self.embedder.get_embedding(node)

        # Prune if we exceed max concepts
        if len(self.concept_embeddings) > self.max_concepts:
            self._prune_concepts()

    def _prune_concepts(self) -> None:
        """Prune the concept space to maintain the maximum size."""
        # Keep concepts with highest degree centrality
        centrality = nx.degree_centrality(self.concept_graph)
        sorted_concepts = sorted(
            centrality.items(), key=lambda x: x[1], reverse=True
        )
        
        keep_concepts = {c[0] for c in sorted_concepts[:self.max_concepts]}
        
        # Remove concepts not in keep_concepts
        remove_concepts = set(self.concept_embeddings.keys()) - keep_concepts
        for concept in remove_concepts:
            del self.concept_embeddings[concept]
            self.concept_graph.remove_node(concept)

    def get_related_concepts(
        self, concept: str, max_depth: int = 2
    ) -> Dict[str, float]:
        """Get related concepts up to a certain depth with their relevance scores.

        Args:
            concept: The concept to start from.
            max_depth: Maximum depth to traverse in the concept graph.

        Returns:
            Dictionary mapping related concepts to their relevance scores.
        """
        if concept not in self.concept_graph:
            self.add_concept(concept)

        related = {}
        visited = {concept}
        queue = [(concept, 0)]

        while queue:
            current, depth = queue.pop(0)
            if depth >= max_depth:
                continue

            for _, neighbor in self.concept_graph.edges(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, depth + 1))
                    weight = self.concept_graph[current][neighbor]["weight"]
                    decay = 1.0 / (depth + 1)
                    related[neighbor] = weight * decay

        return related

    def get_concept_embedding(self, concept: str) -> NDArray[np.float32]:
        """Get the embedding for a concept.

        Args:
            concept: The concept to get embedding for.

        Returns:
            The concept's embedding vector.
        """
        if concept not in self.concept_embeddings:
            self.add_concept(concept)
        return self.concept_embeddings[concept] 