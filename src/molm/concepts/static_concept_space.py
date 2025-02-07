"""
Module for managing a static set of pre-computed concept vectors.
"""
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict

import numpy as np
import torch
from numpy.typing import NDArray
import pickle

from ..embeddings.nomic_embedder import NomicEmbedder


class StaticConceptSpace:
    """A class to manage pre-computed concept vectors and relationships.
    
    This class loads pre-computed concept embeddings and relationships from disk,
    making it much more efficient than computing them on the fly. The concept
    space is organized into domains and abstraction levels.
    
    Attributes:
        concept_embeddings: Dictionary mapping concepts to their embeddings
        concept_relationships: Dictionary mapping concepts to their relationships
        domains: Dictionary mapping domains to their concepts
        abstraction_levels: Dictionary mapping levels to their concepts
    """

    def __init__(
        self,
        embedder: Optional[NomicEmbedder] = None,
        concepts_dir: Optional[str] = None,
    ) -> None:
        """Initialize the StaticConceptSpace.

        Args:
            embedder: Optional NomicEmbedder for computing new embeddings
            concepts_dir: Directory containing pre-computed concept data
        """
        self.embedder = embedder
        self.concepts_dir = Path(concepts_dir) if concepts_dir else Path(__file__).parent / "data"
        
        # Load pre-computed data
        self.concept_embeddings: Dict[str, NDArray[np.float32]] = {}
        self.concept_relationships: Dict[str, Dict[str, float]] = {}
        self.domains: Dict[str, Set[str]] = {}
        self.abstraction_levels: Dict[str, Set[str]] = {}
        
        self._load_concept_data()

    def _load_concept_data(self) -> None:
        """Load pre-computed concept data from disk."""
        # Try regular files first
        embeddings_file = self.concepts_dir / "embeddings.pt"
        relationships_file = self.concepts_dir / "relationships.pt"
        
        # Check checkpoint files if regular files don't exist or are corrupted
        checkpoint_dir = self.concepts_dir / "checkpoints"
        embeddings_checkpoint = checkpoint_dir / "embeddings.pkl"
        relationships_checkpoint = checkpoint_dir / "relationships.pkl"
        abstraction_checkpoint = checkpoint_dir / "abstraction_levels.pkl"
        wordnet_checkpoint = checkpoint_dir / "wordnet.pkl"
        
        # Load embeddings
        print("Loading concept embeddings...")
        try:
            embeddings_dict = torch.load(embeddings_file)
        except:
            print("Using embeddings checkpoint...")
            with open(embeddings_checkpoint, "rb") as f:
                embeddings_dict = pickle.load(f)["embeddings"]
        
        self.concept_embeddings = {
            concept: emb.numpy() if isinstance(emb, torch.Tensor) else emb
            for concept, emb in embeddings_dict.items()
        }
        
        # Load relationships
        print("Loading concept relationships...")
        try:
            self.concept_relationships = torch.load(relationships_file)
        except:
            print("Using relationships checkpoint...")
            with open(relationships_checkpoint, "rb") as f:
                self.concept_relationships = pickle.load(f)["relationships"]
        
        # Load abstraction levels and domains
        print("Loading concept metadata...")
        try:
            with open(abstraction_checkpoint, "rb") as f:
                abstraction_data = pickle.load(f)
                self.abstraction_levels = defaultdict(set, abstraction_data["levels"])
            with open(wordnet_checkpoint, "rb") as f:
                wordnet_data = pickle.load(f)
                self.domains = defaultdict(set, wordnet_data["domains"])
            print("Loaded concept metadata from checkpoints")
        except Exception as e:
            print(f"Warning: Could not load concept metadata: {e}")
            self.domains = defaultdict(set)
            self.abstraction_levels = defaultdict(set)
        
        print(f"Loaded {len(self.concept_embeddings)} concepts with "
              f"{sum(len(r) for r in self.concept_relationships.values())} relationships")

    def get_concept_embedding(self, concept: str) -> NDArray[np.float32]:
        """Get the embedding for a concept.

        Args:
            concept: The concept to get embedding for.

        Returns:
            The concept's embedding vector.
            
        Raises:
            KeyError: If the concept is not in the static space.
        """
        try:
            return self.concept_embeddings[concept]
        except KeyError:
            if self.embedder:
                # Compute embedding on the fly if embedder is available
                return self.embedder.get_embedding(concept)
            raise KeyError(f"Concept '{concept}' not found in static space")

    def get_related_concepts(
        self, concept: str, max_depth: int = 2
    ) -> Dict[str, float]:
        """Get related concepts up to a certain depth with their relevance scores.

        Args:
            concept: The concept to start from.
            max_depth: Maximum depth to traverse in relationships.

        Returns:
            Dictionary mapping related concepts to their relevance scores.
        """
        if concept not in self.concept_relationships:
            return {}
            
        # Start with direct relationships
        related = self.concept_relationships[concept].copy()
        
        # Add transitive relationships if depth > 1
        if max_depth > 1:
            visited = {concept}
            current_depth = 1
            current_concepts = list(related.keys())
            
            while current_depth < max_depth and current_concepts:
                next_concepts = []
                for c in current_concepts:
                    if c in visited:
                        continue
                    visited.add(c)
                    
                    if c in self.concept_relationships:
                        for next_c, score in self.concept_relationships[c].items():
                            if next_c not in related:
                                # Decay score based on depth
                                related[next_c] = score / (current_depth + 1)
                                next_concepts.append(next_c)
                
                current_concepts = next_concepts
                current_depth += 1
        
        return related

    def get_concepts_by_domain(self, domain: str) -> Set[str]:
        """Get all concepts in a specific domain.

        Args:
            domain: The domain to get concepts for.

        Returns:
            Set of concepts in the domain.
        """
        return self.domains.get(domain, set())

    def get_concepts_by_level(self, level: str) -> Set[str]:
        """Get all concepts at a specific abstraction level.

        Args:
            level: The abstraction level to get concepts for.

        Returns:
            Set of concepts at the level.
        """
        return self.abstraction_levels.get(level, set())

    def get_concept_info(self, concept: str) -> Dict[str, str]:
        """Get metadata about a concept.

        Args:
            concept: The concept to get info for.

        Returns:
            Dictionary with concept metadata (domain, level, etc.).
        """
        info = {}
        
        # Find domain
        for domain, concepts in self.domains.items():
            if concept in concepts:
                info["domain"] = domain
                break
                
        # Find abstraction level
        for level, concepts in self.abstraction_levels.items():
            if concept in concepts:
                info["abstraction_level"] = level
                break
        
        return info 