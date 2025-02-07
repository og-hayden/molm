"""
Module implementing the core token modification mechanism.

This module implements a two-stage concept-aware token modification process:
1. First-order concept gathering for individual tokens
2. Second-order concept formation across the sequence
3. Combined modification using both concept levels

Example:
    For the phrase "Time flows like a river":
    
    1. First-order concepts:
        "Time" -> [Temporality (0.8), Continuity (0.6)]
        "flows" -> [Movement (0.9), Fluidity (0.7)]
        "river" -> [Fluidity (0.9), Nature (0.8)]
    
    2. Second-order formation:
        - Aggregates related concepts (Fluidity, Movement, Continuity)
        - Creates higher-order concept like "Continuous Flow"
    
    3. Modification:
        - Each token modified by its first-order concepts
        - All tokens influenced by second-order concept
        - Weighting based on concept relevance and participation
"""
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from ..concepts.concept_space import ConceptSpace
from ..embeddings.nomic_embedder import NomicEmbedder


class TokenModifier:
    """Class implementing the concept-aware token modification mechanism.
    
    This class modifies token embeddings using both first-order (token-specific)
    and second-order (sequence-level) concept information. The modification process:
    1. Identifies relevant concepts for each token
    2. Forms higher-order concepts from concept overlap
    3. Modifies tokens using both concept levels
    
    Attributes:
        embedder: NomicEmbedder instance for token embeddings
        concept_space: ConceptSpace instance for concept management
        first_order_weight: Weight for first-order concept modifications
        second_order_weight: Weight for second-order concept modifications
        top_k_concepts: Number of top concepts to consider per token
    """

    def __init__(
        self,
        embedder: NomicEmbedder,
        concept_space: ConceptSpace,
        first_order_weight: float = 0.3,
        second_order_weight: float = 0.2,
        top_k_concepts: int = 5,
    ) -> None:
        """Initialize the TokenModifier.

        Args:
            embedder: NomicEmbedder instance for token embeddings.
            concept_space: ConceptSpace instance for concept management.
            first_order_weight: Weight for first-order concept modifications.
            second_order_weight: Weight for second-order concept modifications.
            top_k_concepts: Number of top concepts to consider per token.
        """
        self.embedder = embedder
        self.concept_space = concept_space
        self.first_order_weight = first_order_weight
        self.second_order_weight = second_order_weight
        self.top_k_concepts = top_k_concepts

    def get_token_embeddings(self, text: str) -> List[NDArray[np.float32]]:
        """Get embeddings for individual tokens in text.

        Args:
            text: Input text to tokenize and embed.

        Returns:
            List of token embeddings.
        """
        # Simple whitespace tokenization for now
        tokens = text.split()
        return [self.embedder.get_embedding(token) for token in tokens]

    def get_first_order_concepts(
        self, token_embeddings: List[NDArray[np.float32]]
    ) -> List[Dict[str, float]]:
        """Get first-order concepts for each token.

        This method identifies the most relevant concepts for each token by:
        1. Projecting the token embedding into concept space
        2. Computing cosine similarity with all concept embeddings
        3. Selecting top-k most relevant concepts
        
        Example:
            For token "river":
            {
                "Fluidity": 0.9,
                "Nature": 0.8,
                "Water": 0.7
            }

        Args:
            token_embeddings: List of token embeddings.

        Returns:
            List of dictionaries mapping concepts to relevance scores.
        """
        first_order_concepts = []
        
        # If concept space is empty, return empty concepts for all tokens
        if not self.concept_space.concept_embeddings:
            return [{} for _ in token_embeddings]
        
        for token_emb in token_embeddings:
            # Calculate similarity with all concept embeddings
            concept_similarities = {}
            for concept, concept_emb in self.concept_space.concept_embeddings.items():
                similarity = np.dot(token_emb, concept_emb) / (
                    np.linalg.norm(token_emb) * np.linalg.norm(concept_emb)
                )
                # Only include concepts with positive similarity
                if similarity > 0:
                    concept_similarities[concept] = similarity

            # Get top-k concepts
            top_concepts = dict(
                sorted(
                    concept_similarities.items(),
                    key=lambda x: x[1],
                    reverse=True,
                )[:self.top_k_concepts]
            )
            
            first_order_concepts.append(top_concepts)

        return first_order_concepts

    def get_second_order_concept(
        self, first_order_concepts: List[Dict[str, float]]
    ) -> Tuple[NDArray[np.float32], List[float]]:
        """Calculate second-order concept and participation weights.

        This method:
        1. Aggregates first-order concepts across all tokens
        2. Weights concepts by their relevance scores
        3. Forms a single higher-order concept vector
        4. Calculates each token's participation in this concept

        Example:
            For "Time flows like a river":
            - Identifies overlap in Fluidity/Movement/Continuity concepts
            - Forms higher-order "Continuous Flow" concept
            - Assigns high participation to "flows" and "river"

        Args:
            first_order_concepts: List of first-order concept dictionaries.

        Returns:
            Tuple of (second_order_concept_embedding, participation_weights).
        """
        # Handle empty concept case
        if not any(concepts for concepts in first_order_concepts):
            # Return zero vector of correct dimension (from first available embedding)
            dim = next(iter(self.concept_space.concept_embeddings.values())).shape[0] \
                if self.concept_space.concept_embeddings else 768  # Default dimension
            return (
                np.zeros(dim, dtype=np.float32),
                [1.0 / len(first_order_concepts)] * len(first_order_concepts)
            )

        # Calculate weighted sum of concept embeddings
        weighted_concepts = []
        for concepts in first_order_concepts:
            for concept, score in concepts.items():
                concept_emb = self.concept_space.get_concept_embedding(concept)
                weighted_concepts.append(concept_emb * score)

        if not weighted_concepts:  # Safeguard against empty list
            dim = next(iter(self.concept_space.concept_embeddings.values())).shape[0]
            second_order = np.zeros(dim, dtype=np.float32)
        else:
            second_order = np.mean(weighted_concepts, axis=0)

        # Calculate participation weights
        participation_weights = []
        for concepts in first_order_concepts:
            weight = np.mean(list(concepts.values())) if concepts else 0.0
            participation_weights.append(weight)

        # Normalize participation weights
        weights_sum = sum(participation_weights)
        if weights_sum > 0:
            participation_weights = [w / weights_sum for w in participation_weights]
        else:
            participation_weights = [1.0 / len(first_order_concepts)] * len(first_order_concepts)

        return second_order, participation_weights

    def modify_tokens(
        self, token_embeddings: List[NDArray[np.float32]]
    ) -> List[NDArray[np.float32]]:
        """Modify token embeddings using concept-aware mechanism.

        This method applies a two-stage modification:
        1. First-Order: Each token modified by its specific concepts
        2. Second-Order: All tokens influenced by sequence-level concept

        Example modification process:
            1. Token "river" gets modified by:
               - First-order: Fluidity (0.9) and Nature (0.8) concepts
               - Second-order: "Continuous Flow" concept weighted by participation
            2. Result maintains core meaning while enhancing semantic coherence

        Args:
            token_embeddings: List of token embeddings to modify.

        Returns:
            List of modified token embeddings.
        """
        # Get first-order concepts
        first_order = self.get_first_order_concepts(token_embeddings)

        # Get second-order concept and participation weights
        second_order, participation = self.get_second_order_concept(first_order)

        # Apply modifications
        modified_embeddings = []
        for token_emb, concepts, weight in zip(
            token_embeddings, first_order, participation
        ):
            # First-order modification
            first_order_mod = np.zeros_like(token_emb)
            for concept, score in concepts.items():
                concept_emb = self.concept_space.get_concept_embedding(concept)
                first_order_mod += concept_emb * score

            # Apply both modifications
            modified = (
                token_emb
                + (first_order_mod * self.first_order_weight)
                + (second_order * self.second_order_weight * weight)
            )

            # Normalize the result
            norm = np.linalg.norm(modified)
            if norm > 0:
                modified = modified / norm
            else:
                modified = token_emb  # Fall back to original embedding if modification fails

            modified_embeddings.append(modified)

        return modified_embeddings 