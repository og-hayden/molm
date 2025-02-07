"""
MOLM: Modular Token Modification in Language Models.

This package implements a focused investigation of concept-aware token modification
in transformer architectures.
"""

__version__ = "0.1.0"

from .embeddings.nomic_embedder import NomicEmbedder
from .concepts.concept_space import ConceptSpace
from .modification.token_modifier import TokenModifier

__all__ = ["NomicEmbedder", "ConceptSpace", "TokenModifier"] 