"""
Tests for the TokenModifier class.
"""
import numpy as np
import pytest

from molm.embeddings.nomic_embedder import NomicEmbedder
from molm.concepts.concept_space import ConceptSpace
from molm.modification.token_modifier import TokenModifier


@pytest.fixture
def embedder():
    """Fixture for NomicEmbedder instance."""
    return NomicEmbedder()


@pytest.fixture
def concept_space(embedder):
    """Fixture for ConceptSpace instance."""
    return ConceptSpace(embedder, max_concepts=10)


@pytest.fixture
def modifier(embedder, concept_space):
    """Fixture for TokenModifier instance."""
    return TokenModifier(embedder, concept_space)


def test_get_token_embeddings(modifier):
    """Test getting token embeddings."""
    text = "test token"
    embeddings = modifier.get_token_embeddings(text)
    
    assert len(embeddings) == 2  # Two tokens
    assert all(isinstance(emb, np.ndarray) for emb in embeddings)
    assert all(emb.shape == embeddings[0].shape for emb in embeddings)


def test_get_first_order_concepts(modifier):
    """Test getting first-order concepts."""
    text = "cat"
    embeddings = modifier.get_token_embeddings(text)
    concepts = modifier.get_first_order_concepts(embeddings)
    
    assert len(concepts) == 1  # One token
    assert isinstance(concepts[0], dict)
    assert len(concepts[0]) <= modifier.top_k_concepts


def test_get_second_order_concept(modifier):
    """Test getting second-order concept."""
    text = "the cat sat"
    embeddings = modifier.get_token_embeddings(text)
    first_order = modifier.get_first_order_concepts(embeddings)
    second_order, weights = modifier.get_second_order_concept(first_order)
    
    assert isinstance(second_order, np.ndarray)
    assert len(weights) == len(embeddings)
    assert abs(sum(weights) - 1.0) < 1e-6  # Weights should sum to 1


def test_modify_tokens(modifier):
    """Test token modification."""
    text = "the cat sat"
    embeddings = modifier.get_token_embeddings(text)
    modified = modifier.modify_tokens(embeddings)
    
    assert len(modified) == len(embeddings)
    assert all(isinstance(emb, np.ndarray) for emb in modified)
    assert all(emb.shape == embeddings[0].shape for emb in modified)
    
    # Check that modifications actually changed the embeddings
    for orig, mod in zip(embeddings, modified):
        # Embeddings should be different but not completely dissimilar
        similarity = np.dot(orig, mod) / (np.linalg.norm(orig) * np.linalg.norm(mod))
        assert 0.5 < similarity < 1.0  # Reasonable similarity range


def test_embedding_normalization(modifier):
    """Test that modified embeddings are properly normalized."""
    text = "test"
    embeddings = modifier.get_token_embeddings(text)
    modified = modifier.modify_tokens(embeddings)
    
    for emb in modified:
        # Check if the vector is normalized (L2 norm ≈ 1)
        assert abs(np.linalg.norm(emb) - 1.0) < 1e-6 