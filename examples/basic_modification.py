"""
Example script demonstrating basic token modification.
"""
import numpy as np

from molm.embeddings.nomic_embedder import NomicEmbedder
from molm.concepts.concept_space import ConceptSpace
from molm.modification.token_modifier import TokenModifier


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return np.dot(a, b) / (norm_a * norm_b)


def initialize_concept_space(concept_space: ConceptSpace) -> None:
    """Initialize the concept space with some basic concepts.
    
    Args:
        concept_space: The ConceptSpace instance to initialize.
    """
    # Add some basic concepts related to our example
    initial_concepts = [
        "cat",
        "animal",
        "pet",
        "feline",
        "sit",
        "rest",
        "position",
        "location",
        "furniture",
        "floor",
    ]
    
    print("Initializing concept space...")
    for concept in initial_concepts:
        print(f"  Adding concept: {concept}")
        concept_space.add_concept(concept)
    print("Concept space initialized\n")


def main() -> None:
    # Initialize components
    print("Initializing components...")
    embedder = NomicEmbedder()
    concept_space = ConceptSpace(embedder, max_concepts=100)
    modifier = TokenModifier(embedder, concept_space)

    # Initialize concept space
    initialize_concept_space(concept_space)

    # Example text
    text = "The cat sat on the mat"
    print(f"Original text: {text}\n")

    # Get token embeddings
    print("Getting token embeddings...")
    token_embeddings = modifier.get_token_embeddings(text)
    
    # Get first-order concepts
    print("Finding first-order concepts...")
    first_order = modifier.get_first_order_concepts(token_embeddings)
    
    # Print first-order concepts for each token
    tokens = text.split()
    for token, concepts in zip(tokens, first_order):
        print(f"\nToken: {token}")
        if not concepts:
            print("No relevant concepts found")
            continue
            
        print("Top concepts:")
        for concept, score in concepts.items():
            print(f"  - {concept}: {score:.3f}")

    # Get modified embeddings
    print("\nModifying token embeddings...")
    modified_embeddings = modifier.modify_tokens(token_embeddings)

    # Compare original and modified embeddings
    print("\nEmbedding modifications:")
    for token, orig, mod in zip(tokens, token_embeddings, modified_embeddings):
        try:
            similarity = cosine_similarity(orig, mod)
            print(f"{token}: similarity = {similarity:.3f}")
        except (ValueError, RuntimeWarning) as e:
            print(f"{token}: Error computing similarity - {str(e)}")


if __name__ == "__main__":
    main() 