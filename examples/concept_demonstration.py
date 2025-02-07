"""
Demonstration of concept-aware token modification with detailed analysis.
"""
import numpy as np
from tabulate import tabulate

from molm.embeddings.nomic_embedder import NomicEmbedder
from molm.concepts.concept_space import ConceptSpace
from molm.modification.token_modifier import TokenModifier


def print_section(title: str) -> None:
    """Print a section title."""
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print('=' * 80)


def print_concepts(token: str, concepts: dict) -> None:
    """Print concepts and their scores in a formatted way."""
    if not concepts:
        print(f"  {token}: No relevant concepts found")
        return
    
    print(f"  {token}:")
    for concept, score in concepts.items():
        print(f"    - {concept}: {score:.3f}")


def analyze_example(
    modifier: TokenModifier, text: str, description: str
) -> None:
    """Analyze a text example with detailed output.
    
    Args:
        modifier: TokenModifier instance
        text: Text to analyze
        description: Description of the example
    """
    print_section(f"Example: {text}")
    print(f"Description: {description}\n")

    # Get token embeddings
    print("1. Getting token embeddings...")
    token_embeddings = modifier.get_token_embeddings(text)
    tokens = text.split()
    print(f"   Found {len(tokens)} tokens\n")
    
    # Get and display first-order concepts
    print("2. First-Order Concept Detection:")
    first_order = modifier.get_first_order_concepts(token_embeddings)
    for token, concepts in zip(tokens, first_order):
        print_concepts(token, concepts)
    print()
    
    # Get and analyze second-order concepts
    print("3. Second-Order Concept Formation:")
    second_order, participation = modifier.get_second_order_concept(first_order)
    
    # Show participation weights
    print("   Token Participation in Second-Order Concept:")
    for token, weight in zip(tokens, participation):
        print(f"    - {token}: {weight:.3f}")
    print()
    
    # Get modified embeddings and show changes
    print("4. Token Modifications:")
    modified = modifier.modify_tokens(token_embeddings)
    
    # Calculate similarities
    similarities = []
    for token, orig, mod in zip(tokens, token_embeddings, modified):
        sim = np.dot(orig, mod) / (np.linalg.norm(orig) * np.linalg.norm(mod))
        similarities.append([token, sim])
    
    print(tabulate(
        similarities,
        headers=["Token", "Similarity to Original"],
        floatfmt=".3f"
    ))
    print()


def main() -> None:
    """Run the concept modification demonstration."""
    # Initialize components
    print("Initializing components...")
    embedder = NomicEmbedder()
    concept_space = ConceptSpace(embedder, max_concepts=200)
    modifier = TokenModifier(
        embedder,
        concept_space,
        first_order_weight=0.3,
        second_order_weight=0.2,
    )

    # Initialize concept space with seed concepts
    print("Initializing concept space...")
    seed_concepts = [
        # Time and movement concepts
        "time", "flow", "movement", "continuous", "temporal",
        # Nature concepts
        "river", "water", "nature", "fluid", "stream",
        # Abstract concepts
        "metaphor", "abstract", "concrete", "physical", "symbolic",
        # Emotional concepts
        "joy", "sadness", "anger", "peace", "serenity",
        # Technical concepts
        "computer", "algorithm", "data", "process", "system"
    ]
    
    for concept in seed_concepts:
        concept_space.add_concept(concept)
    print(f"Added {len(seed_concepts)} seed concepts\n")

    # Example 1: Metaphorical expression
    analyze_example(
        modifier,
        "Time flows like a river",
        "A metaphorical expression combining temporal and fluid concepts"
    )

    # Example 2: Technical description
    analyze_example(
        modifier,
        "The algorithm processes data efficiently",
        "A technical description with computational concepts"
    )

    # Example 3: Emotional expression
    analyze_example(
        modifier,
        "Peace flows through gentle streams",
        "An emotional expression with nature imagery"
    )


if __name__ == "__main__":
    main() 