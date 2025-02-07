"""
Demonstration of using pre-computed concepts with token modification.
"""
import numpy as np
from tabulate import tabulate
from pathlib import Path

from molm.embeddings.nomic_embedder import NomicEmbedder
from molm.concepts.static_concept_space import StaticConceptSpace
from molm.modification.token_modifier import TokenModifier


def print_section(title: str) -> None:
    """Print a section title."""
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print('=' * 80)


def explore_concept_space(concept_space: StaticConceptSpace) -> None:
    """Explore the pre-computed concept space.
    
    Args:
        concept_space: The StaticConceptSpace instance
    """
    print_section("Exploring Pre-computed Concept Space")
    
    # Example concepts to explore
    example_concepts = [
        "river",  # Concrete nature concept
        "algorithm",  # Technical concept
        "freedom",  # Abstract concept
        "network",  # Multiple domain concept
    ]
    
    for concept in example_concepts:
        print(f"\nExploring concept: {concept}")
        
        # Get concept info
        domains = [
            domain for domain, concepts in concept_space.domains.items()
            if concept in concepts
        ]
        levels = [
            level for level, concepts in concept_space.abstraction_levels.items()
            if concept in concepts
        ]
        
        print(f"Domain(s): {', '.join(domains) if domains else 'Unknown'}")
        print(f"Abstraction Level(s): {', '.join(levels) if levels else 'Unknown'}")
        
        # Get related concepts
        related = concept_space.get_related_concepts(concept, max_depth=2)
        if related:
            print("\nTop related concepts:")
            top_related = sorted(related.items(), key=lambda x: x[1], reverse=True)[:5]
            for rel_concept, score in top_related:
                # Get domain and level for related concept
                rel_domains = [d for d, c in concept_space.domains.items() if rel_concept in c]
                rel_levels = [l for l, c in concept_space.abstraction_levels.items() if rel_concept in c]
                print(f"  - {rel_concept}: {score:.3f}")
                if rel_domains or rel_levels:
                    print(f"    ({', '.join(rel_domains) if rel_domains else 'Unknown'} | "
                          f"{', '.join(rel_levels) if rel_levels else 'Unknown'})")
        else:
            print("\nNo related concepts found")


def analyze_text_with_concepts(
    modifier: TokenModifier,
    text: str,
    description: str
) -> None:
    """Analyze text using concept-aware token modification.
    
    Args:
        modifier: TokenModifier instance
        text: Text to analyze
        description: Description of the example
    """
    print_section(f"Analyzing: {text}")
    print(f"Description: {description}\n")

    # Get token embeddings
    tokens = text.split()
    token_embeddings = modifier.get_token_embeddings(text)
    
    # Get first-order concepts
    print("1. First-Order Concept Analysis:")
    first_order = modifier.get_first_order_concepts(token_embeddings)
    for token, concepts in zip(tokens, first_order):
        print(f"\n{token}:")
        if not concepts:
            print("  No relevant concepts found")
            continue
        for concept, score in sorted(concepts.items(), key=lambda x: x[1], reverse=True)[:3]:
            # Get domain and level for concept
            domains = [d for d, c in modifier.concept_space.domains.items() if concept in c]
            levels = [l for l, c in modifier.concept_space.abstraction_levels.items() if concept in c]
            print(f"  - {concept}: {score:.3f}")
            if domains or levels:
                print(f"    ({', '.join(domains) if domains else 'Unknown'} | "
                      f"{', '.join(levels) if levels else 'Unknown'})")
    
    # Get second-order concept
    print("\n2. Second-Order Concept Analysis:")
    second_order, participation = modifier.get_second_order_concept(first_order)
    
    # Find most similar concepts to second-order concept
    print("Emergent sequence-level concepts:")
    similarities = []
    for concept, emb in modifier.concept_space.concept_embeddings.items():
        sim = np.dot(second_order, emb) / (np.linalg.norm(second_order) * np.linalg.norm(emb))
        similarities.append((concept, sim))
    
    top_concepts = sorted(similarities, key=lambda x: x[1], reverse=True)[:5]
    for concept, sim in top_concepts:
        domains = [d for d, c in modifier.concept_space.domains.items() if concept in c]
        levels = [l for l, c in modifier.concept_space.abstraction_levels.items() if concept in c]
        print(f"  - {concept}: {sim:.3f}")
        if domains or levels:
            print(f"    ({', '.join(domains) if domains else 'Unknown'} | "
                  f"{', '.join(levels) if levels else 'Unknown'})")
    
    print("\nToken participation in sequence-level concept:")
    for token, weight in zip(tokens, participation):
        print(f"  - {token}: {weight:.3f}")
    
    # Apply modifications
    print("\n3. Token Modifications:")
    modified = modifier.modify_tokens(token_embeddings)
    
    # Compare original and modified embeddings
    print("\nSimilarity between original and modified embeddings:")
    similarities = []
    for token, orig, mod in zip(tokens, token_embeddings, modified):
        sim = np.dot(orig, mod) / (np.linalg.norm(orig) * np.linalg.norm(mod))
        similarities.append([token, sim])
    
    print(tabulate(
        similarities,
        headers=["Token", "Similarity"],
        floatfmt=".3f"
    ))


def main() -> None:
    """Run the concept space demonstration."""
    # Initialize components
    print("Initializing components...")
    embedder = NomicEmbedder()
    
    # Load pre-computed concept space
    concepts_dir = Path("data/concepts")
    if not concepts_dir.exists():
        raise FileNotFoundError(
            "Pre-computed concept data not found. "
            "Please run scripts/harvest_concepts.py first."
        )
    
    concept_space = StaticConceptSpace(embedder, str(concepts_dir))
    modifier = TokenModifier(
        embedder,
        concept_space,
        first_order_weight=0.3,
        second_order_weight=0.2,
    )
    
    # Explore concept space
    explore_concept_space(concept_space)
    
    # Analyze example texts
    examples = [
        (
            "The neural network processes data efficiently",
            "Technical domain with concept overlap between 'neural', 'network', and 'processes'"
        ),
        (
            "Freedom flows like a river",
            "Mixed domains: abstract concept with concrete nature metaphor"
        ),
        (
            "The algorithm learns from patterns",
            "Technical domain with learning/cognition concepts"
        )
    ]
    
    for text, description in examples:
        analyze_text_with_concepts(modifier, text, description)
        print("\n")


if __name__ == "__main__":
    main() 