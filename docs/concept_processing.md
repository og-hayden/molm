# Concept Processing and Token Modification Flow

This document explains the detailed flow of concept processing and token modification in the MOLM system.

## Overview

The system modifies token embeddings using a two-stage concept-aware process:
1. First-order concept gathering for individual tokens
2. Second-order concept formation across the sequence
3. Combined modification using both concept levels

## First-Order Concept Gathering

For each individual token, we:
1. Project the token embedding into concept space
2. Find top-k most relevant concepts using cosine similarity
3. Each token gets its own set of first-order concepts with relevance scores

### Example

For the phrase "Time flows like a river":
```python
token_concepts = {
    "Time": [
        {"concept": "Temporality", "score": 0.8},
        {"concept": "Continuity", "score": 0.6}
    ],
    "flows": [
        {"concept": "Movement", "score": 0.9},
        {"concept": "Fluidity", "score": 0.7}
    ],
    "river": [
        {"concept": "Fluidity", "score": 0.9},
        {"concept": "Nature", "score": 0.8}
    ]
}
```

## Second-Order Concept Formation

The system then:
1. Examines ALL first-order concepts that appeared across ALL tokens
2. Aggregates strongly related concepts via vector addition
3. Weights the addition by the original relevance scores
4. Produces a single higher-order concept vector representing the sequence's "semantic thread"

### Example

```python
# High overlap between Fluidity, Movement, and Continuity concepts
# Creates a higher-order concept like "Continuous Flow"
second_order = (fluidity_vector * 0.8 + 
                movement_vector * 0.85 + 
                continuity_vector * 0.7)
```

## Token Modification Process

Each token undergoes modification in two ways:

### 1. First-Order Modification
- Token is modified only by its own first-order concepts
- Modifications are weighted by concept relevance scores

```python
# For "flows" token:
flows_modified = flows + (movement_vector * 0.9 + fluidity_vector * 0.7)
```

### 2. Second-Order Modification
- ALL tokens are modified by the second-order concept
- Modification is weighted by each token's contribution to the second-order concept

```python
participation_weight = cosine_similarity(token_concepts, second_order)
final_token = token_modified + (second_order * participation_weight)
```

## Example Effects

Using our example "Time flows like a river":

- **"flows"**:
  - Heavy first-order modification from "Movement" and "Fluidity" concepts
  - Strong second-order influence due to high concept overlap
  - Result: Embedding strengthened in the "continuous motion" direction

- **"like"**:
  - Minimal first-order modifications (function word)
  - Moderate second-order influence to maintain sequence coherence
  - Result: Slight shift toward the sequence's fluid/temporal meaning

- **"river"**:
  - Strong first-order modification from "Fluidity" and "Nature"
  - Strong second-order influence due to fluidity concept overlap
  - Result: Embedding enhanced in both physical and metaphorical fluidity aspects

## Implementation Details

The process is implemented in three main methods in the `TokenModifier` class:

1. `get_first_order_concepts()`: Handles individual token concept detection
2. `get_second_order_concept()`: Aggregates concepts into higher-order representation
3. `modify_tokens()`: Applies both modification types with proper weighting

See the code documentation for specific implementation details. 