# MOLM: Modular Token Modification in Language Models

This project implements a focused investigation of concept-aware token modification in transformer architectures. Instead of training a complete language model, it isolates and tests the core mechanism of using concept vectors to modify token representations.

## Core Components

1. **Token Embeddings** (`src/molm/embeddings/`)
   - Uses the Nomic-Embed-Text model for generating token embeddings
   - Provides a clean interface for embedding individual tokens and sequences
   - Integrates with Ollama for efficient local embedding generation

2. **Concept Space** (`src/molm/concepts/`)
   - Two implementations for concept management:
     - `ConceptSpace`: Dynamic concept space with real-time relationship computation
     - `StaticConceptSpace`: Efficient pre-computed concept vectors and relationships
   - Integrates with WordNet for semantic relationships
   - Uses spaCy for abstraction level detection
   - Supports concept harvesting and pre-computation for better performance

3. **Token Modification** (`src/molm/modification/`)
   - Implements two-stage concept-aware token modification:
     1. First-order concept gathering for individual tokens
     2. Second-order concept formation across sequences
   - Balances token-specific meaning with sequence-level coherence
   - Configurable weights for concept influence

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/molm.git
cd molm
```

2. Install dependencies:
```bash
pip install -e .
```

3. Install and start Ollama:
   - Follow instructions at [Ollama's website](https://ollama.ai)
   - Pull the Nomic-Embed-Text model:
```bash
ollama pull nomic-embed-text
```

4. Download required NLTK and spaCy data:
```bash
python -m nltk.downloader wordnet omw-1.4
python -m spacy download en_core_web_lg
```

## Usage

### Quick Start
```python
from molm.embeddings.nomic_embedder import NomicEmbedder
from molm.concepts.static_concept_space import StaticConceptSpace
from molm.modification.token_modifier import TokenModifier

# Initialize components
embedder = NomicEmbedder()
concept_space = StaticConceptSpace(embedder, "data/concepts")
modifier = TokenModifier(embedder, concept_space)

# Get and modify token embeddings
text = "The neural network processes data efficiently"
token_embeddings = modifier.get_token_embeddings(text)
modified_embeddings = modifier.modify_tokens(token_embeddings)
```

### Concept Harvesting
To build the pre-computed concept space:
```bash
python scripts/harvest_concepts.py
```
This will:
1. Harvest concepts and relationships from WordNet
2. Detect abstraction levels using spaCy
3. Compute concept embeddings using Nomic-Embed-Text
4. Build and save the concept relationship graph
5. Save all data for efficient loading by StaticConceptSpace

### Examples
Run the demonstration scripts:
```bash
# Basic token modification example
python examples/basic_modification.py

# Concept space exploration and analysis
python examples/concept_space_demo.py

# Detailed concept demonstration
python examples/concept_demonstration.py
```

## Project Structure

```
molm/
├── src/
│   └── molm/
│       ├── embeddings/
│       │   └── nomic_embedder.py
│       ├── concepts/
│       │   ├── concept_space.py
│       │   ├── static_concept_space.py
│       │   └── harvester.py
│       └── modification/
│           └── token_modifier.py
├── examples/
│   ├── basic_modification.py
│   ├── concept_space_demo.py
│   └── concept_demonstration.py
├── scripts/
│   ├── harvest_concepts.py
│   └── verify_embeddings.py
├── tests/
├── docs/
└── README.md
```

## Features

- **Modular Design**: Each component is independent and easily modifiable
- **Type Safety**: Comprehensive type hints throughout the codebase
- **Efficient Concept Management**:
  - Pre-computed concept vectors and relationships
  - Checkpointing for long-running operations
  - Efficient batch processing for large concept spaces
- **Rich Concept Analysis**:
  - Domain and abstraction level detection
  - First and second-order concept relationships
  - Cross-domain concept mapping
- **Configurable**: Easy to adjust weights, thresholds, and parameters

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{molm2024,
  title = {MOLM: Modular Token Modification in Language Models},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/molm}
}
``` 