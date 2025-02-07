"""
Module for harvesting and pre-computing concept data.
"""
import json
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import spacy
import torch
from nltk.corpus import wordnet as wn
from tqdm import tqdm

from ..embeddings.nomic_embedder import NomicEmbedder


class ConceptHarvester:
    """Class for harvesting and pre-computing concept data.
    
    This class combines multiple sources (WordNet, ConceptNet, spaCy) to build
    a rich concept space with relationships and metadata. The harvested data
    is saved to disk for efficient loading by StaticConceptSpace.
    """

    def __init__(
        self,
        embedder: NomicEmbedder,
        output_dir: str,
        similarity_threshold: float = 0.5,
    ) -> None:
        """Initialize the ConceptHarvester.

        Args:
            embedder: NomicEmbedder for computing concept embeddings
            output_dir: Directory to save pre-computed data
            similarity_threshold: Minimum similarity for concept relationships
        """
        self.embedder = embedder
        self.output_dir = Path(output_dir)
        self.similarity_threshold = similarity_threshold
        
        # Load spaCy model for abstraction level detection
        print("Loading spaCy model...")
        self.nlp = spacy.load("en_core_web_lg")
        
        # Initialize data structures
        self.concepts: Dict[str, Dict] = {}
        self.domains: Dict[str, Set[str]] = defaultdict(set)
        self.abstraction_levels: Dict[str, Set[str]] = defaultdict(set)
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create checkpoint directory
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)

    def _save_checkpoint(self, name: str, data: dict) -> None:
        """Save a checkpoint.
        
        Args:
            name: Name of the checkpoint
            data: Data to save
        """
        checkpoint_file = self.checkpoint_dir / f"{name}.pkl"
        print(f"\nSaving checkpoint: {checkpoint_file}")
        with open(checkpoint_file, "wb") as f:
            pickle.dump(data, f)

    def _load_checkpoint(self, name: str) -> Optional[dict]:
        """Load a checkpoint if it exists.
        
        Args:
            name: Name of the checkpoint
            
        Returns:
            Checkpoint data if it exists, None otherwise
        """
        checkpoint_file = self.checkpoint_dir / f"{name}.pkl"
        if checkpoint_file.exists():
            print(f"\nLoading checkpoint: {checkpoint_file}")
            with open(checkpoint_file, "rb") as f:
                return pickle.load(f)
        return None

    def harvest_from_wordnet(self) -> None:
        """Harvest concepts and relationships from WordNet."""
        # Check for checkpoint
        checkpoint = self._load_checkpoint("wordnet")
        if checkpoint:
            self.concepts = checkpoint["concepts"]
            self.domains = defaultdict(set, checkpoint["domains"])
            print(f"Restored {len(self.concepts)} concepts from checkpoint")
            return

        print("\nHarvesting from WordNet...")
        for synset in tqdm(list(wn.all_synsets())):
            if len(synset.lemma_names()) < 2:
                continue
                
            concept = synset.name().split(".")[0]
            if concept not in self.concepts:
                self.concepts[concept] = {
                    "definitions": [],
                    "examples": [],
                    "relationships": defaultdict(set)
                }
            
            self.concepts[concept]["definitions"].append(synset.definition())
            self.concepts[concept]["examples"].extend(synset.examples())
            
            domain = synset.lexname().split(".")[0]
            self.domains[domain].add(concept)
            
            for hyper in synset.hypernyms():
                hyper_name = hyper.name().split(".")[0]
                self.concepts[concept]["relationships"]["is_a"].add(hyper_name)
            
            for hypo in synset.hyponyms():
                hypo_name = hypo.name().split(".")[0]
                self.concepts[concept]["relationships"]["includes"].add(hypo_name)
            
            for mero in synset.part_meronyms():
                mero_name = mero.name().split(".")[0]
                self.concepts[concept]["relationships"]["has_part"].add(mero_name)

        # Save checkpoint
        self._save_checkpoint("wordnet", {
            "concepts": self.concepts,
            "domains": dict(self.domains)
        })
        print(f"Harvested {len(self.concepts)} base concepts")

    def detect_abstraction_levels(self) -> None:
        """Detect abstraction levels for concepts using spaCy embeddings."""
        # Check for checkpoint
        checkpoint = self._load_checkpoint("abstraction_levels")
        if checkpoint:
            self.abstraction_levels = defaultdict(set, checkpoint["levels"])
            for concept, info in checkpoint["concept_levels"].items():
                if concept in self.concepts:
                    self.concepts[concept]["abstraction_level"] = info["level"]
            print("Restored abstraction levels from checkpoint")
            return

        print("\nDetecting abstraction levels...")
        levels = {
            "concrete": ["physical", "material", "tangible", "concrete"],
            "functional": ["process", "system", "pattern", "relationship"],
            "abstract": ["abstract", "theoretical", "philosophical", "universal"]
        }
        
        level_docs = {
            level: [self.nlp(kw) for kw in keywords]
            for level, keywords in levels.items()
        }
        
        for concept in tqdm(self.concepts):
            doc = self.nlp(concept)
            max_sim = -1
            best_level = None
            
            for level, kw_docs in level_docs.items():
                sim = max(doc.similarity(kw_doc) for kw_doc in kw_docs)
                if sim > max_sim:
                    max_sim = sim
                    best_level = level
            
            if best_level:
                self.abstraction_levels[best_level].add(concept)
                self.concepts[concept]["abstraction_level"] = best_level

        # Save checkpoint
        concept_levels = {
            c: {"level": info.get("abstraction_level")}
            for c, info in self.concepts.items()
            if "abstraction_level" in info
        }
        self._save_checkpoint("abstraction_levels", {
            "levels": dict(self.abstraction_levels),
            "concept_levels": concept_levels
        })
        print("Detected abstraction levels:", 
              {k: len(v) for k, v in self.abstraction_levels.items()})

    def compute_embeddings(self) -> Dict[str, torch.Tensor]:
        """Compute embeddings for all concepts.

        Returns:
            Dictionary mapping concepts to their embeddings.
        """
        # Check for checkpoint
        checkpoint = self._load_checkpoint("embeddings")
        if checkpoint:
            print(f"Restored {len(checkpoint['embeddings'])} embeddings from checkpoint")
            return checkpoint["embeddings"]

        print("\nComputing concept embeddings...")
        embeddings = {}
        
        try:
            for concept in tqdm(self.concepts):
                emb = self.embedder.get_embedding(concept)
                if emb.size > 0:  # Only store non-empty embeddings
                    embeddings[concept] = torch.from_numpy(emb)
                
                # Save checkpoint every 1000 concepts
                if len(embeddings) % 1000 == 0:
                    self._save_checkpoint("embeddings", {"embeddings": embeddings})
        
        except Exception as e:
            # Save checkpoint on error
            if embeddings:
                self._save_checkpoint("embeddings", {"embeddings": embeddings})
            raise e

        # Save final checkpoint
        self._save_checkpoint("embeddings", {"embeddings": embeddings})
        print(f"Computed {len(embeddings)} concept embeddings")
        return embeddings

    def compute_relationships(
        self, embeddings: Dict[str, torch.Tensor]
    ) -> Dict[str, Dict[str, float]]:
        """Compute relationships between concepts based on embedding similarity.

        Args:
            embeddings: Dictionary of concept embeddings.

        Returns:
            Dictionary mapping concepts to their related concepts and scores.
        """
        # Check for checkpoint
        checkpoint = self._load_checkpoint("relationships")
        if checkpoint:
            print(f"Restored relationships from checkpoint")
            return checkpoint["relationships"]

        print("\nComputing concept relationships...")
        relationships = defaultdict(dict)
        
        # Convert to unit vectors and filter invalid ones
        print("Normalizing embeddings...")
        valid_concepts = []
        valid_embeddings = []
        
        for concept, emb in tqdm(embeddings.items()):
            norm = torch.norm(emb)
            if norm > 0:
                valid_concepts.append(concept)
                valid_embeddings.append(emb / norm)
        
        valid_embeddings = torch.stack(valid_embeddings)
        print(f"Computing similarities for {len(valid_concepts)} valid concepts")
        
        # Process in batches to reduce memory usage
        batch_size = 1000
        try:
            for i in tqdm(range(0, len(valid_concepts), batch_size)):
                batch_concepts = valid_concepts[i:i + batch_size]
                batch_embeddings = valid_embeddings[i:i + batch_size]
                
                # Compute similarities with all concepts after this batch
                for j, c1 in enumerate(batch_concepts):
                    e1 = batch_embeddings[j]
                    
                    # Compute similarities with remaining concepts
                    start_idx = i + j + 1
                    if start_idx < len(valid_concepts):
                        sims = torch.matmul(
                            valid_embeddings[start_idx:],
                            e1
                        )
                        
                        # Get indices where similarity exceeds threshold
                        matches = torch.nonzero(sims >= self.similarity_threshold).squeeze()
                        if matches.dim() == 0 and matches.numel() > 0:
                            # Handle single match case
                            idx = start_idx + matches.item()
                            c2 = valid_concepts[idx]
                            sim = float(sims[matches])
                            relationships[c1][c2] = sim
                            relationships[c2][c1] = sim
                        else:
                            for match_idx in matches:
                                idx = start_idx + match_idx.item()
                                c2 = valid_concepts[idx]
                                sim = float(sims[match_idx])
                                relationships[c1][c2] = sim
                                relationships[c2][c1] = sim
                
                # Save checkpoint after each batch
                self._save_checkpoint("relationships", {
                    "relationships": dict(relationships)
                })
        
        except Exception as e:
            # Save checkpoint on error
            if relationships:
                self._save_checkpoint("relationships", {
                    "relationships": dict(relationships)
                })
            raise e

        # Save final checkpoint
        self._save_checkpoint("relationships", {"relationships": dict(relationships)})
        return relationships

    def save_data(
        self,
        embeddings: Dict[str, torch.Tensor],
        relationships: Dict[str, Dict[str, float]],
    ) -> None:
        """Save pre-computed data to disk.

        Args:
            embeddings: Dictionary of concept embeddings
            relationships: Dictionary of concept relationships
        """
        print("\nSaving pre-computed data...")
        
        # Save embeddings
        torch.save(embeddings, self.output_dir / "embeddings.pt")
        
        # Save relationships
        torch.save(relationships, self.output_dir / "relationships.pt")
        
        # Save metadata
        metadata = {
            "domains": {k: list(v) for k, v in self.domains.items()},
            "abstraction_levels": {k: list(v) for k, v in self.abstraction_levels.items()},
            "concepts": self.concepts
        }
        torch.save(metadata, self.output_dir / "metadata.pt")
        
        # Save human-readable metadata
        with open(self.output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    def run_harvest(self) -> None:
        """Run the complete concept harvesting process."""
        # Step 1: Harvest base concepts and relationships
        self.harvest_from_wordnet()
        
        # Step 2: Detect abstraction levels
        self.detect_abstraction_levels()
        
        # Step 3: Compute embeddings
        embeddings = self.compute_embeddings()
        
        # Step 4: Compute relationships
        relationships = self.compute_relationships(embeddings)
        total_rels = sum(len(rels) for rels in relationships.values())
        print(f"Found {total_rels} concept relationships")
        
        # Step 5: Save data
        self.save_data(embeddings, relationships)
        print(f"\nSaved pre-computed data to {self.output_dir}")


def main() -> None:
    """Run the concept harvesting process."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Harvest and pre-compute concept data")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/concepts",
        help="Directory to save pre-computed data",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.5,
        help="Minimum similarity for concept relationships",
    )
    
    args = parser.parse_args()
    
    # Initialize components
    print("Initializing concept harvester...")
    embedder = NomicEmbedder()
    harvester = ConceptHarvester(
        embedder,
        args.output_dir,
        args.similarity_threshold,
    )
    
    # Run harvest
    harvester.run_harvest()


if __name__ == "__main__":
    main() 