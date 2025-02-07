#!/usr/bin/env python3
"""
Script to run the concept harvesting process.
"""
import os
import ssl
import subprocess
import sys
from pathlib import Path

import nltk
import spacy

from molm.concepts.harvester import ConceptHarvester
from molm.embeddings.nomic_embedder import NomicEmbedder


def download_nltk_data() -> None:
    """Download required NLTK data, handling SSL issues."""
    print("Downloading NLTK data...")
    
    try:
        # Try to create an unverified SSL context
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        # Handle systems that don't have SSL certificates properly installed
        ssl._create_default_https_context = _create_unverified_https_context

    # Set NLTK data path to user's home directory
    nltk_data_dir = os.path.expanduser("~/nltk_data")
    os.makedirs(nltk_data_dir, exist_ok=True)
    nltk.data.path.append(nltk_data_dir)

    # Download required data
    for resource in ["wordnet", "omw-1.4"]:
        try:
            nltk.download(resource, quiet=True, raise_on_error=True)
            print(f"  ✓ Downloaded {resource}")
        except Exception as e:
            print(f"  ✗ Error downloading {resource}: {str(e)}")
            sys.exit(1)


def check_dependencies() -> None:
    """Check and install required dependencies."""
    print("Checking dependencies...")
    
    # Check spaCy model
    print("\nChecking spaCy model...")
    try:
        spacy.load("en_core_web_lg")
        print("  ✓ spaCy model found")
    except OSError:
        print("  ✗ spaCy model not found, installing...")
        try:
            subprocess.run(
                [sys.executable, "-m", "spacy", "download", "en_core_web_lg"],
                check=True,
                capture_output=True,
            )
            print("  ✓ spaCy model installed")
        except subprocess.CalledProcessError as e:
            print(f"  ✗ Error installing spaCy model: {e.stderr.decode()}")
            sys.exit(1)
    
    # Check NLTK data
    print("\nChecking NLTK data...")
    try:
        from nltk.corpus import wordnet
        wordnet.all_synsets()
        print("  ✓ NLTK data found")
    except LookupError:
        download_nltk_data()


def main() -> None:
    """Run the concept harvesting process."""
    # Check dependencies
    check_dependencies()
    
    # Set up output directory
    output_dir = Path(__file__).parent.parent / "data" / "concepts"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize components
    print("\nInitializing components...")
    embedder = NomicEmbedder()
    harvester = ConceptHarvester(
        embedder=embedder,
        output_dir=str(output_dir),
        similarity_threshold=0.5,
    )
    
    # Run harvest
    print("\nStarting concept harvest...")
    harvester.run_harvest()


if __name__ == "__main__":
    main() 