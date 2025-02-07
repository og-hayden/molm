"""
Script to verify the Nomic embeddings model availability and functionality.
"""
import sys
import time
from typing import Optional

import numpy as np
import requests


def check_ollama_server(api_url: str = "http://localhost:11434") -> bool:
    """Check if Ollama server is running.

    Args:
        api_url: The URL of the Ollama API endpoint.

    Returns:
        True if server is accessible, False otherwise.
    """
    try:
        response = requests.get(f"{api_url}/api/tags")
        return response.status_code == 200
    except requests.RequestException:
        return False


def check_model_availability(
    api_url: str = "http://localhost:11434", model: str = "nomic-embed-text"
) -> Optional[str]:
    """Check if the specified model is available.

    Args:
        api_url: The URL of the Ollama API endpoint.
        model: The name of the model to check.

    Returns:
        Model version if available, None otherwise.
    """
    try:
        # Try to generate an embedding - if it works, the model is available
        response = requests.post(
            f"{api_url}/api/embeddings",
            json={"model": model, "prompt": "test"},
            timeout=5,
        )
        if response.status_code == 200:
            return "available"  # Actual version info not easily accessible
        return None
    except requests.RequestException:
        return None


def test_embedding_generation(
    api_url: str = "http://localhost:11434", model: str = "nomic-embed-text"
) -> tuple[bool, Optional[int], float]:
    """Test embedding generation with a simple example.

    Args:
        api_url: The URL of the Ollama API endpoint.
        model: The name of the model to use.

    Returns:
        Tuple of (success, embedding_dimension, response_time).
    """
    test_text = "This is a test sentence."
    start_time = time.time()
    
    try:
        response = requests.post(
            f"{api_url}/api/embeddings",
            json={"model": model, "prompt": test_text},
        )
        
        if response.status_code != 200:
            return False, None, time.time() - start_time
            
        embedding = response.json().get("embedding")
        if not embedding or not isinstance(embedding, list):
            return False, None, time.time() - start_time
            
        return True, len(embedding), time.time() - start_time
    except requests.RequestException:
        return False, None, time.time() - start_time


def main() -> None:
    """Run the verification checks."""
    api_url = "http://localhost:11434"
    model = "nomic-embed-text"
    
    print("\n=== Nomic Embeddings Model Verification ===\n")
    
    # Check 1: Ollama Server
    print("1. Checking Ollama server...")
    if check_ollama_server(api_url):
        print("✓ Ollama server is running")
    else:
        print("✗ Ollama server is not accessible")
        print(f"  Make sure Ollama is running and accessible at {api_url}")
        sys.exit(1)
    
    # Check 2: Model Availability
    print("\n2. Checking model availability...")
    model_version = check_model_availability(api_url, model)
    if model_version:
        print(f"✓ Model '{model}' is available")
    else:
        print(f"✗ Model '{model}' is not available")
        print(f"  Try running: ollama pull {model}")
        sys.exit(1)
    
    # Check 3: Embedding Generation
    print("\n3. Testing embedding generation...")
    success, dim, response_time = test_embedding_generation(api_url, model)
    if success:
        print("✓ Successfully generated embeddings")
        print(f"  - Embedding dimension: {dim}")
        print(f"  - Response time: {response_time:.2f}s")
    else:
        print("✗ Failed to generate embeddings")
        print("  Check the Ollama server logs for more details")
        sys.exit(1)
    
    print("\n=== All checks passed successfully! ===")


if __name__ == "__main__":
    main() 