"""
Module for handling token embeddings using the Nomic API.
"""
from typing import List, Optional

import numpy as np
import requests
from numpy.typing import NDArray


class NomicEmbedder:
    """A class to handle token embeddings using the Nomic API."""

    def __init__(self, api_url: str = "http://localhost:11434") -> None:
        """Initialize the NomicEmbedder.

        Args:
            api_url: The URL of the Ollama API endpoint.
        """
        self.api_url = api_url.rstrip("/")
        self._validate_connection()

    def _validate_connection(self) -> None:
        """Validate the connection to the Ollama API.

        Raises:
            ConnectionError: If the API is not accessible.
        """
        try:
            response = requests.get(f"{self.api_url}/api/tags")
            if response.status_code != 200:
                raise ConnectionError(f"API returned status code {response.status_code}")
        except requests.RequestException as e:
            raise ConnectionError(f"Failed to connect to Ollama API: {e}")

    def get_embeddings(
        self, texts: List[str], batch_size: Optional[int] = None
    ) -> NDArray[np.float32]:
        """Get embeddings for a list of texts.

        Args:
            texts: List of texts to embed.
            batch_size: Optional batch size for processing multiple texts.

        Returns:
            Array of embeddings with shape (n_texts, embedding_dim).

        Raises:
            RuntimeError: If the API call fails.
        """
        if batch_size is None:
            batch_size = len(texts)

        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            batch_embeddings = []
            
            for text in batch:
                response = requests.post(
                    f"{self.api_url}/api/embeddings",
                    json={"model": "nomic-embed-text", "prompt": text},
                )
                
                if response.status_code != 200:
                    raise RuntimeError(
                        f"Failed to get embeddings: {response.status_code}"
                    )
                
                embedding = response.json()["embedding"]
                batch_embeddings.append(embedding)
            
            embeddings.extend(batch_embeddings)

        return np.array(embeddings, dtype=np.float32)

    def get_embedding(self, text: str) -> NDArray[np.float32]:
        """Get embedding for a single text.

        Args:
            text: Text to embed.

        Returns:
            Embedding vector.
        """
        return self.get_embeddings([text])[0] 