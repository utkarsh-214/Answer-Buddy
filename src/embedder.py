"""
embedder.py
-----------
Wraps the SentenceTransformer model for generating text embeddings.
Runs entirely locally — no API key required, no network calls after first download.
"""

from typing import List

from sentence_transformers import SentenceTransformer

from src.config import EMBEDDING_MODEL


class Embedder:
    """
    Thin wrapper around SentenceTransformer that provides batched encoding.

    The model is downloaded on first use and cached locally by
    sentence-transformers (usually in ~/.cache/huggingface/).
    Subsequent runs load from cache instantly.
    """

    def __init__(self) -> None:
        print(f"  🔧 Loading embedding model: {EMBEDDING_MODEL} ...")
        self._model = SentenceTransformer(EMBEDDING_MODEL)
        print(f"  ✅ Embedding model loaded.")

    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of text strings.

        Args:
            texts: List of strings to embed.

        Returns:
            List of embedding vectors (each a list of floats).
        """
        if not texts:
            return []

        vectors = self._model.encode(
            texts,
            batch_size=64,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,  # cosine similarity becomes dot product
        )
        return vectors.tolist()

    def embed_single(self, text: str) -> List[float]:
        """Convenience method: embed a single string."""
        return self.embed([text])[0]
