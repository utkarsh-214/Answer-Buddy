"""
vector_store.py
---------------
Manages all interactions with the Qdrant vector database:
  - Collection creation (idempotent)
  - Upserting chunks with their embeddings and metadata
  - Semantic search by query embedding
"""

import uuid
from typing import List

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from src.config import (
    QDRANT_URL,
    QDRANT_API_KEY,
    QDRANT_COLLECTION_NAME,
    EMBEDDING_DIMENSION,
    TOP_K,
)
from src.pdf_parser import Chunk


class VectorStore:
    """Qdrant-backed vector store for document chunks."""

    def __init__(self) -> None:
        if not QDRANT_URL:
            raise EnvironmentError(
                "QDRANT_URL is not set. Please add it to your .env file."
            )
        if not QDRANT_API_KEY:
            raise EnvironmentError(
                "QDRANT_API_KEY is not set. Please add it to your .env file."
            )

        self._client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
            timeout=30,
        )
        self._collection = QDRANT_COLLECTION_NAME

    # ── Collection management ────────────────────────────────────────────────

    def create_collection(self, recreate: bool = False) -> None:
        """
        Creates the Qdrant collection if it doesn't already exist.

        Args:
            recreate: If True, deletes and recreates the collection.
                      Useful for a fresh re-ingest.
        """
        existing = [c.name for c in self._client.get_collections().collections]

        if recreate and self._collection in existing:
            print(f"  🗑️  Deleting existing collection '{self._collection}' ...")
            self._client.delete_collection(self._collection)
            existing = []

        if self._collection not in existing:
            print(f"  📦 Creating collection '{self._collection}' ...")
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=qmodels.VectorParams(
                    size=EMBEDDING_DIMENSION,
                    distance=qmodels.Distance.COSINE,
                ),
            )
            print(f"  ✅ Collection created.")
        else:
            print(f"  ✅ Collection '{self._collection}' already exists.")

    # ── Ingestion ────────────────────────────────────────────────────────────

    def upsert_chunks(
        self,
        chunks: List[Chunk],
        embeddings: List[List[float]],
        batch_size: int = 100,
    ) -> None:
        """
        Stores chunks and their embeddings in Qdrant.
        Each point's payload stores: text, doc_name, page_number.

        Args:
            chunks:     List of Chunk objects (text + metadata).
            embeddings: Corresponding embedding vectors.
            batch_size: Number of points to upsert per API call.
        """
        assert len(chunks) == len(embeddings), (
            "chunks and embeddings must have the same length"
        )

        points: List[qmodels.PointStruct] = []

        for chunk, vector in zip(chunks, embeddings):
            points.append(
                qmodels.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload={
                        "text": chunk.text,
                        "doc_name": chunk.doc_name,
                        "page_number": chunk.page_number,
                    },
                )
            )

        # Upload in batches to avoid payload size limits
        total = len(points)
        for i in range(0, total, batch_size):
            batch = points[i : i + batch_size]
            self._client.upsert(
                collection_name=self._collection,
                points=batch,
            )
            uploaded = min(i + batch_size, total)
            print(f"  ⬆️  Uploaded {uploaded}/{total} chunks ...", end="\r")

        print(f"\n  ✅ All {total} chunks stored in Qdrant.")

    # ── Retrieval ────────────────────────────────────────────────────────────

    def search(
        self,
        query_embedding: List[float],
        top_k: int = TOP_K,
    ) -> List[dict]:
        """
        Finds the most semantically similar chunks for a query embedding.

        Args:
            query_embedding: The embedded query vector.
            top_k:           Number of results to return.

        Returns:
            List of dicts with keys: text, doc_name, page_number, score.
        """
        results = self._client.search(
            collection_name=self._collection,
            query_vector=query_embedding,
            limit=top_k,
            with_payload=True,
        )

        return [
            {
                "text": hit.payload["text"],
                "doc_name": hit.payload["doc_name"],
                "page_number": hit.payload["page_number"],
                "score": round(hit.score, 4),
            }
            for hit in results
        ]
