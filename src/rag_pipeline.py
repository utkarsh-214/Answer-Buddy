"""
rag_pipeline.py
---------------
Orchestrates the full RAG workflow:
  - Ingest: PDF parsing → embedding → Qdrant storage
  - Query:  embedding → Qdrant retrieval → LLM generation → structured answer
"""

from dataclasses import dataclass, field
from typing import List

from tqdm import tqdm

from src.config import PDF_DIR, TOP_K, RELEVANCE_THRESHOLD
from src.pdf_parser import parse_all_pdfs, Chunk
from src.embedder import Embedder
from src.vector_store import VectorStore
from src.llm_client import LLMClient


# ── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class Citation:
    """Represents a single source citation for an answer."""
    doc_name: str
    page_number: int
    snippet: str
    score: float


@dataclass
class Answer:
    """Structured response returned by the RAG pipeline."""
    question: str
    answer_text: str
    citations: List[Citation] = field(default_factory=list)
    found_in_docs: bool = True


# ── Pipeline ──────────────────────────────────────────────────────────────────

class RAGPipeline:
    """
    End-to-end RAG pipeline: ingest PDFs or answer questions.
    
    Components are lazy-initialised so startup is fast when 
    only one mode (ingest or query) is needed.
    """

    def __init__(self) -> None:
        self._embedder: Embedder | None = None
        self._vector_store: VectorStore | None = None
        self._llm: LLMClient | None = None

    # ── Lazy property accessors ──────────────────────────────────────────────

    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = Embedder()
        return self._embedder

    @property
    def vector_store(self) -> VectorStore:
        if self._vector_store is None:
            self._vector_store = VectorStore()
        return self._vector_store

    @property
    def llm(self) -> LLMClient:
        if self._llm is None:
            self._llm = LLMClient()
        return self._llm

    # ── Ingest ────────────────────────────────────────────────────────────────

    def ingest(self, pdf_dir: str = PDF_DIR, recreate: bool = False) -> None:
        """
        Full ingestion pipeline:
          1. Parse all PDFs in pdf_dir into chunks.
          2. Generate embeddings for every chunk.
          3. Store chunks + embeddings in Qdrant.

        Args:
            pdf_dir:  Directory containing PDF files.
            recreate: If True, wipe and recreate the Qdrant collection first.
        """
        print("\n🚀 Starting ingestion pipeline ...\n")

        # Step 1: Parse PDFs
        print("Step 1/3 — Parsing PDFs")
        chunks: List[Chunk] = parse_all_pdfs(pdf_dir)
        print(f"\n  Total chunks: {len(chunks)}\n")

        # Step 2: Generate embeddings (with progress bar)
        print("Step 2/3 — Generating embeddings")
        _ = self.embedder  # triggers model load print
        texts = [c.text for c in chunks]
        
        # Batch embed with tqdm progress bar
        batch_size = 64
        all_embeddings: list = []
        for i in tqdm(range(0, len(texts), batch_size), desc="  Embedding", unit="batch"):
            batch = texts[i : i + batch_size]
            all_embeddings.extend(self.embedder.embed(batch))
        print(f"  ✅ {len(all_embeddings)} embeddings generated.\n")

        # Step 3: Store in Qdrant
        print("Step 3/3 — Storing in Qdrant")
        self.vector_store.create_collection(recreate=recreate)
        self.vector_store.upsert_chunks(chunks, all_embeddings)

        print("\n✅ Ingestion complete!\n")

    # ── Query ─────────────────────────────────────────────────────────────────

    def query(self, question: str) -> Answer:
        """
        Answers a user question using retrieved context from Qdrant + the LLM.

        Workflow:
          1. Embed the question.
          2. Search Qdrant for top-k similar chunks.
          3. Apply relevance threshold — if no chunk is relevant enough,
             return a "not found" answer without calling the LLM.
          4. Call the LLM with the question + retrieved context.
          5. Return a structured Answer with citations.

        Args:
            question: The user's natural-language question.

        Returns:
            Answer dataclass with answer_text, citations, and found_in_docs flag.
        """
        # Step 1: Embed query
        query_vector = self.embedder.embed_single(question)

        # Step 2: Retrieve top-k chunks
        hits = self.vector_store.search(query_vector, top_k=TOP_K)

        # Step 3: Relevance threshold check
        if not hits or hits[0]["score"] < RELEVANCE_THRESHOLD:
            return Answer(
                question=question,
                answer_text=(
                    "The information is not available in the supplied documents."
                ),
                citations=[],
                found_in_docs=False,
            )

        # Step 4: Generate answer via LLM
        answer_text = self.llm.generate_answer(question, hits)

        # Step 5: Build citation list from retrieved chunks
        # De-duplicate by (doc_name, page_number) while preserving order
        seen: set = set()
        citations: List[Citation] = []
        for hit in hits:
            key = (hit["doc_name"], hit["page_number"])
            if key not in seen:
                seen.add(key)
                citations.append(
                    Citation(
                        doc_name=hit["doc_name"],
                        page_number=hit["page_number"],
                        snippet=hit["text"][:300],  # truncate long snippets
                        score=hit["score"],
                    )
                )

        return Answer(
            question=question,
            answer_text=answer_text,
            citations=citations,
            found_in_docs=True,
        )
