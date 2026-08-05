"""
config.py
---------
Central configuration for Answer-Buddy.
All constants and environment variable loading live here.
"""

import os
from dotenv import load_dotenv

# Load .env from the project root (one level above src/)
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))


# ── OpenRouter ────────────────────────────────────────────────────────────────
OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"

# Free model on OpenRouter — change if this model becomes unavailable
# Alternatives: "meta-llama/llama-3.1-8b-instruct:free", "mistralai/mistral-7b-instruct:free"
LLM_MODEL: str = "google/gemma-3-27b-it:free"


# ── Qdrant ────────────────────────────────────────────────────────────────────
QDRANT_URL: str = os.getenv("QDRANT_URL", "")
QDRANT_API_KEY: str = os.getenv("QDRANT_API_KEY", "")
QDRANT_COLLECTION_NAME: str = "answer_buddy_docs"


# ── Embedding Model ───────────────────────────────────────────────────────────
# Runs locally — no API key required
EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION: int = 384  # Output dimension for all-MiniLM-L6-v2


# ── Chunking ──────────────────────────────────────────────────────────────────
CHUNK_SIZE: int = 500        # Max characters per chunk
CHUNK_OVERLAP: int = 80      # Character overlap between adjacent chunks


# ── Retrieval ─────────────────────────────────────────────────────────────────
TOP_K: int = 5                    # Number of chunks to retrieve per query
RELEVANCE_THRESHOLD: float = 0.30 # Minimum similarity score; below = "not found"


# ── PDF directory ─────────────────────────────────────────────────────────────
PDF_DIR: str = os.path.join(os.path.dirname(__file__), "..", "pdfs")
