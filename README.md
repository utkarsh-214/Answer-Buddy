# Answer-Buddy

A Retrieval-Augmented Generation (RAG) application that answers questions from PDF documents with accurate citations.

---

## Architecture

```
PDF Files → PyMuPDF (parse + chunk) → SentenceTransformer (embed)
                                              ↓
                                      Qdrant Cloud (store)

User Query → SentenceTransformer (embed) → Qdrant (search top-5)
                                              ↓
                                    OpenRouter LLM (generate)
                                              ↓
                              Answer + Citations (doc, page, snippet)
```

**Components:**

| File | Role |
|---|---|
| `src/config.py` | All constants and env-var loading |
| `src/pdf_parser.py` | PDF ingestion and sliding-window chunking |
| `src/embedder.py` | Local embedding model wrapper |
| `src/vector_store.py` | Qdrant upsert and search |
| `src/llm_client.py` | OpenRouter API calls |
| `src/rag_pipeline.py` | Orchestrates ingest and query flows |
| `main.py` | CLI entry point |

---

## Libraries Used

| Library | Purpose |
|---|---|
| `PyMuPDF` | Fast PDF parsing with page-level text extraction |
| `sentence-transformers` | Local embedding generation |
| `qdrant-client` | Qdrant Cloud vector database client |
| `openai` | OpenRouter API (OpenAI-compatible) |
| `python-dotenv` | `.env` file loading |
| `tqdm` | Progress bars during ingestion |

---

## Embedding Model

**`all-MiniLM-L6-v2`** (via `sentence-transformers`)

- Runs fully locally — no API key or network call after first download
- 384-dimensional vectors
- Normalised embeddings for cosine similarity

---

## Assumptions

- PDFs are placed in the `pdfs/` directory before running `--ingest`
- Qdrant Cloud free tier is used (no local Docker required)
- OpenRouter free-tier model `google/gemma-3-27b-it:free` is used by default
- If the top retrieved chunk scores below `0.30` similarity, the answer is treated as "not found in documents"

---

## How to Run

### 1. Clone and set up virtual environment

```bash
git clone <repo-url>
cd Answer-Buddy

python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` and fill in your keys:

```
OPENROUTER_API_KEY=your_key_here
QDRANT_URL=https://your-cluster.qdrant.io:6333
QDRANT_API_KEY=your_qdrant_key_here
```

### 3. Add your PDFs

Place all PDF files into the `pdfs/` folder.

### 4. Ingest PDFs

```bash
python main.py --ingest
```

To wipe existing data and re-ingest from scratch:

```bash
python main.py --ingest --fresh
```

### 5. Ask questions

**Interactive mode:**
```bash
python main.py --query
```

**Single question:**
```bash
python main.py --query -q "What is the leave policy?"
```

---

## Example Output

```
──────────────────────────────────────────────────────────────────────
Question: What is the leave policy?

Answer:
Employees are entitled to 24 annual leave days per year.

Sources:
  [1] employee_handbook.pdf  |  Page 17  (relevance: 0.82)
      "All full-time employees receive 24 days of annual leave..."

──────────────────────────────────────────────────────────────────────
```

If the answer is not in the documents:

```
Answer:
The information is not available in the supplied documents.
```
