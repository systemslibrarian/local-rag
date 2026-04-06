# Local RAG

A fully **private, offline** Retrieval-Augmented Generation (RAG) application built with Streamlit, LangChain, PGVector, and Ollama. Upload PDF documents, ask questions in natural language, and receive **cited, grounded answers** — all without sending a single byte to the cloud.

---

## Features

| Category | Details |
|---|---|
| **Streaming responses** | LLM output streams token-by-token with a live cursor; no waiting for the full response |
| **Conversation memory** | Sliding window of past messages sent to the LLM for coherent follow-up questions |
| **Document ingestion** | Upload PDFs, background indexing with auto-refreshing real-time job status panel |
| **Disk-based file storage** | PDF files stored on disk (`FILE_STORAGE_FOLDER`), not as DB blobs |
| **Cited answers** | Every response includes file name, page number, and excerpt from the source |
| **No-evidence guard** | Configurable similarity threshold prevents hallucination when content is irrelevant |
| **Multi-query retrieval** | Generates multiple query variants for better recall (toggleable) |
| **Chat management** | Create, rename, search, and sort chats; cascade-delete cleans all data |
| **Document management** | Re-index or delete individual files with full vector cleanup |
| **Retrieval settings** | Per-chat sliders: Top-K chunks, citation limit, multi-query toggle |
| **Export** | Download any conversation as a Markdown file |
| **DB connection pooling** | Configurable `pool_size`, `max_overflow`, `pool_timeout`, `pool_recycle` |
| **Startup health checks** | Validates DB connectivity, Ollama availability, and storage folder on launch |
| **Structured JSON logging** | Latency metrics around retrieval, embedding, and generation |
| **Accessibility** | WCAG AA contrast ratios, `role="log"` live region, keyboard focus rings, `aria-hidden` on decorative elements |
| **Mobile-friendly** | Responsive CSS with `@media` breakpoints for narrow viewports |
| **Docker Compose** | One-command stack: pgvector, Ollama (auto-pulls models), and the app |
| **Test suite** | pytest + pytest-asyncio covering `AIService` and `FileService` |

---

## Models

| Purpose | Default model |
|---|---|
| LLM | `llama3.2` |
| Embeddings | `nomic-embed-text` |

Change both in `.env` — no code changes needed.

---

## Prerequisites

- Python **3.12+**
- [Ollama](https://ollama.com) running locally (`http://localhost:11434`)
- PostgreSQL with the **pgvector** extension enabled — **or** use Docker Compose (see below)
- `pip` / `venv`

### Pull required Ollama models

```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

### Enable pgvector in PostgreSQL

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

---

## Installation

```bash
# 1. Clone
git clone https://github.com/dbunt1tled/local-rag.git
cd local-rag

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Configuration

Copy the example env file and edit it:

```bash
cp .env.example .env
```

| Variable | Description | Default |
|---|---|---|
| `PG_DSN` | PostgreSQL connection string | `postgresql+psycopg://user:password@localhost:5432/db` |
| `COLLECTION_NAME` | PGVector collection name | `local-rag` |
| `LLM_MODEL` | Ollama chat model | `llama3.2` |
| `TEXT_EMBEDDING_MODEL` | Ollama embedding model | `nomic-embed-text` |
| `OLLAMA_HOST` | Ollama base URL | `http://localhost:11434` |
| `FILE_STORAGE_FOLDER` | Directory where uploaded PDFs are saved | `./data/files` |
| `SIMILARITY_THRESHOLD` | Minimum relevance score (0–1) to include a chunk | `0.30` |
| `HISTORY_WINDOW` | Number of past message pairs sent to the LLM | `6` |
| `DB_POOL_SIZE` | SQLAlchemy connection pool size | `5` |
| `DB_MAX_OVERFLOW` | Max connections above pool size | `10` |

---

## Database setup

Run Alembic migrations before first launch:

```bash
alembic upgrade head
```

---

## Docker Compose (recommended)

The easiest way to run the full stack with no manual setup:

```bash
cp .env.example .env   # review and adjust if needed
docker compose up -d
```

This starts:
- **postgres** — pgvector-enabled PostgreSQL 16
- **ollama** — pulls `llama3.2` and `nomic-embed-text` automatically on first boot
- **app** — runs `alembic upgrade head` then `streamlit run Home.py`

Open **http://localhost:8501** in your browser.

---

## Running

```bash
streamlit run Home.py
```

Open **http://localhost:8501** in your browser.

---

## Project structure

```
local-rag/
├── Home.py                         # Streamlit entrypoint + health checks
├── Dockerfile
├── docker-compose.yml
├── pytest.ini
├── tests/
│   ├── test_ai_service.py          # AIService unit tests
│   └── test_file_service.py        # FileService unit tests
├── app/
│   ├── ai/                         # LLM streaming query, retrieval, citations
│   ├── chat/                       # Chat CRUD, rename, cascade delete
│   ├── file/                       # File upload, background indexing, vector management
│   │   └── model/index_job.py      # Persistent indexing job records
│   └── message/                    # Message storage and streaming rendering
├── internal/
│   ├── config/
│   │   ├── setting.py              # Pydantic settings from .env
│   │   └── logging_config.py       # Structured JSON logging + timed() helper
│   ├── di/container.py             # dependency-injector wiring
│   └── domain/                     # Base repository and entity classes
├── migration/                      # Alembic migrations
│   └── versions/
├── data/files/                     # Uploaded PDFs (created at runtime, git-ignored)
├── alembic.ini
├── pyproject.toml
└── requirements.txt
```

---

## Architecture overview

```
User → Streamlit UI
         │
         ├── FileService ──► PyPDFLoader → TextSplitter → PGVector (embeddings)
         │       │                └── PDF written to FILE_STORAGE_FOLDER on disk
         │       └── IndexJobRepository (tracks background jobs in DB)
         │
         └── AIService ──► similarity_search_with_relevance_scores
                  │              (threshold filter → no-hallucination guard)
                  ├── MultiQueryRetriever (optional)
                  ├── MessagesPlaceholder (conversation memory, sliding window)
                  └── ChatOllama.astream() → token chunks → live Streamlit UI
                                              └── Citations appended at end
```

---

## Running tests

```bash
pytest
```

---

## Contributing

Pull requests are welcome. Please open an issue first to discuss significant changes.

## License

MIT
