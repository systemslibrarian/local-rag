# Local RAG

A fully **private, offline** Retrieval-Augmented Generation (RAG) application built with Streamlit, LangChain, PGVector, and Ollama. Upload PDF documents, ask questions in natural language, and receive **cited, grounded answers** — all without sending a single byte to the cloud.

---

## Features

| Category | Details |
|---|---|
| **Document ingestion** | Upload PDFs, background indexing with real-time job status panel |
| **Cited answers** | Every response includes file name, page number, and excerpt from the source |
| **No-evidence guard** | Similarity threshold prevents the LLM from hallucinating when content is irrelevant |
| **Multi-query retrieval** | Generates multiple query variants for better recall (toggleable) |
| **Chat management** | Create, rename, search, and sort chats; cascade-delete cleans all data |
| **Document management** | Re-index or delete individual files with full vector cleanup |
| **Retrieval settings** | Per-chat sliders: Top-K chunks, citation limit, multi-query toggle |
| **Export** | Download any conversation as a Markdown file |
| **Startup health checks** | Validates DB connectivity, Ollama availability, and temp folder on launch |
| **Structured JSON logging** | Latency metrics around retrieval, embedding, and generation |
| **Accessibility** | WCAG AA contrast ratios, `role="log"` live region, keyboard focus rings, `aria-hidden` on decorative elements |
| **Mobile-friendly** | Responsive CSS with `@media` breakpoints for narrow viewports |

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
- PostgreSQL with the **pgvector** extension enabled
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

Copy and edit the environment file:

```bash
cp .env .env.local   # or just edit .env in-place
```

| Variable | Description | Default |
|---|---|---|
| `PG_DSN` | PostgreSQL connection string | `postgresql+psycopg://user:password@localhost:5432/db` |
| `COLLECTION_NAME` | PGVector collection name | `local-rag` |
| `LLM_MODEL` | Ollama chat model | `llama3.2` |
| `TEXT_EMBEDDING_MODEL` | Ollama embedding model | `nomic-embed-text` |
| `OLLAMA_HOST` | Ollama base URL | `http://localhost:11434` |
| `TEMP_FOLDER` | Scratch folder for PDF processing | `/tmp/local-rag` |

---

## Database setup

Run Alembic migrations before first launch:

```bash
alembic upgrade head
```

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
├── app/
│   ├── ai/                         # LLM query, retrieval, citations
│   ├── chat/                       # Chat CRUD, rename, cascade delete
│   ├── file/                       # File upload, background indexing, vector management
│   │   └── model/index_job.py      # Persistent indexing job records
│   └── message/                    # Message storage and rendering
├── internal/
│   ├── config/
│   │   ├── setting.py              # Pydantic settings from .env
│   │   └── logging_config.py       # Structured JSON logging + timed() helper
│   ├── di/container.py             # dependency-injector wiring
│   └── domain/                     # Base repository and entity classes
├── migration/                      # Alembic migrations
│   └── versions/
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
         │       └── IndexJobRepository (tracks background jobs in DB)
         │
         └── AIService ──► similarity_search_with_relevance_scores
                  │              (threshold filter → no-hallucination guard)
                  ├── MultiQueryRetriever (optional)
                  └── ChatOllama → cited AIAnswer
```

---

## Contributing

Pull requests are welcome. Please open an issue first to discuss significant changes.

## License

MIT


Contributions are welcome! Please feel free to submit a Pull Request.

## Support

For support, please open an issue in the repository.

---

*This project was built with ❤️ using Streamlit, Langchain and other amazing open-source tools.*
