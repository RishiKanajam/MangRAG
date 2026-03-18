https://mangrag.streamlit.app/

# 🥭 MangRAG

A production-ready RAG (Retrieval-Augmented Generation) pipeline that lets you upload PDF documents and ask questions about them using natural language.

**Stack:** MongoDB Atlas Vector Search · Groq (LLaMA 3.3) · HuggingFace Embeddings · FastAPI · Streamlit

---

## How it works

PDF → Load → Chunk → Embed (local) → Store (MongoDB) → Vector Search → Groq LLM → Answer



1. **Ingest** — Upload a PDF or paste a URL. The document is split into chunks, each chunk is converted into a vector embedding using a local HuggingFace model, and stored in MongoDB Atlas.
2. **Ask** — Type a question. Your query is embedded and matched against stored chunks via vector similarity search. The top matches are sent to Groq's LLaMA 3.3 to generate an answer.

---

## Quickstart

### 1. Clone and install
```bash
git clone https://github.com/YOUR_USERNAME/MangRAG.git
cd MangRAG
pip install uv
uv sync
2. Set up environment

cp .env.example .env
Edit .env with your credentials:


GROQ_API_KEY=your_groq_api_key        # free at console.groq.com
MONGODB_URI=mongodb+srv://...         # MongoDB Atlas connection string
3. Run
Streamlit UI


uv run streamlit run app.py
REST API (with Swagger at http://localhost:8000/docs)


uv run uvicorn api:app --reload
CLI


uv run python main.py ingest https://example.com/doc.pdf
uv run python main.py ask "What are the key findings?"
Docker


docker-compose up --build
API Endpoints
Method	Endpoint	Description
GET	/health	Service health check
GET	/stats	Collection statistics
POST	/ingest	Ingest a PDF by URL
POST	/query	Ask a question
Project Structure

MangRAG/
├── mangrag/          # Core Python package
│   ├── config.py     # Settings (Pydantic)
│   ├── models.py     # Request/response models
│   ├── db.py         # MongoDB client
│   ├── embeddings.py # HuggingFace embeddings
│   ├── llm.py        # Groq LLM
│   ├── ingest.py     # Ingest pipeline
│   └── query.py      # Query pipeline
├── api.py            # FastAPI app
├── app.py            # Streamlit UI
├── main.py           # CLI
├── tests/            # pytest tests
├── Dockerfile
└── docker-compose.yml
Running Tests

uv run pytest tests/ -v
Environment Variables
Variable	Required	Description
GROQ_API_KEY	✅	Groq API key (free at console.groq.com)
MONGODB_URI	✅	MongoDB Atlas connection string
MONGODB_DB	optional	Database name (default: mangrag)
MONGODB_COLLECTION	optional	Collection name (default: documents)

