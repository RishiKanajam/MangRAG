import logging
import dotenv
dotenv.load_dotenv()

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from mangrag import ingest, query
from mangrag.config import settings
from mangrag.db import get_collection, ensure_vector_index
from mangrag.embeddings import get_model
from mangrag.models import (
    IngestRequest, IngestResponse,
    QueryRequest, QueryResponse, RetrievedChunk,
    StatsResponse, HealthResponse,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting MangRAG API — pre-loading embedding model...")
    get_model()
    logger.info("Ready")
    yield
    logger.info("Shutting down")


app = FastAPI(
    title="MangRAG API",
    description="RAG pipeline backed by MongoDB Atlas Vector Search + Groq",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["system"])
def health():
    try:
        get_collection().find_one({}, {"_id": 1})
        mongo_status = "ok"
    except Exception as e:
        mongo_status = f"error: {e}"

    return HealthResponse(
        status="ok",
        mongodb=mongo_status,
        embeddings="ok",
    )


@app.get("/stats", response_model=StatsResponse, tags=["system"])
def stats():
    col = get_collection()
    return StatsResponse(
        total_chunks=col.count_documents({}),
        sources=col.distinct("source"),
    )


@app.post(
    "/ingest",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["pipeline"],
)
def ingest_document(req: IngestRequest):
    logger.info("Ingest request: %s", req.source)
    try:
        col = get_collection()
        count = ingest.run(req.source, col)
        return IngestResponse(source=req.source, chunks_stored=count)
    except Exception as e:
        logger.exception("Ingest failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryResponse, tags=["pipeline"])
def query_document(req: QueryRequest):
    logger.info("Query: %r (top_k=%d)", req.query, req.top_k)
    try:
        col = get_collection()
        answer, docs = query.run(req.query, col, top_k=req.top_k)
        chunks = [
            RetrievedChunk(
                content=d["content"],
                source=d.get("source", ""),
                page=d.get("page", 0),
                score=round(d.get("score", 0.0), 4),
            )
            for d in docs
        ]
        return QueryResponse(query=req.query, answer=answer, chunks=chunks)
    except Exception as e:
        logger.exception("Query failed")
        raise HTTPException(status_code=500, detail=str(e))
