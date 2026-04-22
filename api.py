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
from mangrag.retrieval.index import get_retriever
from mangrag import eval as rag_eval
from mangrag.models import (
    IngestRequest, IngestResponse,
    QueryRequest, QueryResponse, RetrievedChunk,
    StatsResponse, HealthResponse,
    EvaluateRequest, EvaluateResponse,
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
    logger.info("Building hybrid FAISS+BM25 index from MongoDB...")
    get_retriever()
    logger.info("Ready")
    yield
    logger.info("Shutting down")


app = FastAPI(
    title="MangRAG API",
    description="RAG pipeline with hybrid FAISS+BM25 retrieval + Groq LLM",
    version="2.0.0",
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
        answer, docs = query.run(req.query, top_k=req.top_k)
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


@app.post("/evaluate", response_model=EvaluateResponse, tags=["evaluation"])
def evaluate(req: EvaluateRequest):
    """Run precision@k and faithfulness evaluation for a single query."""
    logger.info("Evaluate: %r (k=%d)", req.query, req.k)
    try:
        result = rag_eval.evaluate(req.query, req.relevant_texts, k=req.k)
        return EvaluateResponse(
            query=result['query'],
            precision_at_k=result[f'precision@{req.k}'],
            faithfulness=result['faithfulness'],
            answer=result['answer'],
            retrieved_count=result['retrieved_count'],
        )
    except Exception as e:
        logger.exception("Evaluation failed")
        raise HTTPException(status_code=500, detail=str(e))
