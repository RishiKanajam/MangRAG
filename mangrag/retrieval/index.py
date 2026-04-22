import logging
from threading import Lock

from mangrag.config import settings
from mangrag.db import get_collection
from .faiss_retriever import FAISSRetriever
from .bm25_retriever import BM25Retriever
from .hybrid import HybridRetriever

logger = logging.getLogger(__name__)

_retriever: HybridRetriever | None = None
_lock = Lock()


def get_retriever() -> HybridRetriever:
    global _retriever
    if _retriever is None:
        with _lock:
            if _retriever is None:
                _retriever = _build()
    return _retriever


def rebuild_retriever() -> None:
    """Rebuild FAISS + BM25 indices from MongoDB. Call after each ingest."""
    global _retriever
    with _lock:
        _retriever = _build()


def _build() -> HybridRetriever:
    faiss_r = FAISSRetriever(dims=settings.embedding_dims)
    bm25_r = BM25Retriever()
    try:
        col = get_collection()
        raw = list(col.find(
            {}, {'_id': 1, 'content': 1, 'embedding': 1, 'source': 1, 'page': 1}
        ))
        embeddings = [d.pop('embedding', []) for d in raw]
        faiss_r.build(raw, embeddings)
        bm25_r.build(raw)
        logger.info("Hybrid index built: %d documents", len(raw))
    except Exception as exc:
        logger.warning("Could not build hybrid index from MongoDB: %s", exc)
    return HybridRetriever(faiss_r, bm25_r)
