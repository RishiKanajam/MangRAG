import logging
from collections.abc import Callable
from pymongo.collection import Collection
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import settings
from .embeddings import embed
from .db import ensure_vector_index

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[int, int], None]


def load_pdf(source: str) -> list:
    logger.info("Loading PDF: %s", source)
    pages = PyPDFLoader(source).load()
    logger.info("Loaded %d pages", len(pages))
    return pages


def chunk_documents(pages: list) -> list:
    logger.info(
        "Chunking: size=%d overlap=%d", settings.chunk_size, settings.chunk_overlap
    )
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    chunks = [c for c in splitter.split_documents(pages) if c.page_content.strip()]
    logger.info("Produced %d chunks", len(chunks))
    return chunks


def embed_and_store(
    chunks: list,
    collection: Collection,
    source: str,
    on_progress: ProgressCallback | None = None,
) -> int:
    logger.info("Embedding %d chunks...", len(chunks))
    docs = []
    for i, chunk in enumerate(chunks):
        docs.append({
            "content": chunk.page_content,
            "embedding": embed(chunk.page_content),
            "source": source,
            "page": chunk.metadata.get("page", 0),
        })
        if on_progress:
            on_progress(i + 1, len(chunks))

    collection.insert_many(docs)
    logger.info("Stored %d documents", len(docs))
    return len(docs)


def run(
    source: str,
    collection: Collection,
    on_progress: ProgressCallback | None = None,
) -> int:
    from .retrieval.index import rebuild_retriever

    pages = load_pdf(source)
    chunks = chunk_documents(pages)
    count = embed_and_store(chunks, collection, source, on_progress=on_progress)
    ensure_vector_index(collection)
    rebuild_retriever()
    return count
