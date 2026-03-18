"""
Unit tests for the MangRAG pipeline.

Run with:
    uv run pytest tests/ -v
"""
import pytest
from unittest.mock import MagicMock, patch


# ─── Config ──────────────────────────────────────────────────────────────────

def test_settings_loads():
    from mangrag.config import settings
    assert settings.embedding_dims == 384
    assert settings.chunk_size > 0
    assert settings.top_k > 0


# ─── Ingest pipeline ─────────────────────────────────────────────────────────

def test_chunk_documents_filters_empty():
    from mangrag.ingest import chunk_documents
    from langchain_core.documents import Document

    pages = [Document(page_content="Hello world. " * 100, metadata={"page": 0})]
    chunks = chunk_documents(pages)
    assert len(chunks) > 0
    assert all(c.page_content.strip() for c in chunks)


def test_chunk_documents_respects_size():
    from mangrag.ingest import chunk_documents
    from mangrag.config import settings
    from langchain_core.documents import Document

    long_text = "word " * 2000
    pages = [Document(page_content=long_text, metadata={"page": 0})]
    chunks = chunk_documents(pages)
    for chunk in chunks:
        assert len(chunk.page_content) <= settings.chunk_size + settings.chunk_overlap


# ─── Query pipeline ──────────────────────────────────────────────────────────

def test_build_answer_no_docs():
    from mangrag.query import build_answer
    result = build_answer("anything", [])
    assert "could not find" in result.lower()


def test_build_answer_calls_llm():
    from mangrag import query
    docs = [{"content": "MongoDB is a NoSQL database.", "page": 1}]

    with patch("mangrag.query.generate", return_value="MongoDB is NoSQL.") as mock_gen:
        result = query.build_answer("What is MongoDB?", docs)
        assert mock_gen.called
        assert result == "MongoDB is NoSQL."


# ─── Embeddings ──────────────────────────────────────────────────────────────

def test_embed_returns_correct_dims():
    from mangrag.embeddings import embed
    from mangrag.config import settings
    vector = embed("test sentence")
    assert isinstance(vector, list)
    assert len(vector) == settings.embedding_dims


# ─── API ─────────────────────────────────────────────────────────────────────

def test_health_endpoint():
    from fastapi.testclient import TestClient
    from api import app

    with patch("api.get_collection") as mock_col:
        mock_col.return_value.find_one.return_value = None
        client = TestClient(app)
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"


def test_stats_endpoint():
    from fastapi.testclient import TestClient
    from api import app

    with patch("api.get_collection") as mock_col:
        col = MagicMock()
        col.count_documents.return_value = 42
        col.distinct.return_value = ["source_a.pdf"]
        mock_col.return_value = col

        client = TestClient(app)
        response = client.get("/stats")
        assert response.status_code == 200
        data = response.json()
        assert data["total_chunks"] == 42
        assert "source_a.pdf" in data["sources"]
