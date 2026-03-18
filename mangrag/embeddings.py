import logging
from langchain_community.embeddings import HuggingFaceEmbeddings

from .config import settings

logger = logging.getLogger(__name__)

_model: HuggingFaceEmbeddings | None = None


def get_model() -> HuggingFaceEmbeddings:
    global _model
    if _model is None:
        logger.info("Loading embedding model: %s", settings.embedding_model)
        _model = HuggingFaceEmbeddings(model_name=settings.embedding_model)
        logger.info("Embedding model loaded")
    return _model


def embed(text: str) -> list[float]:
    return get_model().embed_query(text)
