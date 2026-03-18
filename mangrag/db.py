import time
import logging
import dns.resolver
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.operations import SearchIndexModel

from .config import settings

logger = logging.getLogger(__name__)

# Fix DNS resolution issues on some networks
dns.resolver.default_resolver = dns.resolver.Resolver(configure=False)
dns.resolver.default_resolver.nameservers = ["8.8.8.8", "8.8.4.4"]

_client: MongoClient | None = None


def get_client() -> MongoClient:
    global _client
    if _client is None:
        _client = MongoClient(settings.mongodb_uri)
    return _client


def get_collection() -> Collection:
    return get_client()[settings.mongodb_db][settings.mongodb_collection]


def ensure_vector_index(collection: Collection) -> None:
    existing = {idx["name"] for idx in collection.list_search_indexes()}
    if settings.index_name in existing:
        logger.debug("Vector index already exists, skipping creation")
        return

    logger.info("Creating vector search index (dims=%d)...", settings.embedding_dims)
    model = SearchIndexModel(
        definition={
            "fields": [{
                "type": "vector",
                "numDimensions": settings.embedding_dims,
                "path": "embedding",
                "similarity": "cosine",
            }]
        },
        name=settings.index_name,
        type="vectorSearch",
    )
    collection.create_search_index(model=model)

    for _ in range(30):
        statuses = [
            idx.get("status")
            for idx in collection.list_search_indexes()
            if idx["name"] == settings.index_name
        ]
        if statuses and statuses[0] == "READY":
            logger.info("Vector index is READY")
            return
        time.sleep(2)

    logger.warning("Vector index creation timed out — it may still be building")
