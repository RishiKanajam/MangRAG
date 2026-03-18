import logging
from pymongo.collection import Collection

from .config import settings
from .embeddings import embed
from .llm import generate

logger = logging.getLogger(__name__)


def retrieve(query: str, collection: Collection, top_k: int | None = None) -> list[dict]:
    top_k = top_k or settings.top_k
    logger.info("Retrieving top %d chunks for query: %r", top_k, query)

    pipeline = [
        {
            "$vectorSearch": {
                "index": settings.index_name,
                "queryVector": embed(query),
                "path": "embedding",
                "numCandidates": settings.num_candidates,
                "limit": top_k,
            }
        },
        {
            "$project": {
                "_id": 0,
                "content": 1,
                "source": 1,
                "page": 1,
                "score": {"$meta": "vectorSearchScore"},
            }
        },
    ]

    docs = list(collection.aggregate(pipeline))
    logger.info("Retrieved %d chunks", len(docs))
    return docs


def build_answer(query: str, docs: list[dict]) -> str:
    if not docs:
        return "I could not find relevant information to answer your question."

    context = "\n\n---\n\n".join(
        f"[Page {doc.get('page', '?')}] {doc['content']}"
        for doc in docs
    )

    prompt = f"""You are a helpful assistant answering questions based on retrieved document excerpts.

Rules:
- Answer using the provided context. Be as helpful as possible with what is available.
- If asked for a summary, summarise the key points visible in the context.
- Only say you cannot answer if the context is completely unrelated to the question.
- Keep your answer concise and well-structured.

Context:
{context}

Question: {query}
Answer:"""

    return generate(prompt)


def run(
    query: str,
    collection: Collection,
    top_k: int | None = None,
) -> tuple[str, list[dict]]:
    docs = retrieve(query, collection, top_k=top_k)
    answer = build_answer(query, docs)
    return answer, docs
