import logging
from .llm import generate

logger = logging.getLogger(__name__)


def precision_at_k(retrieved: list[dict], relevant_texts: list[str], k: int) -> float:
    """Fraction of top-k chunks whose content overlaps with any relevant reference text."""
    top_k = retrieved[:k]
    if not top_k or not relevant_texts:
        return 0.0
    hits = sum(
        1 for doc in top_k
        if any(ref.strip().lower() in doc['content'].lower() for ref in relevant_texts if ref.strip())
    )
    return round(hits / k, 4)


def faithfulness_score(answer: str, context_chunks: list[dict]) -> float:
    """LLM-as-judge: rate how well the answer stays grounded in retrieved context (0.0–1.0)."""
    if not context_chunks or not answer.strip():
        return 0.0
    context = "\n\n".join(
        f"[Chunk {i + 1}] {c['content']}" for i, c in enumerate(context_chunks)
    )
    prompt = f"""You are an evaluation assistant. Rate how faithful the answer is to the provided context.

Rules:
- 1.0: every claim in the answer is directly supported by the context.
- 0.0: the answer contains claims entirely absent from the context.
- Score intermediate values proportionally.

Context:
{context}

Answer:
{answer}

Respond with ONLY a single decimal number between 0.0 and 1.0."""
    try:
        raw = generate(prompt).strip()
        score = float(raw.split()[0])
        return round(max(0.0, min(1.0, score)), 4)
    except Exception as exc:
        logger.warning("Faithfulness scoring failed: %s", exc)
        return 0.0


def evaluate(query: str, relevant_texts: list[str], k: int = 5) -> dict:
    """Run a full precision@k + faithfulness evaluation for one query."""
    from .query import retrieve, build_answer

    docs = retrieve(query, top_k=k)
    answer = build_answer(query, docs)

    return {
        'query': query,
        f'precision@{k}': precision_at_k(docs, relevant_texts, k),
        'faithfulness': faithfulness_score(answer, docs),
        'answer': answer,
        'retrieved_count': len(docs),
    }
