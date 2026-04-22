from .faiss_retriever import FAISSRetriever
from .bm25_retriever import BM25Retriever


class HybridRetriever:
    """Combines dense (FAISS) and sparse (BM25) retrieval via Reciprocal Rank Fusion."""

    def __init__(self, faiss: FAISSRetriever, bm25: BM25Retriever, rrf_k: int = 60):
        self.faiss = faiss
        self.bm25 = bm25
        self.rrf_k = rrf_k

    def retrieve(
        self, query: str, query_embedding: list[float], top_k: int = 5
    ) -> list[dict]:
        fetch_k = top_k * 3

        dense = self.faiss.search(query_embedding, fetch_k)
        sparse = self.bm25.search(query, fetch_k)

        rrf_scores: dict[str, float] = {}
        doc_map: dict[str, dict] = {}

        for rank, (doc, _) in enumerate(dense):
            key = str(doc.get('_id', id(doc)))
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (self.rrf_k + rank + 1)
            doc_map[key] = doc

        for rank, (doc, _) in enumerate(sparse):
            key = str(doc.get('_id', id(doc)))
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (self.rrf_k + rank + 1)
            doc_map[key] = doc

        ranked = sorted(rrf_scores.items(), key=lambda x: -x[1])[:top_k]
        return [
            {
                'content': doc_map[k]['content'],
                'source': doc_map[k].get('source', ''),
                'page': doc_map[k].get('page', 0),
                'score': score,
            }
            for k, score in ranked
        ]
