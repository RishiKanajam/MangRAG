import numpy as np
import faiss


class FAISSRetriever:
    def __init__(self, dims: int = 384):
        self.dims = dims
        self.index = faiss.IndexFlatIP(dims)
        self.docs: list[dict] = []

    def build(self, docs: list[dict], embeddings: list[list[float]]) -> None:
        self.docs = list(docs)
        self.index = faiss.IndexFlatIP(self.dims)
        if not embeddings:
            return
        vecs = np.array(embeddings, dtype='float32')
        faiss.normalize_L2(vecs)
        self.index.add(vecs)

    def add(self, doc: dict, embedding: list[float]) -> None:
        self.docs.append(doc)
        vec = np.array([embedding], dtype='float32')
        faiss.normalize_L2(vec)
        self.index.add(vec)

    def search(self, query_embedding: list[float], top_k: int) -> list[tuple[dict, float]]:
        if self.index.ntotal == 0:
            return []
        q = np.array([query_embedding], dtype='float32')
        faiss.normalize_L2(q)
        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(q, k)
        return [
            (self.docs[idx], float(scores[0][j]))
            for j, idx in enumerate(indices[0])
            if idx >= 0
        ]
