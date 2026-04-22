import re
from rank_bm25 import BM25Okapi


class BM25Retriever:
    def __init__(self):
        self.bm25: BM25Okapi | None = None
        self.docs: list[dict] = []

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r'\w+', text.lower())

    def build(self, docs: list[dict]) -> None:
        self.docs = list(docs)
        corpus = [self._tokenize(d['content']) for d in self.docs]
        self.bm25 = BM25Okapi(corpus) if corpus else None

    def search(self, query: str, top_k: int) -> list[tuple[dict, float]]:
        if not self.bm25 or not self.docs:
            return []
        tokens = self._tokenize(query)
        scores = self.bm25.get_scores(tokens)
        indices = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_k]
        return [(self.docs[i], float(scores[i])) for i in indices if scores[i] > 0]
