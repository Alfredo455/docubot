"""BM25 retrieval with a smoothed IDF (stays positive on tiny corpora)."""
import math
import re
from collections import Counter

from .ingest import Chunk

_TOKEN = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class Retriever:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1, self.b = k1, b
        self._tf = [Counter(tokenize(c.text)) for c in chunks]
        self._len = [sum(tf.values()) for tf in self._tf]
        self._avg = (sum(self._len) / len(chunks)) if chunks else 0.0
        df: Counter[str] = Counter()
        for tf in self._tf:
            df.update(tf.keys())
        n = len(chunks)
        self._idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    def _score(self, i: int, query_terms: list[str]) -> float:
        tf, length = self._tf[i], self._len[i]
        score = 0.0
        for t in query_terms:
            f = tf.get(t, 0)
            if not f:
                continue
            norm = f + self.k1 * (1 - self.b + self.b * length / self._avg)
            score += self._idf[t] * f * (self.k1 + 1) / norm
        return score

    def search(self, query: str, k: int = 3) -> list[tuple[Chunk, float]]:
        terms = tokenize(query)
        scored = [(c, self._score(i, terms)) for i, c in enumerate(self.chunks)]
        scored.sort(key=lambda p: p[1], reverse=True)
        return [(c, s) for c, s in scored[:k] if s > 0]
