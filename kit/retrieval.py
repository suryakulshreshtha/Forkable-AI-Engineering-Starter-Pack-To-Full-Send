"""Chunking and BM25 ranking. Unlocked by drills 6.1 and 6.2."""
from __future__ import annotations

import math
from collections import Counter


def chunk_words(text, size, overlap=0):
    if size <= 0:
        raise ValueError("size must be positive")
    if not 0 <= overlap < size:
        raise ValueError("overlap must be in [0, size)")
    tokens = text.split()
    if not tokens:
        return []
    step = size - overlap
    chunks = []
    for start in range(0, len(tokens), step):
        chunks.append(" ".join(tokens[start:start + size]))
        if start + size >= len(tokens):
            break
    return chunks


class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.docs = [list(d) for d in docs]
        self.k1, self.b = k1, b
        self.n = len(self.docs)
        self.lengths = [len(d) for d in self.docs]
        self.avg_len = (sum(self.lengths) / self.n) if self.n else 0.0
        self.tfs = [Counter(d) for d in self.docs]
        df = Counter()
        for tf in self.tfs:
            df.update(tf.keys())
        self.df = df

    def idf(self, term):
        n_t = self.df.get(term, 0)
        return math.log((self.n - n_t + 0.5) / (n_t + 0.5) + 1.0)

    def score(self, query, i):
        tf, length = self.tfs[i], self.lengths[i]
        norm = 1.0 - self.b + self.b * (length / self.avg_len if self.avg_len else 0.0)
        total = 0.0
        for term in query:
            f = tf.get(term, 0)
            if f:
                total += self.idf(term) * f * (self.k1 + 1) / (f + self.k1 * norm)
        return total

    def search(self, query, top_k=5):
        query = list(query)
        scored = [(i, self.score(query, i)) for i in range(self.n)]
        scored = [(i, s) for i, s in scored if s > 0]
        scored.sort(key=lambda pair: (-pair[1], pair[0]))
        return scored[:top_k]
