"""
Title: BM25
XP: 30

## The job
Implement BM25, the ranking function behind most keyword search. `BM25(docs, k1, b)` takes
documents as token lists. `idf(term)` is ln((N - n + 0.5) / (n + 0.5) + 1), where n is
how many documents contain the term. `score(query, i)` sums, over query terms,
idf * f * (k1 + 1) / (f + k1 * (1 - b + b * len_i / avg_len)). `search(query, top_k)`
returns `(doc_index, score)` pairs with positive scores, best first, ties by lower index.

## Why it matters
BM25 is decades old and still a strong baseline that embedding search has to beat,
especially for names, codes and rare words. Many production systems run it alongside
vector search. Its two knobs encode real intuitions: the tenth mention of a word matters
less than the first (k1), and a match in a short document means more than in a long one (b).

## Hints
- Precompute in `__init__`: a `Counter` of term frequencies per document, each document's length, the average length, and how many documents contain each term.
- The +1 inside the log keeps idf positive even for terms that appear in most documents.
- In `score`, skip terms with frequency zero; they contribute nothing.
- Sort with key=lambda pair: (-pair[1], pair[0]), drop zero scores, then cut to top_k.
"""
from __future__ import annotations

import math
from collections import Counter


class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        """docs: list of token lists."""
        # >>> fullsend: precompute term frequencies, lengths, average length, document frequencies
        self.docs = [list(d) for d in docs]
        self.k1, self.b = k1, b
        self.n = len(self.docs)
        self.lengths = [len(d) for d in self.docs]
        self.avg_len = (sum(self.lengths) / self.n) if self.n else 0.0
        self.tfs = [Counter(d) for d in self.docs]
        self.df = Counter()
        for tf in self.tfs:
            self.df.update(tf.keys())
        # <<< fullsend

    def idf(self, term):
        # >>> fullsend: ln((N - n + 0.5) / (n + 0.5) + 1)
        n_t = self.df.get(term, 0)
        return math.log((self.n - n_t + 0.5) / (n_t + 0.5) + 1.0)
        # <<< fullsend

    def score(self, query, i):
        """BM25 score of document i for a list of query tokens."""
        # >>> fullsend: saturating term frequency times idf, length-normalised
        tf, length = self.tfs[i], self.lengths[i]
        norm = 1.0 - self.b + self.b * (length / self.avg_len if self.avg_len else 0.0)
        total = 0.0
        for term in query:
            f = tf.get(term, 0)
            if f:
                total += self.idf(term) * f * (self.k1 + 1) / (f + self.k1 * norm)
        return total
        # <<< fullsend

    def search(self, query, top_k=5):
        """[(doc_index, score), ...] with score > 0, best first, ties by index."""
        # >>> fullsend: score all, drop zeros, sort, cut
        query = list(query)
        scored = [(i, self.score(query, i)) for i in range(self.n)]
        scored = [(i, s) for i, s in scored if s > 0]
        scored.sort(key=lambda pair: (-pair[1], pair[0]))
        return scored[:top_k]
        # <<< fullsend
