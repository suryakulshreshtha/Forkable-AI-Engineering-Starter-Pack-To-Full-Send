"""
Title: Retrieve
XP: 30

## The job
Index chunks for search. `chunk_tokens(chunk)` returns the content words of the heading
(counted twice, because headings are dense with meaning) followed by the content words of
the text. `Index(chunks)` builds a BM25 ranker over those tokens. `search(query, k,
max_per_path)` returns up to k `(chunk, score)` pairs, best first, never more than
`max_per_path` from the same file when that limit is set. A query made only of stopwords
returns nothing.

## Why it matters
Retrieval quality caps answer quality: the best writer in the world cannot cite a passage
that never reached it. Two cheap tricks pay off constantly in practice: field weighting
(a heading match means more than a body match) and result diversity (five chunks from one
long file crowd out the one paragraph elsewhere that actually answers the question).

## Hints
- `kit.text.content_words` lowercases, tokenises and drops stopwords. Use it for chunks and queries alike.
- `kit.retrieval.BM25` is unlocked; ask it for every positive-scoring document, then apply your own per-path limit while collecting the top k.
- Keep a dict of how many results you have taken from each path.
"""
from __future__ import annotations

from kit.retrieval import BM25
from kit.text import content_words


def chunk_tokens(chunk):
    """Heading words twice, then body words."""
    # >>> fullsend: weight the heading by repeating it
    heading = content_words(chunk.get("heading", ""))
    return heading + heading + content_words(chunk["text"])
    # <<< fullsend


class Index:
    def __init__(self, chunks):
        # >>> fullsend: keep the chunks, build BM25 over their tokens
        self.chunks = list(chunks)
        self.bm25 = BM25([chunk_tokens(c) for c in self.chunks])
        # <<< fullsend

    def search(self, query, k=3, max_per_path=None):
        """[(chunk, score), ...] best first, at most k, at most max_per_path per file."""
        # >>> fullsend: tokenize the query, rank everything, collect with the per-path cap
        terms = content_words(query)
        if not terms:
            return []
        results, taken = [], {}
        for i, score in self.bm25.search(terms, top_k=len(self.chunks)):
            chunk = self.chunks[i]
            if max_per_path is not None and taken.get(chunk["path"], 0) >= max_per_path:
                continue
            taken[chunk["path"]] = taken.get(chunk["path"], 0) + 1
            results.append((chunk, score))
            if len(results) == k:
                break
        return results
        # <<< fullsend
