"""
Title: Rank Fusion and Recall
XP: 25

## The job
Combine rankings and measure them. `rrf(rankings, k)` is reciprocal rank fusion: each
document earns 1 / (k + rank) from every list it appears in (rank starts at 1), and the
result is a list of `(doc_id, score)` sorted by score, ties by id. `recall_at_k` is the
share of relevant documents found in the top k. `reciprocal_rank` is 1 / rank of the first
relevant hit (0.0 if none). `mean_reciprocal_rank` averages it over many queries.

## Why it matters
Hybrid search runs keyword and vector retrieval side by side, and their scores live on
different scales; RRF merges them using ranks alone, which is why it is so popular. Recall
at k answers the question that matters most for retrieval-augmented generation: did the
evidence even make it into the prompt?

## Hints
- Use a dict of running totals. `enumerate(ranking, start=1)` gives you 1-based ranks.
- Sort with key=lambda pair: (-pair[1], pair[0]) for "best first, ties by id".
- recall_at_k with no relevant documents is 0.0, not a crash.
- Reciprocal rank looks only at the first relevant document and ignores the rest.
"""
from __future__ import annotations


def rrf(rankings, k=60):
    """rankings: list of ranked id lists. Returns [(id, score), ...] best first."""
    # >>> fullsend: add 1 / (k + rank) for every appearance
    scores = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking, start=1):
            scores[doc] = scores.get(doc, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))
    # <<< fullsend


def recall_at_k(ranked, relevant, k):
    """|relevant within the top k| / |relevant|."""
    # >>> fullsend: count relevant ids in the first k
    relevant = set(relevant)
    if not relevant:
        return 0.0
    return len(relevant.intersection(ranked[:k])) / len(relevant)
    # <<< fullsend


def reciprocal_rank(ranked, relevant):
    """1 / (1-based position of the first relevant id), or 0.0."""
    # >>> fullsend: find the first hit
    relevant = set(relevant)
    for rank, doc in enumerate(ranked, start=1):
        if doc in relevant:
            return 1.0 / rank
    return 0.0
    # <<< fullsend


def mean_reciprocal_rank(runs):
    """runs: list of (ranked, relevant) pairs. Mean of reciprocal ranks, 0.0 if empty."""
    # >>> fullsend: average reciprocal_rank over runs
    if not runs:
        return 0.0
    return sum(reciprocal_rank(r, rel) for r, rel in runs) / len(runs)
    # <<< fullsend
