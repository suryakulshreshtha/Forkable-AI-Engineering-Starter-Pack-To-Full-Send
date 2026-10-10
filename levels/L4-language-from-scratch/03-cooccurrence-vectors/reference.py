"""
Title: Co-occurrence Vectors
XP: 25

## The job
Turn words into vectors using only counting. `build_vectors(sentences, window)` maps each
word to a sparse `Counter` of the words seen within `window` positions on either side.
`cosine(u, v)` compares two sparse vectors (0.0 if either is all zeros). `nearest(word,
vectors, k)` returns the k most similar other words as `(word, score)` pairs, best first,
ties broken alphabetically. Unknown words raise `KeyError`.

## Why it matters
"You shall know a word by the company it keeps" (J. R. Firth, 1957) is the idea behind every embedding model,
including the ones powering semantic search and retrieval-augmented generation. Learned
embeddings are denser and smarter, but the contract is the same: similar usage, nearby
vectors, compared with cosine.

## Hints
- For position i, the neighbours are positions max(0, i - window) through i + window, skipping i itself.
- Store vectors sparsely: a Counter only keeps the contexts that actually occurred.
- Cosine = dot(u, v) / (norm(u) * norm(v)). For sparse dicts, loop over the shorter one for the dot product.
- Sort candidates with key=lambda pair: (-pair[1], pair[0]) to get "best first, then alphabetical".
"""
from __future__ import annotations

import math
from collections import Counter


def build_vectors(sentences, window=2):
    """sentences: list of token lists. Returns {word: Counter(context -> count)}."""
    # >>> fullsend: count neighbours within the window
    vectors = {}
    for tokens in sentences:
        for i, word in enumerate(tokens):
            ctx = vectors.setdefault(word, Counter())
            lo, hi = max(0, i - window), min(len(tokens), i + window + 1)
            for j in range(lo, hi):
                if j != i:
                    ctx[tokens[j]] += 1
    return vectors
    # <<< fullsend


def cosine(u, v):
    """Cosine similarity of two sparse vectors given as dicts."""
    # >>> fullsend: sparse dot over the product of norms
    if len(u) > len(v):
        u, v = v, u
    dot = sum(val * v.get(key, 0) for key, val in u.items())
    nu = math.sqrt(sum(x * x for x in u.values()))
    nv = math.sqrt(sum(x * x for x in v.values()))
    if nu == 0 or nv == 0:
        return 0.0
    return dot / (nu * nv)
    # <<< fullsend


def nearest(word, vectors, k=3):
    """The k most similar other words, best first, ties alphabetical."""
    # >>> fullsend: score every other word and sort
    if word not in vectors:
        raise KeyError(word)
    target = vectors[word]
    scored = [(other, cosine(target, vec)) for other, vec in vectors.items() if other != word]
    scored.sort(key=lambda pair: (-pair[1], pair[0]))
    return scored[:k]
    # <<< fullsend
