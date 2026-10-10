"""
Title: Self-Attention
XP: 30

## The job
Implement scaled dot-product attention on lists. `attention(Q, K, V, causal)` scores each
query against each key with dot(q, k) / sqrt(d_k), masks future positions to -infinity when
`causal` is True, softmaxes each row into weights, and returns `(output, weights)` where
output = weights @ V. `self_attention(X, Wq, Wk, Wv, causal)` projects one sequence into
queries, keys and values, then calls `attention`.

## Why it matters
This function, repeated and stacked, is the transformer. The causal mask is why a chat
model cannot peek at the words it has not written yet, and the square-root scaling is why
training does not stall with saturated softmaxes. Returning the weights lets you see what
each position looked at, which is how people debug attention in practice.

## Hints
- `kit.linalg` (matmul, transpose, dot) and `kit.mathx.softmax` are unlocked; use them.
- d_k is the length of a key vector, len(K[0]).
- With the causal mask, position i may attend to positions 0..i. Set scores[i][j] = -math.inf for j > i; a stable softmax turns those into exact zeros.
- Output row i is the weighted average of the value rows, using weights row i.
"""
from __future__ import annotations

import math

from kit.linalg import dot, matmul
from kit.mathx import softmax


def attention(Q, K, V, causal=False):
    """Scaled dot-product attention. Returns (output, weights)."""
    # >>> fullsend: scores, optional mask, row softmax, weighted sum of values
    if not Q or not K or len(K) != len(V):
        raise ValueError("need queries, and one value row per key")
    scale = math.sqrt(len(K[0]))
    scores = [[dot(q, k) / scale for k in K] for q in Q]
    if causal:
        for i, row in enumerate(scores):
            for j in range(i + 1, len(row)):
                row[j] = -math.inf
    weights = [softmax(row) for row in scores]
    return matmul(weights, V), weights
    # <<< fullsend


def self_attention(X, Wq, Wk, Wv, causal=False):
    """X: sequence of token vectors (n x d). W*: projection matrices (d x d_head)."""
    # >>> fullsend: project X three ways, then attend
    return attention(matmul(X, Wq), matmul(X, Wk), matmul(X, Wv), causal)
    # <<< fullsend
