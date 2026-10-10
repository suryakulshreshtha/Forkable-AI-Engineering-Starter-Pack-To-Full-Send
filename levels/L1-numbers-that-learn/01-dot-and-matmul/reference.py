"""
Title: Dot, Matvec, Matmul
XP: 10

## The job
Write four functions on plain lists: `dot(a, b)`, `matvec(M, v)`, `transpose(M)` and
`matmul(A, B)`. A vector is a list of numbers; a matrix is a list of rows. Mismatched
shapes raise `ValueError` instead of quietly producing garbage.

## Why it matters
A dense layer in a neural network is `matvec(weights, inputs)` plus a bias. Attention is
two matmuls and a softmax. Most "the model outputs nonsense" bugs you will ever chase are
shape bugs, so the habit of refusing bad shapes loudly starts here.

## Hints
- `zip(a, b)` walks two lists in lockstep. Check the lengths first, because zip silently stops at the shorter one.
- `matvec` is one `dot` per row of the matrix.
- `transpose` turns columns into rows: `zip(*M)` hands you the columns as tuples.
- `matmul(A, B)[i][j]` is the dot of row i of A with column j of B. Transposing B first makes the columns easy to reach.
"""
from __future__ import annotations


def dot(a, b):
    """Sum of pairwise products. Raise ValueError if lengths differ."""
    # >>> fullsend: check lengths, then multiply pairwise and sum
    if len(a) != len(b):
        raise ValueError(f"length mismatch: {len(a)} vs {len(b)}")
    return sum(x * y for x, y in zip(a, b))
    # <<< fullsend


def matvec(M, v):
    """Matrix (list of rows) times vector: one dot product per row."""
    # >>> fullsend: one dot per row
    return [dot(row, v) for row in M]
    # <<< fullsend


def transpose(M):
    """Rows become columns. transpose([]) is []."""
    # >>> fullsend: flip rows and columns
    if not M:
        return []
    return [list(col) for col in zip(*M)]
    # <<< fullsend


def matmul(A, B):
    """(n x k) @ (k x m) -> (n x m). Raise ValueError when the inner sizes differ."""
    # >>> fullsend: check the inner dimension, then dot rows of A with columns of B
    if not A or not B:
        raise ValueError("empty matrix")
    if len(A[0]) != len(B):
        raise ValueError(f"shape mismatch: {len(A)}x{len(A[0])} @ {len(B)}x{len(B[0])}")
    cols = transpose(B)
    return [[dot(row, col) for col in cols] for row in A]
    # <<< fullsend
