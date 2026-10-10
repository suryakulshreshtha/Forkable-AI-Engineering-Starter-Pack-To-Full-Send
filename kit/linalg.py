"""Plain-list linear algebra. Unlocked by drill 1.1."""
from __future__ import annotations

import math


def dot(a, b):
    if len(a) != len(b):
        raise ValueError(f"length mismatch: {len(a)} vs {len(b)}")
    return sum(x * y for x, y in zip(a, b))


def matvec(M, v):
    return [dot(row, v) for row in M]


def transpose(M):
    if not M:
        return []
    return [list(col) for col in zip(*M)]


def matmul(A, B):
    if not A or not B:
        raise ValueError("empty matrix")
    if len(A[0]) != len(B):
        raise ValueError(f"shape mismatch: {len(A)}x{len(A[0])} @ {len(B)}x{len(B[0])}")
    cols = transpose(B)
    return [[dot(row, col) for col in cols] for row in A]


def add(a, b):
    if len(a) != len(b):
        raise ValueError("length mismatch")
    return [x + y for x, y in zip(a, b)]


def scale(a, k):
    return [k * x for x in a]


def norm(a):
    return math.sqrt(sum(x * x for x in a))
