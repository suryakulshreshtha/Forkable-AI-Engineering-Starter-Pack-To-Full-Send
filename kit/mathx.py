"""Numerically careful softmax family. Unlocked by drill 5.1."""
from __future__ import annotations

import math


def logsumexp(xs):
    if not xs:
        raise ValueError("empty input")
    top = max(xs)
    if top == -math.inf:
        return -math.inf
    return top + math.log(sum(math.exp(x - top) for x in xs))


def softmax(xs, temperature=1.0):
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    scaled = [x / temperature for x in xs]
    top = max(scaled)
    exps = [math.exp(x - top) for x in scaled]
    total = sum(exps)
    return [e / total for e in exps]


def log_softmax(xs):
    lse = logsumexp(xs)
    return [x - lse for x in xs]


def sigmoid(z):
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)
