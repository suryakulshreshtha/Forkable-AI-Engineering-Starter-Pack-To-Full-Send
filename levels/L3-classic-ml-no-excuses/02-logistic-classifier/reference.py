"""
Title: Logistic Classifier
XP: 20

## The job
Build a binary classifier. `sigmoid(z)` must survive z = -1000 and z = 1000 without an
overflow. `bce(p, y)` is binary cross-entropy with p clipped away from 0 and 1. The
`LogisticRegression` class learns weights and a bias by gradient descent in `fit`, then
offers `predict_proba`, `predict` (with a threshold) and `loss`.

## Why it matters
This is the output layer of every yes/no model, from spam filters to the reward models used
to tune chatbots. The overflow-safe sigmoid and the clipped log are the same numerical care
that keeps large models from producing NaN losses at 3 a.m.

## Hints
- Stable sigmoid: for z >= 0 use 1 / (1 + e^-z); for z < 0 use e^z / (1 + e^z). Neither branch ever exponentiates a large positive number.
- Clip p into [eps, 1 - eps] before taking logs.
- The gradient of mean BCE with respect to weight j is mean((p - y) * x_j). The bias gets mean(p - y).
- `fit` should return self so you can write `LogisticRegression().fit(X, y).predict(X)`.
"""
from __future__ import annotations

import math


def sigmoid(z):
    """1 / (1 + e^-z), computed without overflow for any finite z."""
    # >>> fullsend: branch on the sign of z
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)
    # <<< fullsend


def bce(p, y, eps=1e-12):
    """Binary cross-entropy for one prediction p in (0, 1) and label y in {0, 1}."""
    # >>> fullsend: clip, then -(y log p + (1 - y) log(1 - p))
    p = min(max(p, eps), 1.0 - eps)
    return -(y * math.log(p) + (1 - y) * math.log(1.0 - p))
    # <<< fullsend


class LogisticRegression:
    def __init__(self, lr=0.1, epochs=500):
        self.lr, self.epochs = lr, epochs
        self.w, self.b = None, 0.0

    def _z(self, row):
        return sum(wj * xj for wj, xj in zip(self.w, row)) + self.b

    def fit(self, X, y):
        """Gradient descent on mean BCE, starting from zero weights. Returns self."""
        # >>> fullsend: zero init, then repeat: probabilities, errors, step
        n, d = len(X), len(X[0])
        self.w, self.b = [0.0] * d, 0.0
        for _ in range(self.epochs):
            errors = [sigmoid(self._z(row)) - t for row, t in zip(X, y)]
            grad_w = [sum(e * row[j] for e, row in zip(errors, X)) / n for j in range(d)]
            grad_b = sum(errors) / n
            self.w = [wj - self.lr * gj for wj, gj in zip(self.w, grad_w)]
            self.b -= self.lr * grad_b
        return self
        # <<< fullsend

    def predict_proba(self, X):
        # >>> fullsend: sigmoid of the weighted sum for each row
        if self.w is None:
            raise RuntimeError("call fit() first")
        return [sigmoid(self._z(row)) for row in X]
        # <<< fullsend

    def predict(self, X, threshold=0.5):
        # >>> fullsend: 1 when the probability reaches the threshold
        return [1 if p >= threshold else 0 for p in self.predict_proba(X)]
        # <<< fullsend

    def loss(self, X, y):
        return sum(bce(p, t) for p, t in zip(self.predict_proba(X), y)) / len(y)
