"""
Title: Honest Metrics
XP: 20

## The job
Write the scoreboard. `train_test_split(rows, test_frac, seed)` shuffles a copy with a
seeded `random.Random` and returns `(train, test)` with `round(len(rows) * test_frac)` rows
in test. `confusion` counts tp, fp, fn and tn. `accuracy` is the share of exact matches.
`precision_recall_f1` returns all three, using 0.0 whenever a ratio would divide by zero.

## Why it matters
A model that always says "not fraud" is 99 percent accurate when 1 percent of cases are
fraud, and completely useless. Precision and recall expose that. Seeded splits make
experiments repeatable, so a better number means a better model and not a luckier shuffle.

## Hints
- `random.Random(seed).shuffle(copy)` keeps the caller's list intact and the order reproducible.
- Precision asks "of everything I flagged, how much was right?" Recall asks "of everything I should have flagged, how much did I catch?"
- F1 is the harmonic mean, 2PR / (P + R), and is 0.0 when P + R is 0.
- Raise ValueError when y_true and y_pred have different lengths.
"""
from __future__ import annotations

import random


def train_test_split(rows, test_frac=0.2, seed=0):
    """Return (train, test) as new lists. Same seed, same split."""
    # >>> fullsend: shuffle a copy with a seeded RNG, slice off the test part
    if not 0.0 <= test_frac <= 1.0:
        raise ValueError("test_frac must be between 0 and 1")
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    n_test = round(len(shuffled) * test_frac)
    return shuffled[n_test:], shuffled[:n_test]
    # <<< fullsend


def _check(y_true, y_pred):
    if len(y_true) != len(y_pred):
        raise ValueError(f"length mismatch: {len(y_true)} vs {len(y_pred)}")


def confusion(y_true, y_pred, positive=1):
    """Return {'tp': .., 'fp': .., 'fn': .., 'tn': ..} for the given positive label."""
    # >>> fullsend: count the four outcomes
    _check(y_true, y_pred)
    counts = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    for t, p in zip(y_true, y_pred):
        if p == positive:
            counts["tp" if t == positive else "fp"] += 1
        else:
            counts["fn" if t == positive else "tn"] += 1
    return counts
    # <<< fullsend


def accuracy(y_true, y_pred):
    # >>> fullsend: share of exact matches
    _check(y_true, y_pred)
    if not y_true:
        return 0.0
    return sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)
    # <<< fullsend


def precision_recall_f1(y_true, y_pred, positive=1):
    """Return (precision, recall, f1). Any 0/0 becomes 0.0."""
    # >>> fullsend: build on confusion()
    c = confusion(y_true, y_pred, positive)
    precision = c["tp"] / (c["tp"] + c["fp"]) if c["tp"] + c["fp"] else 0.0
    recall = c["tp"] / (c["tp"] + c["fn"]) if c["tp"] + c["fn"] else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1
    # <<< fullsend
