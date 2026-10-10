"""
Title: BPE Tokenizer
XP: 30

## The job
Build byte-pair encoding on characters. `pair_counts(tokens)` counts adjacent pairs.
`merge(tokens, pair)` fuses every left-to-right, non-overlapping occurrence of the pair.
`train_bpe(text, num_merges)` repeatedly merges the most frequent pair (ties go to the pair
seen first) and stops early when no pair occurs at least twice. `encode` replays the learned
merges in order on new text; `decode` joins tokens back into a string.

## Why it matters
Every modern language model starts here: the vocabulary is learned, not hand-written, which
is why a model can count letters badly, why some languages cost more tokens than others, and
why "token count" decides your bill. Round-tripping text exactly is a hard requirement:
a tokenizer that loses one space corrupts everything downstream.

## Hints
- Build the counts by walking left to right with a `collections.Counter`. Counters remember insertion order, and `max(counts, key=counts.get)` returns the first of several equal maxima.
- In `merge`, scan with an index. When tokens[i] and tokens[i + 1] match the pair, emit the joined string and jump ahead by two; otherwise emit tokens[i] and move by one.
- `encode` starts from `list(text)` and applies each merge in the order it was learned.
- Merges are pairs of strings. The new token is simply `a + b`.
"""
from __future__ import annotations

from collections import Counter


def pair_counts(tokens):
    """Counter of adjacent (left, right) pairs, in order of first appearance."""
    # >>> fullsend: count every neighbouring pair
    return Counter(zip(tokens, tokens[1:]))
    # <<< fullsend


def merge(tokens, pair):
    """Replace each non-overlapping occurrence of pair (scanning left to right) with a + b."""
    # >>> fullsend: index scan, jump two on a match
    a, b = pair
    out, i = [], 0
    while i < len(tokens):
        if i + 1 < len(tokens) and tokens[i] == a and tokens[i + 1] == b:
            out.append(a + b)
            i += 2
        else:
            out.append(tokens[i])
            i += 1
    return out
    # <<< fullsend


def train_bpe(text, num_merges):
    """Learn up to num_merges merges from text. Returns a list of (a, b) pairs."""
    # >>> fullsend: count, pick the best pair, stop below 2, merge, repeat
    tokens = list(text)
    merges = []
    for _ in range(num_merges):
        counts = pair_counts(tokens)
        if not counts:
            break
        best = max(counts, key=counts.get)
        if counts[best] < 2:
            break
        merges.append(best)
        tokens = merge(tokens, best)
    return merges
    # <<< fullsend


def encode(text, merges):
    """Characters first, then every learned merge in order."""
    # >>> fullsend: replay merges
    tokens = list(text)
    for pair in merges:
        tokens = merge(tokens, pair)
    return tokens
    # <<< fullsend


def decode(tokens):
    # >>> fullsend: glue tokens back together
    return "".join(tokens)
    # <<< fullsend
