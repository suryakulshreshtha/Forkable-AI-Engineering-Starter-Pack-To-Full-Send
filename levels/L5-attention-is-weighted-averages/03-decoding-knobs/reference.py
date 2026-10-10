"""
Title: Decoding Knobs
XP: 25

## The job
Write the controls that turn next-token probabilities into an actual choice. `greedy`
picks the most likely index (ties go to the lowest index). `top_k_filter` keeps the k most
likely tokens and renormalises. `top_p_filter` keeps the smallest set of most-likely tokens
whose probabilities add up to at least p. `sample` draws an index with a seeded RNG.
`decode_step` chains them: softmax with temperature, optional top-k, optional top-p, sample.

## Why it matters
The same model can sound robotic, creative or unhinged depending only on these knobs. When
someone says "set temperature to 0 for evals" or "top-p 0.9 for chat", this is the code they
mean. Seeded sampling is also how you make a stochastic system reproducible enough to test.

## Hints
- Rank tokens with `sorted(range(n), key=lambda i: (-probs[i], i))` so ties are broken by index.
- Filters return a new list of the same length with zeros for dropped tokens, renormalised to sum to 1.
- For top-p, walk the ranked tokens adding probabilities and stop right after the running total reaches p.
- `sample`: draw r = rng.random(), then return the first index where the running total exceeds r. Skip zero-probability tokens so they can never be chosen.
- `kit.mathx.softmax` handles temperature for you in `decode_step`.
"""
from __future__ import annotations

from kit.mathx import softmax


def _ranked(probs):
    return sorted(range(len(probs)), key=lambda i: (-probs[i], i))


def _renormalise(probs, keep):
    total = sum(probs[i] for i in keep)
    return [probs[i] / total if i in keep else 0.0 for i in range(len(probs))]


def greedy(probs):
    """Index of the largest probability; ties go to the lowest index."""
    # >>> fullsend: the first item of the ranking
    return _ranked(probs)[0]
    # <<< fullsend


def top_k_filter(probs, k):
    """Keep the k most likely tokens, zero the rest, renormalise."""
    # >>> fullsend: rank, keep k, renormalise
    if k < 1:
        raise ValueError("k must be at least 1")
    keep = set(_ranked(probs)[:k])
    return _renormalise(probs, keep)
    # <<< fullsend


def top_p_filter(probs, p):
    """Keep the smallest top set whose mass reaches p (0 < p <= 1), renormalise."""
    # >>> fullsend: accumulate down the ranking until the mass reaches p
    if not 0 < p <= 1:
        raise ValueError("p must be in (0, 1]")
    keep, mass = set(), 0.0
    for i in _ranked(probs):
        keep.add(i)
        mass += probs[i]
        if mass >= p:
            break
    return _renormalise(probs, keep)
    # <<< fullsend


def sample(probs, rng):
    """Draw one index according to probs using rng.random()."""
    # >>> fullsend: inverse-CDF sampling that never returns a zero-probability index
    r = rng.random() * sum(probs)
    cumulative, last = 0.0, None
    for i, p in enumerate(probs):
        if p <= 0:
            continue
        cumulative += p
        last = i
        if r < cumulative:
            return i
    return last
    # <<< fullsend


def decode_step(logits, rng, temperature=1.0, top_k=None, top_p=None):
    """softmax(temperature) -> top_k -> top_p -> sample. Returns the chosen index."""
    # >>> fullsend: chain the knobs
    probs = softmax(logits, temperature)
    if top_k is not None:
        probs = top_k_filter(probs, top_k)
    if top_p is not None:
        probs = top_p_filter(probs, top_p)
    return sample(probs, rng)
    # <<< fullsend
