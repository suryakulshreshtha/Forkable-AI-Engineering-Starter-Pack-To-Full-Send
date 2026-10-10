"""
Title: Softmax and Friends
XP: 15

## The job
Write the numerically careful softmax family. `softmax(xs, temperature)` turns scores into
probabilities without overflowing on scores like 1000. `logsumexp(xs)` computes
log(sum(exp(x))) safely. `log_softmax(xs)` returns log probabilities that stay finite
where `log(softmax(xs))` would hit log(0). `cross_entropy(logits, target)` is the loss
for the correct class index.

## Why it matters
Every classifier and every language model ends in exactly these functions, and the naive
versions break the moment a logit gets large. Temperature is also the first knob anyone
turns on a chat model, so it helps to know it is just a division before the softmax.

## Hints
- Subtracting the maximum from every score before exponentiating changes nothing mathematically and prevents overflow.
- logsumexp(xs) = m + log(sum(exp(x - m))) where m = max(xs).
- log_softmax(xs)[i] = xs[i] - logsumexp(xs). No exp of a big number, no log of zero.
- Cross-entropy for one example is minus the log probability of the target: use log_softmax, not log of softmax.
- Divide by the temperature first; a temperature of zero or below is a ValueError.
"""
from __future__ import annotations

import math


def softmax(xs, temperature=1.0):
    """Probabilities proportional to exp(x / temperature)."""
    # >>> fullsend: scale, shift by the max, exponentiate, normalise
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    scaled = [x / temperature for x in xs]
    top = max(scaled)
    exps = [math.exp(x - top) for x in scaled]
    total = sum(exps)
    return [e / total for e in exps]
    # <<< fullsend


def logsumexp(xs):
    """log(sum(exp(x) for x in xs)), computed without overflow."""
    # >>> fullsend: pull the max out of the sum
    if not xs:
        raise ValueError("empty input")
    top = max(xs)
    if top == -math.inf:
        return -math.inf
    return top + math.log(sum(math.exp(x - top) for x in xs))
    # <<< fullsend


def log_softmax(xs):
    """Log probabilities: xs[i] - logsumexp(xs)."""
    # >>> fullsend: subtract logsumexp from every score
    lse = logsumexp(xs)
    return [x - lse for x in xs]
    # <<< fullsend


def cross_entropy(logits, target):
    """Negative log probability of the target index."""
    # >>> fullsend: minus the target's log-softmax
    return -log_softmax(logits)[target]
    # <<< fullsend
