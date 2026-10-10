"""
Title: Downhill
XP: 20

## The job
Implement the walk that trains every model. `gradient_descent(grad, start, lr, steps)`
returns the whole trajectory, start included. `momentum_descent` adds a velocity that
remembers past gradients: `v = beta * v + g`, then `x = x - lr * v`. `steps_to_converge`
counts plain descent steps until the gradient's length drops below `tol`, and returns
`None` if the walk blows up or runs out of steps.

## Why it matters
Learning rate is the first knob you touch on every training run, and "loss went to NaN"
is the first failure you meet. Watching a walk converge, crawl or explode on a bowl you
understand is how you build intuition for what an optimizer does on a bowl you don't.

## Hints
- `grad(point)` returns a list the same length as the point. Update every coordinate with the same learning rate.
- Store copies in the trajectory. Appending the same list object and then mutating it is a classic bug.
- Momentum keeps one velocity number per coordinate, starting at zero.
- Divergence check: stop and return None as soon as any coordinate is not finite or the gradient's length passes 1e12.
"""
from __future__ import annotations

import math


def gradient_descent(grad, start, lr, steps):
    """Return [start, p1, ..., p_steps], each a new list."""
    # >>> fullsend: step against the gradient, recording every point
    point = list(start)
    path = [list(point)]
    for _ in range(steps):
        g = grad(point)
        point = [p - lr * gi for p, gi in zip(point, g)]
        path.append(list(point))
    return path
    # <<< fullsend


def momentum_descent(grad, start, lr, beta, steps):
    """Heavy-ball momentum: v = beta*v + g; x = x - lr*v. Return the trajectory."""
    # >>> fullsend: keep a velocity per coordinate and step with it
    point = list(start)
    velocity = [0.0] * len(point)
    path = [list(point)]
    for _ in range(steps):
        g = grad(point)
        velocity = [beta * v + gi for v, gi in zip(velocity, g)]
        point = [p - lr * v for p, v in zip(point, velocity)]
        path.append(list(point))
    return path
    # <<< fullsend


def steps_to_converge(grad, start, lr, tol=1e-6, max_steps=10_000):
    """Plain descent steps until |grad| < tol. None if it diverges or never gets there."""
    # >>> fullsend: walk, measure the gradient's length, bail out on divergence
    point = list(start)
    for step in range(max_steps + 1):
        g = grad(point)
        size = math.sqrt(sum(gi * gi for gi in g))
        if not all(math.isfinite(v) for v in point) or not math.isfinite(size) or size > 1e12:
            return None
        if size < tol:
            return step
        point = [p - lr * gi for p, gi in zip(point, g)]
    return None
    # <<< fullsend
