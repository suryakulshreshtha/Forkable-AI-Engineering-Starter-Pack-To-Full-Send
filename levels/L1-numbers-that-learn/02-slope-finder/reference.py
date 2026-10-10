"""
Title: Slope Finder
XP: 15

## The job
Measure slopes numerically. `derivative(f, x)` estimates f'(x) with a central difference.
`gradient(f, point)` returns every partial derivative of a function of several variables.
`check_gradient(f, grad_f, point)` compares a hand-written gradient against your numeric
one and returns True only when every component agrees.

## Why it matters
Gradient checking is how people debug backpropagation code: when the analytic gradient and
the brute-force one disagree, the analytic one is almost always the liar. You will use this
exact trick to test your autograd engine in level 2.

## Hints
- Central difference: (f(x + h) - f(x - h)) / (2h). It is far more accurate than the one-sided version for the same h.
- For a partial derivative, nudge one coordinate up and down while holding the others still. Copy the point list before you change it.
- Compare with relative error, |a - n| / max(1e-8, |a| + |n|), so big and small gradients are judged fairly.
"""
from __future__ import annotations


def derivative(f, x, h=1e-5):
    """Central-difference estimate of f'(x)."""
    # >>> fullsend: return (f(x + h) - f(x - h)) / (2h)
    return (f(x + h) - f(x - h)) / (2 * h)
    # <<< fullsend


def gradient(f, point, h=1e-5):
    """Numeric gradient of f at point (a list). f takes a list and returns a number."""
    # >>> fullsend: one central difference per coordinate
    grads = []
    for i in range(len(point)):
        up, down = list(point), list(point)
        up[i] += h
        down[i] -= h
        grads.append((f(up) - f(down)) / (2 * h))
    return grads
    # <<< fullsend


def check_gradient(f, grad_f, point, tol=1e-4):
    """True when grad_f(point) matches the numeric gradient in every component."""
    # >>> fullsend: compare analytic and numeric gradients by relative error
    analytic = grad_f(point)
    numeric = gradient(f, point)
    if len(analytic) != len(numeric):
        return False
    for a, n in zip(analytic, numeric):
        if abs(a - n) / max(1e-8, abs(a) + abs(n)) > tol:
            return False
    return True
    # <<< fullsend
