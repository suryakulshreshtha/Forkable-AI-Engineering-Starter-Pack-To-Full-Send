"""
Title: Line Fit
XP: 15

## The job
Fit lines two ways. `fit_line(xs, ys)` uses the closed-form least-squares answer and returns
`(slope, intercept)`. `r_squared` says how much of the variation the line explains.
`fit_gd(X, y, lr, epochs)` learns several weights plus a bias by gradient descent on mean
squared error, starting from zeros. `predict(X, w, b)` applies them.

## Why it matters
Linear regression is the baseline you should run before any neural network, and comparing
the closed form against gradient descent shows that training is just one way of solving an
equation. R squared is also your first lesson in "a single number can mislead": it says
nothing about whether the relationship was linear to begin with.

## Hints
- slope = sum((x - mean_x) * (y - mean_y)) / sum((x - mean_x)^2); the intercept puts the line through the two means.
- If every x is the same, the slope is undefined: raise ValueError.
- R squared = 1 - SS_residual / SS_total. Guard against SS_total being zero.
- For gradient descent on MSE, the gradient for weight j is (2/n) * sum(error * x_j) and for the bias (2/n) * sum(error), where error = prediction - target.
"""
from __future__ import annotations


def fit_line(xs, ys):
    """Closed-form least squares for one feature. Returns (slope, intercept)."""
    # >>> fullsend: covariance over variance, then the intercept
    if len(xs) != len(ys) or len(xs) < 2:
        raise ValueError("need at least two (x, y) pairs of equal length")
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    var = sum((x - mx) ** 2 for x in xs)
    if var == 0:
        raise ValueError("all x values are identical; the slope is undefined")
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = cov / var
    return slope, my - slope * mx
    # <<< fullsend


def r_squared(xs, ys, slope, intercept):
    """1 - SS_res / SS_tot. A flat target that is predicted perfectly scores 1.0."""
    # >>> fullsend: compare residual error with total variation
    my = sum(ys) / len(ys)
    ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - my) ** 2 for y in ys)
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    return 1.0 - ss_res / ss_tot
    # <<< fullsend


def predict(X, w, b):
    """X is a list of rows. Returns w . row + b for each row."""
    # >>> fullsend: one weighted sum per row
    return [sum(wj * xj for wj, xj in zip(w, row)) + b for row in X]
    # <<< fullsend


def fit_gd(X, y, lr=0.05, epochs=1000):
    """Batch gradient descent on mean squared error. Start at zeros. Returns (w, b)."""
    # >>> fullsend: repeat: predict, measure errors, step every weight and the bias
    n, d = len(X), len(X[0])
    w, b = [0.0] * d, 0.0
    for _ in range(epochs):
        errors = [p - t for p, t in zip(predict(X, w, b), y)]
        grad_w = [2.0 / n * sum(e * row[j] for e, row in zip(errors, X)) for j in range(d)]
        grad_b = 2.0 / n * sum(errors)
        w = [wj - lr * gj for wj, gj in zip(w, grad_w)]
        b -= lr * grad_b
    return w, b
    # <<< fullsend
