"""
Title: Train XOR
XP: 30

## The job
Write the training loop. `mse(preds, targets)` returns the mean squared error as a `Value`.
`sgd_step(params, lr)` nudges each parameter against its gradient. `train(model, xs, ys,
lr, epochs)` repeats forward, loss, zero_grad, backward, step, and returns the loss after
each epoch as plain floats. `predict_sign` turns a network output into +1 or -1.

## Why it matters
XOR cannot be solved by a single linear neuron, which is why it was a famous wall for early
networks. Watching your loop push a hidden layer through it is the smallest complete
example of deep learning: data, model, loss, gradients, update. Everything bigger is the
same loop with more numbers.

## Hints
- `mse`: sum the squared differences as Values, then divide by the count.
- `sgd_step` changes `p.data`, not `p.grad`, and moves against the gradient: data minus lr times grad.
- Call `model.zero_grad()` before `loss.backward()` every epoch, or gradients from old epochs pile up.
- Record `loss.data` (a float), not the Value itself, so the history does not keep every graph alive.
"""
from __future__ import annotations

from kit.autograd import Value


def mse(preds, targets):
    """Mean of (pred - target)^2. preds are Values, targets are numbers. Returns a Value."""
    # >>> fullsend: sum squared errors, divide by n
    if len(preds) != len(targets) or not preds:
        raise ValueError("preds and targets must be non-empty and the same length")
    total = Value(0.0)
    for p, t in zip(preds, targets):
        total = total + (p - t) ** 2
    return total / len(preds)
    # <<< fullsend


def sgd_step(params, lr):
    """Move every parameter a little against its gradient."""
    # >>> fullsend: data -= lr * grad
    for p in params:
        p.data -= lr * p.grad
    # <<< fullsend


def train(model, xs, ys, lr=0.1, epochs=100):
    """Full-batch training. Returns a list of float losses, one per epoch."""
    # >>> fullsend: forward, loss, zero_grad, backward, step; remember loss.data
    history = []
    for _ in range(epochs):
        preds = [model(x) for x in xs]
        loss = mse(preds, ys)
        model.zero_grad()
        loss.backward()
        sgd_step(model.parameters(), lr)
        history.append(loss.data)
    return history
    # <<< fullsend


def predict_sign(model, x):
    """+1 if the model's output is >= 0, else -1."""
    # >>> fullsend: threshold at zero
    return 1 if model(x).data >= 0 else -1
    # <<< fullsend
