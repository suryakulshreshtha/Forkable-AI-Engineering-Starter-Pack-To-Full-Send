"""
Title: Scalar Graph
XP: 30

## The job
Finish `Value`, a number that records the operations that produced it. Addition and
multiplication already build the graph; you write how gradients flow back through
`+`, `*`, `**`, `tanh`, `exp` and `relu`, then `topo_order()` (every node once, parents
before children) and `backward()` (zero the graph's grads, seed the output with 1, run
each node's local rule from the output back to the inputs).

## Why it matters
PyTorch, JAX and every other framework do exactly this, only on tensors and in C++.
Once you have written one, "gradient is None", "gradients doubled" and "loss.backward()
twice" stop being mysteries, because you have caused each of them yourself.

## Hints
- Each local rule asks: if this input wiggles by a tiny amount, how much does `out` wiggle? Multiply that by `out.grad`.
- Use `+=`, never `=`. A value used twice (like `x * x`) receives gradient from both uses.
- For `tanh`, the slope is 1 - tanh(x)^2, and you already computed tanh(x) on the way forward.
- `topo_order` is a depth-first search: visit the parents first, then append the node. Track visited nodes with `id(node)`.
- `backward` runs the `_backward` closures in reverse topological order, after setting the output's grad to 1.0.
"""
from __future__ import annotations

import math


class Value:
    """A scalar that remembers its parents and how to pass gradients to them."""

    def __init__(self, data, parents=(), op=""):
        self.data = float(data)
        self.grad = 0.0
        self._backward = lambda: None
        self._parents = tuple(parents)
        self.op = op

    def __repr__(self):
        return f"Value(data={self.data:.6g}, grad={self.grad:.6g})"

    @staticmethod
    def lift(x):
        return x if isinstance(x, Value) else Value(x)

    def __add__(self, other):
        other = Value.lift(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            # >>> fullsend: both inputs pass out.grad straight through
            self.grad += out.grad
            other.grad += out.grad
            # <<< fullsend

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = Value.lift(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            # >>> fullsend: each input's slope is the other input's value
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
            # <<< fullsend

        out._backward = _backward
        return out

    def __pow__(self, k):
        if not isinstance(k, (int, float)):
            raise TypeError("only constant exponents are supported")
        out = Value(self.data ** k, (self,), f"**{k}")

        def _backward():
            # >>> fullsend: power rule, k * x^(k-1)
            self.grad += k * self.data ** (k - 1) * out.grad
            # <<< fullsend

        out._backward = _backward
        return out

    def tanh(self):
        # >>> fullsend: forward with math.tanh, backward with 1 - t^2
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")

        def _backward():
            self.grad += (1.0 - t * t) * out.grad

        out._backward = _backward
        return out
        # <<< fullsend

    def exp(self):
        # >>> fullsend: e^x is its own slope
        e = math.exp(self.data)
        out = Value(e, (self,), "exp")

        def _backward():
            self.grad += e * out.grad

        out._backward = _backward
        return out
        # <<< fullsend

    def relu(self):
        # >>> fullsend: pass x when positive, else 0; slope is 1 or 0
        out = Value(self.data if self.data > 0 else 0.0, (self,), "relu")

        def _backward():
            self.grad += (1.0 if self.data > 0 else 0.0) * out.grad

        out._backward = _backward
        return out
        # <<< fullsend

    # The remaining operators are built from the ones above, so they get gradients for free.
    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        return self + (-Value.lift(other))

    def __truediv__(self, other):
        return self * Value.lift(other) ** -1

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __rsub__(self, other):
        return Value.lift(other) - self

    def __rtruediv__(self, other):
        return Value.lift(other) / self

    def topo_order(self):
        """Every node reachable from self, each once, parents before children, self last."""
        # >>> fullsend: depth-first search, append a node after its parents
        order, seen = [], set()

        def visit(node):
            if id(node) in seen:
                return
            seen.add(id(node))
            for parent in node._parents:
                visit(parent)
            order.append(node)

        visit(self)
        return order
        # <<< fullsend

    def backward(self):
        """Zero every grad in the graph, set self.grad = 1, then apply local rules in reverse order."""
        # >>> fullsend: reset, seed, walk backwards
        order = self.topo_order()
        for node in order:
            node.grad = 0.0
        self.grad = 1.0
        for node in reversed(order):
            node._backward()
        # <<< fullsend
