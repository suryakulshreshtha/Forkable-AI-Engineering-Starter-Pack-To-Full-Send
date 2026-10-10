"""A scalar automatic-differentiation engine. Unlocked by drill 2.1."""
from __future__ import annotations

import math


class Value:
    """A number that remembers how it was made, so gradients can flow back."""

    __slots__ = ("data", "grad", "_backward", "_parents", "op")

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
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = Value.lift(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __pow__(self, k):
        if not isinstance(k, (int, float)):
            raise TypeError("only constant exponents are supported")
        out = Value(self.data ** k, (self,), f"**{k}")

        def _backward():
            self.grad += k * self.data ** (k - 1) * out.grad

        out._backward = _backward
        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")

        def _backward():
            self.grad += (1.0 - t * t) * out.grad

        out._backward = _backward
        return out

    def exp(self):
        e = math.exp(self.data)
        out = Value(e, (self,), "exp")

        def _backward():
            self.grad += e * out.grad

        out._backward = _backward
        return out

    def relu(self):
        out = Value(self.data if self.data > 0 else 0.0, (self,), "relu")

        def _backward():
            self.grad += (1.0 if self.data > 0 else 0.0) * out.grad

        out._backward = _backward
        return out

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

    def backward(self):
        order = self.topo_order()
        for node in order:
            node.grad = 0.0
        self.grad = 1.0
        for node in reversed(order):
            node._backward()
