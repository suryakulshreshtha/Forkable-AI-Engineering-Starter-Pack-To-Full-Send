"""Neurons, layers and a multi-layer perceptron on kit.autograd. Unlocked by drill 2.2."""
from __future__ import annotations

from kit.autograd import Value


class Neuron:
    def __init__(self, n_in, rng, nonlin=True):
        self.w = [Value(rng.uniform(-1.0, 1.0)) for _ in range(n_in)]
        self.b = Value(0.0)
        self.nonlin = nonlin

    def __call__(self, xs):
        if len(xs) != len(self.w):
            raise ValueError(f"expected {len(self.w)} inputs, got {len(xs)}")
        act = sum((wi * xi for wi, xi in zip(self.w, xs)), self.b)
        return act.tanh() if self.nonlin else act

    def parameters(self):
        return self.w + [self.b]


class Layer:
    def __init__(self, n_in, n_out, rng, nonlin=True):
        self.neurons = [Neuron(n_in, rng, nonlin) for _ in range(n_out)]

    def __call__(self, xs):
        return [n(xs) for n in self.neurons]

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]


class MLP:
    def __init__(self, n_in, sizes, rng):
        dims = [n_in] + list(sizes)
        self.layers = [
            Layer(dims[i], dims[i + 1], rng, nonlin=(i < len(sizes) - 1)) for i in range(len(sizes))
        ]

    def __call__(self, xs):
        out = xs
        for layer in self.layers:
            out = layer(out)
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0
