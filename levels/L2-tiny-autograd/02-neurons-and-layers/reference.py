"""
Title: Neurons and Layers
XP: 25

## The job
Stack `Value`s into a network. A `Neuron` computes tanh(w . x + b) (or just w . x + b when
`nonlin=False`). A `Layer` is a list of neurons that all see the same inputs. An `MLP`
chains layers: tanh in the hidden layers, a linear last layer, a single `Value` out when
the last layer has one neuron. Every piece reports its `parameters()`.

## Why it matters
"Count the parameters" is the first sanity check on any architecture, and "the output
layer must be linear for regression" is a real bug people ship. Building the containers
yourself also explains what `model.parameters()` and `optimizer.zero_grad()` are for.

## Hints
- Start the weighted sum at the bias: `sum((w * x for ...), self.b)` keeps everything as Values.
- Raise ValueError when a neuron gets the wrong number of inputs. Silent zip truncation hides shape bugs.
- For layer sizes, pair up consecutive dimensions: [n_in] + sizes gives (n_in, s0), (s0, s1), ...
- Only the last layer is linear: nonlin is True for every layer index except the final one.
"""
from __future__ import annotations

from kit.autograd import Value


class Neuron:
    def __init__(self, n_in, rng, nonlin=True):
        self.w = [Value(rng.uniform(-1.0, 1.0)) for _ in range(n_in)]
        self.b = Value(0.0)
        self.nonlin = nonlin

    def __call__(self, xs):
        """xs: list of numbers or Values. Returns one Value."""
        # >>> fullsend: weighted sum plus bias, tanh if nonlin
        if len(xs) != len(self.w):
            raise ValueError(f"expected {len(self.w)} inputs, got {len(xs)}")
        act = sum((wi * xi for wi, xi in zip(self.w, xs)), self.b)
        return act.tanh() if self.nonlin else act
        # <<< fullsend

    def parameters(self):
        # >>> fullsend: weights then bias
        return self.w + [self.b]
        # <<< fullsend


class Layer:
    def __init__(self, n_in, n_out, rng, nonlin=True):
        self.neurons = [Neuron(n_in, rng, nonlin) for _ in range(n_out)]

    def __call__(self, xs):
        """Return a list with one output Value per neuron."""
        # >>> fullsend: every neuron sees the same inputs
        return [n(xs) for n in self.neurons]
        # <<< fullsend

    def parameters(self):
        # >>> fullsend: all neurons' parameters in one flat list
        return [p for n in self.neurons for p in n.parameters()]
        # <<< fullsend


class MLP:
    def __init__(self, n_in, sizes, rng):
        """sizes like [4, 4, 1]: two hidden layers of 4, one output. Last layer is linear."""
        # >>> fullsend: build one Layer per size, linear only at the end
        dims = [n_in] + list(sizes)
        self.layers = [
            Layer(dims[i], dims[i + 1], rng, nonlin=(i < len(sizes) - 1)) for i in range(len(sizes))
        ]
        # <<< fullsend

    def __call__(self, xs):
        """Feed forward. A single-output network returns one Value, otherwise a list."""
        # >>> fullsend: pass the outputs of each layer into the next
        out = xs
        for layer in self.layers:
            out = layer(out)
        return out[0] if len(out) == 1 else out
        # <<< fullsend

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def zero_grad(self):
        # >>> fullsend: set every parameter's grad to 0
        for p in self.parameters():
            p.grad = 0.0
        # <<< fullsend
