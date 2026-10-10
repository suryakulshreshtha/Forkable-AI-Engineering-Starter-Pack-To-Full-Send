"""The judge for drill 2.2."""
import math
import random

from kit.autograd import Value


def _close(a, b, tol=1e-9):
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def test_neuron_is_tanh_of_weighted_sum(m):
    n = m.Neuron(3, random.Random(0))
    n.b.data = 0.25
    xs = [1.0, -2.0, 0.5]
    expected = math.tanh(sum(w.data * x for w, x in zip(n.w, xs)) + 0.25)
    out = n(xs)
    assert isinstance(out, Value), "a neuron should return a Value"
    assert _close(out.data, expected)


def test_linear_neuron_skips_tanh(m):
    n = m.Neuron(2, random.Random(1), nonlin=False)
    n.w[0].data, n.w[1].data, n.b.data = 3.0, -1.0, 2.0
    assert n([4.0, 5.0]).data == 9.0


def test_neuron_refuses_the_wrong_number_of_inputs(m):
    n = m.Neuron(3, random.Random(0))
    try:
        n([1.0, 2.0])
    except ValueError:
        return
    raise AssertionError("a 3-input neuron given 2 inputs should raise ValueError")


def test_layer_outputs_and_parameter_count(m):
    layer = m.Layer(3, 5, random.Random(2))
    out = layer([0.1, 0.2, 0.3])
    assert isinstance(out, list) and len(out) == 5
    assert len(layer.parameters()) == 5 * (3 + 1)


def test_mlp_parameter_count(m):
    net = m.MLP(3, [4, 4, 1], random.Random(3))
    assert len(net.parameters()) == 41, "(3+1)*4 + (4+1)*4 + (4+1)*1 = 41"


def test_mlp_output_shape(m):
    one = m.MLP(2, [3, 1], random.Random(4))([0.5, -0.5])
    many = m.MLP(2, [3, 2], random.Random(4))([0.5, -0.5])
    assert isinstance(one, Value), "one output neuron -> a single Value"
    assert isinstance(many, list) and len(many) == 2


def test_only_the_last_layer_is_linear(m):
    net = m.MLP(2, [3, 3, 1], random.Random(5))
    flags = [layer.neurons[0].nonlin for layer in net.layers]
    assert flags == [True, True, False], f"expected [True, True, False], got {flags}"


def test_gradients_reach_every_layer(m):
    net = m.MLP(3, [4, 1], random.Random(6))
    out = net([0.3, -0.7, 0.9])
    out.backward()
    nonzero = sum(1 for p in net.parameters() if p.grad != 0.0)
    assert nonzero >= len(net.parameters()) - 1, "gradients should reach (nearly) every parameter"


def test_zero_grad_clears_everything(m):
    net = m.MLP(2, [3, 1], random.Random(7))
    net([1.0, 2.0]).backward()
    net.zero_grad()
    assert all(p.grad == 0.0 for p in net.parameters())


def test_same_seed_same_network(m):
    a = m.MLP(2, [4, 1], random.Random(42))([0.2, 0.4]).data
    b = m.MLP(2, [4, 1], random.Random(42))([0.2, 0.4]).data
    assert a == b
