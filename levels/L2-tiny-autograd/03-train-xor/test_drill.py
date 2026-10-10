"""The judge for drill 2.3."""
import random

from kit.autograd import Value
from kit.nn import MLP

XOR_X = [[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]]
XOR_Y = [-1.0, 1.0, 1.0, -1.0]


def test_mse_by_hand(m):
    loss = m.mse([Value(1.0), Value(3.0)], [0.0, 1.0])
    assert isinstance(loss, Value), "mse must return a Value so it can backpropagate"
    assert abs(loss.data - 2.5) < 1e-12, "((1-0)^2 + (3-1)^2) / 2 = 2.5"


def test_mse_gradient_points_the_right_way(m):
    p = Value(3.0)
    m.mse([p], [1.0]).backward()
    assert p.grad == 4.0, "d/dp (p - 1)^2 at p = 3 is 4"


def test_sgd_step_moves_against_the_gradient(m):
    a, b = Value(1.0), Value(-2.0)
    a.grad, b.grad = 2.0, -4.0
    m.sgd_step([a, b], lr=0.1)
    assert abs(a.data - 0.8) < 1e-12 and abs(b.data - (-1.6)) < 1e-12


def test_train_returns_one_float_per_epoch(m):
    net = MLP(2, [3, 1], random.Random(0))
    history = m.train(net, XOR_X, XOR_Y, lr=0.05, epochs=7)
    assert len(history) == 7 and all(isinstance(h, float) for h in history)


def test_network_learns_xor(m):
    net = MLP(2, [4, 1], random.Random(1337))
    history = m.train(net, XOR_X, XOR_Y, lr=0.1, epochs=200)
    assert history[-1] < history[0] / 10, f"loss should fall a lot: {history[0]:.3f} -> {history[-1]:.3f}"
    assert history[-1] < 0.01, f"final loss {history[-1]:.4f} should be below 0.01"
    preds = [m.predict_sign(net, x) for x in XOR_X]
    assert preds == [-1, 1, 1, -1], f"predictions {preds}"


def test_predict_sign_thresholds_at_zero(m):
    class Fixed:
        def __init__(self, v):
            self.v = v

        def __call__(self, x):
            return Value(self.v)

    assert m.predict_sign(Fixed(0.3), [0]) == 1
    assert m.predict_sign(Fixed(-0.3), [0]) == -1
    assert m.predict_sign(Fixed(0.0), [0]) == 1
