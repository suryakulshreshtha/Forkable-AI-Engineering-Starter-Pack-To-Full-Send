"""The judge for drill 2.1."""
import math


def _close(a, b, tol=1e-6):
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def _numeric(f, xs, i, h=1e-6):
    up, down = list(xs), list(xs)
    up[i] += h
    down[i] -= h
    return (f(*up) - f(*down)) / (2 * h)


def test_forward_arithmetic(m):
    V = m.Value
    a, b, c = V(2.0), V(-3.0), V(10.0)
    assert (a * b + c).data == 4.0
    assert (a - b).data == 5.0 and (-a).data == -2.0
    assert _close((c / a).data, 5.0)
    assert (1 + a).data == 3.0 and (3 * a).data == 6.0


def test_sum_and_product_gradients(m):
    V = m.Value
    a, b, c = V(2.0), V(-3.0), V(10.0)
    out = a * b + c
    out.backward()
    assert (a.grad, b.grad, c.grad) == (-3.0, 2.0, 1.0), f"got {(a.grad, b.grad, c.grad)}"


def test_reused_values_accumulate_gradient(m):
    x = m.Value(3.0)
    y = x * x + x        # dy/dx = 2x + 1 = 7
    y.backward()
    assert x.grad == 7.0, f"x is used three times; expected 7.0, got {x.grad}"


def test_power_and_division(m):
    V = m.Value
    x = V(2.0)
    (x ** 3).backward()
    assert _close(x.grad, 12.0)
    a, b = V(6.0), V(4.0)
    (a / b).backward()
    assert _close(a.grad, 0.25) and _close(b.grad, -6.0 / 16.0)


def test_tanh_exp_relu_values_and_slopes(m):
    V = m.Value
    x = V(0.5)
    t = x.tanh()
    t.backward()
    assert _close(t.data, math.tanh(0.5)) and _close(x.grad, 1 - math.tanh(0.5) ** 2)
    y = V(1.2)
    e = y.exp()
    e.backward()
    assert _close(e.data, math.exp(1.2)) and _close(y.grad, math.exp(1.2))
    p, n = V(2.0), V(-2.0)
    rp, rn = p.relu(), n.relu()
    rp.backward()
    rn.backward()
    assert (rp.data, rn.data, p.grad, n.grad) == (2.0, 0.0, 1.0, 0.0)


def test_topo_order_lists_parents_first_and_each_node_once(m):
    V = m.Value
    a, b = V(1.0), V(2.0)
    c = a * b
    d = c + a
    order = d.topo_order()
    ids = [id(n) for n in order]
    assert len(ids) == len(set(ids)) == 4, "expect exactly a, b, c, d"
    assert order[-1] is d
    assert ids.index(id(a)) < ids.index(id(c)) < ids.index(id(d))
    assert ids.index(id(b)) < ids.index(id(c))


def test_matches_numeric_gradient_on_a_messy_expression(m):
    V = m.Value

    def f(x, y):
        return math.tanh(x * y + y ** 2) * math.exp(x / 3) - max(0.0, x - y)

    xs = [0.8, -0.4]
    x, y = V(xs[0]), V(xs[1])
    out = (x * y + y ** 2).tanh() * (x / 3).exp() - (x - y).relu()
    out.backward()
    assert _close(out.data, f(*xs))
    assert _close(x.grad, _numeric(f, xs, 0), 1e-5), f"dx: got {x.grad}, numeric {_numeric(f, xs, 0)}"
    assert _close(y.grad, _numeric(f, xs, 1), 1e-5), f"dy: got {y.grad}, numeric {_numeric(f, xs, 1)}"


def test_calling_backward_twice_gives_the_same_answer(m):
    x = m.Value(1.5)
    y = (x * x).tanh()
    y.backward()
    first = x.grad
    y.backward()
    assert _close(x.grad, first), "backward() must zero the graph's grads before accumulating"
