"""The judge for drill 1.2."""
import math


def _close(a, b, tol=1e-5):
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def test_derivative_of_a_cube(m):
    got = m.derivative(lambda x: x ** 3, 2.0)
    assert _close(got, 12.0), f"d/dx x^3 at 2 is 12, got {got}"


def test_derivative_of_sine_at_zero(m):
    got = m.derivative(math.sin, 0.0)
    assert _close(got, 1.0), f"cos(0) is 1, got {got}"


def test_central_difference_beats_one_sided(m):
    f, x, h = math.exp, 1.0, 1e-3
    one_sided_error = abs((f(x + h) - f(x)) / h - math.e)
    central_error = abs(m.derivative(f, x, h=h) - math.e)
    assert central_error < one_sided_error / 100, "use the central difference, not (f(x+h)-f(x))/h"


def test_gradient_of_two_variables(m):
    f = lambda p: p[0] ** 2 * p[1] + p[1]      # df/dx = 2xy, df/dy = x^2 + 1
    g = m.gradient(f, [3.0, 2.0])
    assert len(g) == 2
    assert _close(g[0], 12.0) and _close(g[1], 10.0), f"expected [12, 10], got {g}"


def test_gradient_does_not_mutate_the_point(m):
    point = [1.0, 2.0, 3.0]
    m.gradient(lambda p: sum(v * v for v in p), point)
    assert point == [1.0, 2.0, 3.0], "gradient() changed the caller's list"


def test_check_gradient_accepts_a_correct_gradient(m):
    f = lambda p: math.sin(p[0]) * p[1] ** 2
    grad = lambda p: [math.cos(p[0]) * p[1] ** 2, 2 * math.sin(p[0]) * p[1]]
    assert m.check_gradient(f, grad, [0.7, -1.3]) is True


def test_check_gradient_catches_a_buggy_gradient(m):
    f = lambda p: math.sin(p[0]) * p[1] ** 2
    buggy = lambda p: [math.cos(p[0]) * p[1] ** 2, math.sin(p[0]) * p[1]]   # forgot the 2
    assert m.check_gradient(f, buggy, [0.7, -1.3]) is False
