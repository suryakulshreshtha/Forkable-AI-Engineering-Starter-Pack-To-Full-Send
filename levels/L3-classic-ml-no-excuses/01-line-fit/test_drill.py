"""The judge for drill 3.1."""
import random


def _close(a, b, tol=1e-9):
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def test_recovers_a_perfect_line(m):
    xs = [0, 1, 2, 3, 4]
    ys = [3 * x - 2 for x in xs]
    slope, intercept = m.fit_line(xs, ys)
    assert _close(slope, 3) and _close(intercept, -2), f"got slope={slope}, intercept={intercept}"


def test_least_squares_on_noisy_points(m):
    slope, intercept = m.fit_line([1, 2, 3, 4], [2, 3, 5, 6])
    assert _close(slope, 1.4) and _close(intercept, 0.5), f"got {slope}, {intercept}"


def test_identical_x_values_raise(m):
    try:
        m.fit_line([2, 2, 2], [1, 2, 3])
    except ValueError:
        return
    raise AssertionError("a vertical cloud of points has no least-squares slope")


def test_r_squared_perfect_and_partial(m):
    xs, ys = [0, 1, 2, 3], [1, 3, 5, 7]
    assert _close(m.r_squared(xs, ys, 2, 1), 1.0)
    xs, ys = [1, 2, 3, 4], [2, 3, 5, 6]
    slope, intercept = m.fit_line(xs, ys)
    assert _close(m.r_squared(xs, ys, slope, intercept), 0.98, 1e-9)


def test_r_squared_of_a_terrible_line_is_negative(m):
    assert m.r_squared([0, 1, 2], [0, 1, 2], -5, 0) < 0, "worse than predicting the mean -> below zero"


def test_predict_applies_weights_and_bias(m):
    assert m.predict([[1, 2], [0, -1]], [3, 4], 0.5) == [11.5, -3.5]


def test_gradient_descent_recovers_known_weights(m):
    rng = random.Random(7)
    X = [[rng.uniform(-1, 1), rng.uniform(-1, 1)] for _ in range(40)]
    y = [2.0 * a - 3.0 * b + 0.5 for a, b in X]
    w, b = m.fit_gd(X, y, lr=0.1, epochs=1500)
    assert abs(w[0] - 2.0) < 1e-3 and abs(w[1] + 3.0) < 1e-3 and abs(b - 0.5) < 1e-3, f"got w={w}, b={b}"


def test_gradient_descent_agrees_with_the_closed_form(m):
    xs = [0.0, 0.5, 1.0, 1.5, 2.0]
    ys = [1.1, 1.9, 3.2, 3.9, 5.1]
    slope, intercept = m.fit_line(xs, ys)
    w, b = m.fit_gd([[x] for x in xs], ys, lr=0.1, epochs=3000)
    assert abs(w[0] - slope) < 1e-4 and abs(b - intercept) < 1e-4
