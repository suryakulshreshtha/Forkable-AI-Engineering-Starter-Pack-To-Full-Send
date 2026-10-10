"""The judge for drill 1.3."""
import math


def bowl_grad(p):
    """Gradient of (x - 3)^2 + (y + 1)^2, whose bottom is at (3, -1)."""
    return [2 * (p[0] - 3), 2 * (p[1] + 1)]


def ravine_grad(p):
    """Gradient of 0.5 * (x^2 + 25 y^2): steep in y, shallow in x."""
    return [p[0], 25 * p[1]]


def _dist(p, q):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p, q)))


def test_trajectory_includes_start_and_every_step(m):
    path = m.gradient_descent(bowl_grad, [0.0, 0.0], lr=0.1, steps=5)
    assert len(path) == 6, "5 steps should give 6 points (start included)"
    assert path[0] == [0.0, 0.0]


def test_first_step_by_hand(m):
    path = m.gradient_descent(bowl_grad, [0.0, 0.0], lr=0.1, steps=1)
    # grad at origin is [-6, 2]; step = origin - 0.1 * grad = [0.6, -0.2]
    assert _dist(path[1], [0.6, -0.2]) < 1e-12, f"expected [0.6, -0.2], got {path[1]}"


def test_trajectory_points_are_independent_copies(m):
    path = m.gradient_descent(bowl_grad, [0.0, 0.0], lr=0.1, steps=3)
    assert len({id(p) for p in path}) == len(path), "every point in the path must be its own list"


def test_descent_reaches_the_bottom(m):
    path = m.gradient_descent(bowl_grad, [10.0, 10.0], lr=0.1, steps=200)
    assert _dist(path[-1], [3.0, -1.0]) < 1e-6


def test_momentum_wins_in_a_ravine(m):
    start, lr, steps = [10.0, 1.0], 0.01, 100
    plain = m.gradient_descent(ravine_grad, start, lr, steps)[-1]
    heavy = m.momentum_descent(ravine_grad, start, lr, 0.9, steps)[-1]
    assert _dist(heavy, [0, 0]) < _dist(plain, [0, 0]) / 5, (
        "with the same learning rate, momentum should end much closer to the bottom"
    )


def test_convergence_step_count(m):
    n = m.steps_to_converge(bowl_grad, [10.0, 10.0], lr=0.1, tol=1e-6)
    assert n is not None and 60 <= n <= 90, f"expected about 75 steps, got {n}"


def test_too_large_a_learning_rate_diverges(m):
    assert m.steps_to_converge(bowl_grad, [10.0, 10.0], lr=1.1) is None


def test_already_at_the_bottom_takes_zero_steps(m):
    assert m.steps_to_converge(bowl_grad, [3.0, -1.0], lr=0.1) == 0
