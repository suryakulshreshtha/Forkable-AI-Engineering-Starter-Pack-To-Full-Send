"""The judge for drill 5.1."""
import math


def _close_lists(a, b, tol=1e-12):
    return len(a) == len(b) and all(abs(x - y) <= tol for x, y in zip(a, b))


def test_softmax_sums_to_one_and_keeps_order(m):
    p = m.softmax([1.0, 2.0, 3.0])
    assert abs(sum(p) - 1.0) < 1e-12
    assert p[0] < p[1] < p[2]
    assert abs(p[2] - math.exp(3) / (math.exp(1) + math.exp(2) + math.exp(3))) < 1e-12


def test_softmax_survives_huge_scores(m):
    big = m.softmax([1000.0, 1001.0, 1002.0])
    small = m.softmax([0.0, 1.0, 2.0])
    assert _close_lists(big, small), "shifting every score by a constant must not change the result"


def test_temperature_sharpens_and_flattens(m):
    xs = [1.0, 2.0, 4.0]
    cold, normal, hot = m.softmax(xs, 0.25), m.softmax(xs), m.softmax(xs, 100.0)
    assert cold[2] > normal[2] > hot[2]
    assert max(hot) - min(hot) < 0.02, "a very high temperature is nearly uniform"


def test_non_positive_temperature_is_rejected(m):
    for t in (0.0, -1.0):
        try:
            m.softmax([1.0, 2.0], t)
        except ValueError:
            continue
        raise AssertionError(f"temperature={t} should raise ValueError")


def test_logsumexp(m):
    assert abs(m.logsumexp([1000.0, 1000.0]) - (1000.0 + math.log(2))) < 1e-9
    assert abs(m.logsumexp([0.0, 0.0, 0.0]) - math.log(3)) < 1e-12


def test_log_softmax_stays_finite_where_log_of_softmax_fails(m):
    ls = m.log_softmax([-1000.0, 0.0])
    assert all(math.isfinite(v) for v in ls), "log_softmax must not produce -inf here"
    assert abs(ls[0] + 1000.0) < 1e-9 and abs(ls[1]) < 1e-12


def test_log_softmax_matches_log_of_softmax_on_tame_inputs(m):
    xs = [0.3, -1.2, 2.5]
    assert _close_lists(m.log_softmax(xs), [math.log(p) for p in m.softmax(xs)], 1e-12)


def test_cross_entropy(m):
    assert abs(m.cross_entropy([0.0, 0.0, 0.0, 0.0], 2) - math.log(4)) < 1e-12, "uniform over 4 -> log 4"
    assert m.cross_entropy([10.0, 0.0], 0) < 1e-4, "confident and right -> near zero"
    assert m.cross_entropy([10.0, 0.0], 1) > 9.9, "confident and wrong -> large"
