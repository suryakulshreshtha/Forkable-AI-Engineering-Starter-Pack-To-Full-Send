"""The judge for drill 5.2."""
import math


def _close(a, b, tol=1e-9):
    return abs(a - b) <= tol


def test_shapes_and_rows_sum_to_one(m):
    Q = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    K = [[1.0, 0.0], [0.0, 1.0]]
    V = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    out, w = m.attention(Q, K, V)
    assert len(out) == 3 and all(len(r) == 3 for r in out), "output is (n_queries x d_value)"
    assert len(w) == 3 and all(len(r) == 2 for r in w), "weights are (n_queries x n_keys)"
    assert all(_close(sum(r), 1.0) for r in w)


def test_scores_are_scaled_by_root_dk(m):
    Q = [[1.0, 1.0, 1.0, 1.0]]
    K = [[1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]]
    V = [[1.0], [0.0]]
    _, w = m.attention(Q, K, V)
    expected = math.exp(2) / (math.exp(2) + 1)   # scores are 4/sqrt(4)=2 and 0
    assert _close(w[0][0], expected), f"expected weight {expected:.4f}, got {w[0][0]:.4f}"


def test_identical_keys_average_the_values(m):
    K = [[0.5, 0.5]] * 3
    V = [[3.0, 0.0], [6.0, 3.0], [9.0, 9.0]]
    out, w = m.attention([[2.0, -1.0]], K, V)
    assert all(_close(x, 1 / 3) for x in w[0])
    assert _close(out[0][0], 6.0) and _close(out[0][1], 4.0)


def test_a_sharp_match_copies_one_value(m):
    K = [[10.0, 0.0], [0.0, 10.0]]
    V = [[1.0, 1.0], [-5.0, 7.0]]
    out, _ = m.attention([[0.0, 10.0]], K, V)
    assert abs(out[0][0] + 5.0) < 1e-6 and abs(out[0][1] - 7.0) < 1e-6


def test_causal_mask_hides_the_future(m):
    X = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    V = [[1.0], [2.0], [3.0]]
    out, w = m.attention(X, X, V, causal=True)
    for i, row in enumerate(w):
        for j, value in enumerate(row):
            if j > i:
                assert value == 0.0, f"position {i} must not see position {j}"
    assert w[0] == [1.0, 0.0, 0.0] and out[0] == [1.0], "the first token can only see itself"


def test_without_the_mask_the_future_is_visible(m):
    X = [[1.0, 0.0], [0.0, 1.0]]
    _, w = m.attention(X, X, [[1.0], [2.0]], causal=False)
    assert w[0][1] > 0.0


def test_self_attention_projects_then_attends(m):
    X = [[1.0, 2.0], [3.0, -1.0], [0.0, 1.0]]
    eye = [[1.0, 0.0], [0.0, 1.0]]
    got_out, got_w = m.self_attention(X, eye, eye, eye, causal=True)
    want_out, want_w = m.attention(X, X, X, causal=True)
    assert got_out == want_out and got_w == want_w
    double = [[2.0, 0.0], [0.0, 2.0]]
    out, _ = m.self_attention(X, eye, eye, double)
    plain, _ = m.attention(X, X, X)
    assert all(_close(a, 2 * b) for ra, rb in zip(out, plain) for a, b in zip(ra, rb)), "Wv scales the values"
