"""The judge for drill 1.1. Each test receives `m`, the module under test."""


def _raises(fn, *args):
    try:
        fn(*args)
    except ValueError:
        return True
    return False


def test_dot_multiplies_pairwise_and_sums(m):
    assert m.dot([1, 2, 3], [4, 5, 6]) == 32, "1*4 + 2*5 + 3*6 should be 32"
    assert m.dot([], []) == 0, "the dot of two empty vectors is 0"


def test_dot_refuses_mismatched_lengths(m):
    assert _raises(m.dot, [1, 2], [1, 2, 3]), "dot([1,2],[1,2,3]) should raise ValueError"


def test_matvec_is_one_dot_per_row(m):
    W = [[1, 0, -1], [2, 2, 2]]
    assert m.matvec(W, [3, 4, 5]) == [-2, 24]


def test_transpose_flips_rows_and_columns(m):
    assert m.transpose([[1, 2, 3], [4, 5, 6]]) == [[1, 4], [2, 5], [3, 6]]
    assert m.transpose([]) == []


def test_matmul_shapes_and_values(m):
    A = [[1, 2, 3], [4, 5, 6]]          # 2 x 3
    B = [[7, 8], [9, 10], [11, 12]]     # 3 x 2
    assert m.matmul(A, B) == [[58, 64], [139, 154]]


def test_matmul_with_identity_changes_nothing(m):
    A = [[2, -1], [0.5, 3]]
    eye = [[1, 0], [0, 1]]
    assert m.matmul(A, eye) == A and m.matmul(eye, A) == A


def test_matmul_refuses_bad_inner_dimension(m):
    assert _raises(m.matmul, [[1, 2]], [[1, 2]]), "(1x2) @ (1x2) must raise ValueError"


def test_a_dense_layer_is_matvec_plus_bias(m):
    weights = [[0.5, -1.0], [2.0, 0.0], [1.0, 1.0]]
    bias = [0.0, -1.0, 0.5]
    x = [2.0, 1.0]
    out = [z + b for z, b in zip(m.matvec(weights, x), bias)]
    assert out == [0.0, 3.0, 3.5]
