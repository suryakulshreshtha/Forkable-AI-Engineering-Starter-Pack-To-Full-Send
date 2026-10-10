"""The judge for drill 3.3."""


def test_split_sizes_and_disjointness(m):
    rows = list(range(100))
    train, test = m.train_test_split(rows, test_frac=0.2, seed=1)
    assert len(test) == 20 and len(train) == 80
    assert sorted(train + test) == rows, "every row lands in exactly one side"


def test_split_is_reproducible_and_actually_shuffles(m):
    rows = list(range(50))
    a = m.train_test_split(rows, 0.3, seed=5)
    b = m.train_test_split(rows, 0.3, seed=5)
    c = m.train_test_split(rows, 0.3, seed=6)
    assert a == b, "same seed must give the same split"
    assert a != c, "a different seed should give a different split"
    assert a[1] != rows[:15] and a[1] != rows[-15:], "the test set should not just be a slice of the input"


def test_split_leaves_the_input_alone(m):
    rows = list(range(10))
    m.train_test_split(rows, 0.5, seed=0)
    assert rows == list(range(10))


def test_confusion_counts(m):
    y_true = [1, 1, 0, 0, 1, 0]
    y_pred = [1, 0, 1, 0, 1, 0]
    assert m.confusion(y_true, y_pred) == {"tp": 2, "fp": 1, "fn": 1, "tn": 2}


def test_confusion_with_string_labels(m):
    c = m.confusion(["spam", "ham", "spam"], ["spam", "spam", "ham"], positive="spam")
    assert c == {"tp": 1, "fp": 1, "fn": 1, "tn": 0}


def test_the_99_percent_trap(m):
    y_true = [0] * 99 + [1]
    lazy = [0] * 100
    assert m.accuracy(y_true, lazy) == 0.99
    precision, recall, f1 = m.precision_recall_f1(y_true, lazy)
    assert (precision, recall, f1) == (0.0, 0.0, 0.0), "the lazy model catches nothing"


def test_precision_recall_f1_values(m):
    y_true = [1, 1, 1, 1, 0, 0, 0, 0]
    y_pred = [1, 1, 1, 0, 1, 1, 0, 0]
    p, r, f = m.precision_recall_f1(y_true, y_pred)
    assert abs(p - 0.6) < 1e-12 and abs(r - 0.75) < 1e-12
    assert abs(f - 2 * 0.6 * 0.75 / 1.35) < 1e-12


def test_length_mismatch_raises(m):
    try:
        m.accuracy([1, 0], [1])
    except ValueError:
        return
    raise AssertionError("mismatched label lists must raise ValueError")
