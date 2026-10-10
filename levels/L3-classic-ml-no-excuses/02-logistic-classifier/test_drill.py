"""The judge for drill 3.2."""
import math
import random


def _blobs(seed=3, n=30):
    rng = random.Random(seed)
    X, y = [], []
    for _ in range(n):
        X.append([rng.gauss(-2, 0.6), rng.gauss(-2, 0.6)])
        y.append(0)
        X.append([rng.gauss(2, 0.6), rng.gauss(2, 0.6)])
        y.append(1)
    return X, y


def test_sigmoid_values(m):
    assert m.sigmoid(0) == 0.5
    assert abs(m.sigmoid(2) - 1 / (1 + math.exp(-2))) < 1e-12
    assert abs(m.sigmoid(-2) + m.sigmoid(2) - 1.0) < 1e-12, "sigmoid(-z) = 1 - sigmoid(z)"


def test_sigmoid_never_overflows(m):
    assert m.sigmoid(-1000) == 0.0 or m.sigmoid(-1000) < 1e-300
    assert m.sigmoid(1000) == 1.0


def test_bce_rewards_confidence_when_right(m):
    assert m.bce(0.9, 1) < m.bce(0.6, 1) < m.bce(0.1, 1)
    assert abs(m.bce(0.5, 0) - math.log(2)) < 1e-12


def test_bce_is_finite_at_the_edges(m):
    assert math.isfinite(m.bce(0.0, 1)) and math.isfinite(m.bce(1.0, 0)), "clip p before taking logs"


def test_fit_returns_self(m):
    X, y = _blobs()
    model = m.LogisticRegression(epochs=5)
    assert model.fit(X, y) is model


def test_separates_two_clouds(m):
    X, y = _blobs()
    model = m.LogisticRegression(lr=0.1, epochs=300).fit(X, y)
    assert model.predict(X) == y, "two well-separated clouds should be classified perfectly"
    assert model.loss(X, y) < 0.1


def test_probabilities_stay_strictly_inside_zero_and_one(m):
    X, y = _blobs()
    probs = m.LogisticRegression(epochs=200).fit(X, y).predict_proba(X)
    assert all(0.0 < p < 1.0 for p in probs)


def test_a_higher_threshold_predicts_fewer_positives(m):
    X, y = _blobs(seed=11)
    X.append([0.1, -0.1])   # a borderline point
    y.append(1)
    model = m.LogisticRegression(epochs=200).fit(X, y)
    assert sum(model.predict(X, threshold=0.9)) <= sum(model.predict(X, threshold=0.1))
