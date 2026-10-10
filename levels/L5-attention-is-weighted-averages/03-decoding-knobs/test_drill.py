"""The judge for drill 5.3."""
import random
from collections import Counter


def _close_lists(a, b, tol=1e-12):
    return len(a) == len(b) and all(abs(x - y) <= tol for x, y in zip(a, b))


def test_greedy_picks_the_top_and_breaks_ties_low(m):
    assert m.greedy([0.1, 0.5, 0.4]) == 1
    assert m.greedy([0.5, 0.5]) == 0


def test_top_k_keeps_the_k_largest(m):
    got = m.top_k_filter([0.1, 0.4, 0.2, 0.3], 2)
    assert _close_lists(got, [0.0, 4 / 7, 0.0, 3 / 7]), f"got {got}"


def test_top_k_larger_than_vocab_changes_nothing(m):
    probs = [0.2, 0.3, 0.5]
    assert _close_lists(m.top_k_filter(probs, 10), probs)


def test_top_p_keeps_the_smallest_set_that_reaches_p(m):
    probs = [0.5, 0.3, 0.15, 0.05]
    got = m.top_p_filter(probs, 0.9)
    assert _close_lists(got, [0.5 / 0.95, 0.3 / 0.95, 0.15 / 0.95, 0.0]), f"got {got}"
    assert _close_lists(m.top_p_filter(probs, 0.4), [1.0, 0.0, 0.0, 0.0])


def test_top_p_ranks_by_probability_not_position(m):
    got = m.top_p_filter([0.05, 0.15, 0.3, 0.5], 0.7)
    assert _close_lists(got, [0.0, 0.0, 0.375, 0.625]), f"got {got}"


def test_sampling_is_seeded(m):
    probs = [0.2, 0.5, 0.3]
    a = [m.sample(probs, random.Random(9)) for _ in range(5)]
    rng1, rng2 = random.Random(9), random.Random(9)
    assert [m.sample(probs, rng1) for _ in range(50)] == [m.sample(probs, rng2) for _ in range(50)]
    assert all(0 <= i < 3 for i in a)


def test_sample_frequencies_follow_the_probabilities(m):
    probs = [0.2, 0.5, 0.3]
    rng = random.Random(0)
    counts = Counter(m.sample(probs, rng) for _ in range(20000))
    for i, p in enumerate(probs):
        assert abs(counts[i] / 20000 - p) < 0.02, f"index {i}: {counts[i] / 20000:.3f} vs {p}"


def test_sample_never_picks_impossible_tokens(m):
    rng = random.Random(1)
    assert {m.sample([0.0, 1.0, 0.0], rng) for _ in range(200)} == {1}


def test_decode_step_knobs(m):
    logits = [1.0, 3.0, 2.5, -1.0]
    rng = random.Random(4)
    assert {m.decode_step(logits, rng, temperature=0.01) for _ in range(50)} == {1}, "near-zero temperature is greedy"
    assert {m.decode_step(logits, rng, top_k=1) for _ in range(50)} == {1}, "top_k=1 is greedy"
    picks = {m.decode_step(logits, rng, top_k=2) for _ in range(300)}
    assert picks <= {1, 2} and len(picks) == 2, "top_k=2 samples only from the two best"
