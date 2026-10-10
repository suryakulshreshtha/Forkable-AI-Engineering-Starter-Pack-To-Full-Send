"""The judge for drill 4.3."""
from kit.text import words

TEXT = [
    "the cat sat on the mat", "the dog sat on the rug",
    "a cat ate the fish", "a dog ate the bone",
    "the cat chased a mouse", "the dog chased a ball",
    "stocks rose on the news", "bonds fell on the news",
    "stocks fell after the report", "bonds rose after the report",
]
SENTENCES = [words(s) for s in TEXT]


def test_window_counts_by_hand(m):
    v = m.build_vectors([["a", "b", "c", "d"]], window=1)
    assert dict(v["b"]) == {"a": 1, "c": 1}
    assert dict(v["a"]) == {"b": 1}, "the first word only has a right neighbour"


def test_wider_window_sees_further(m):
    v = m.build_vectors([["a", "b", "c", "d"]], window=2)
    assert dict(v["a"]) == {"b": 1, "c": 1}
    assert dict(v["c"]) == {"a": 1, "b": 1, "d": 1}


def test_counts_accumulate_across_sentences(m):
    v = m.build_vectors([["x", "y"], ["x", "y"], ["y", "x"]], window=1)
    assert v["x"]["y"] == 3


def test_cosine_basics(m):
    assert abs(m.cosine({"a": 1, "b": 2}, {"a": 2, "b": 4}) - 1.0) < 1e-12, "same direction -> 1"
    assert m.cosine({"a": 1}, {"b": 1}) == 0.0, "no shared context -> 0"
    assert m.cosine({}, {"a": 1}) == 0.0, "an empty vector has no direction"


def test_similar_usage_means_nearby_vectors(m):
    vectors = m.build_vectors(SENTENCES, window=2)
    assert m.nearest("cat", vectors, k=1)[0][0] == "dog"
    assert m.nearest("stocks", vectors, k=1)[0][0] == "bonds"
    assert m.nearest("rose", vectors, k=1)[0][0] == "fell"


def test_nearest_excludes_itself_and_respects_k(m):
    vectors = m.build_vectors(SENTENCES, window=2)
    result = m.nearest("fish", vectors, k=4)
    assert len(result) == 4 and all(w != "fish" for w, _ in result)
    scores = [s for _, s in result]
    assert scores == sorted(scores, reverse=True), "best first"


def test_ties_are_alphabetical(m):
    vectors = m.build_vectors(SENTENCES, window=2)
    top_two = [w for w, _ in m.nearest("mat", vectors, k=2)]
    assert top_two == ["news", "rug"], f"'news' and 'rug' tie at 1.0; alphabetical order expected, got {top_two}"


def test_unknown_word_raises_key_error(m):
    vectors = m.build_vectors(SENTENCES)
    try:
        m.nearest("giraffe", vectors)
    except KeyError:
        return
    raise AssertionError("asking about a word with no vector should raise KeyError")
