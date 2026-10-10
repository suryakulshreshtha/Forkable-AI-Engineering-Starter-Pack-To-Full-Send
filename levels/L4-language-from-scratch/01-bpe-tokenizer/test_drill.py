"""The judge for drill 4.1."""

CORPUS = (
    "the rain in spain stays mainly in the plain. "
    "the train in the rain is plainly late again. "
    "rainy days and plain trains remain the main complaint."
)


def test_pair_counts(m):
    counts = m.pair_counts(list("aaab"))
    assert dict(counts) == {("a", "a"): 2, ("a", "b"): 1}


def test_merge_is_left_to_right_and_non_overlapping(m):
    assert m.merge(list("aaab"), ("a", "a")) == ["aa", "a", "b"]
    assert m.merge(list("aaaa"), ("a", "a")) == ["aa", "aa"]
    assert m.merge(["x", "y", "z"], ("q", "r")) == ["x", "y", "z"], "no match, no change"


def test_ties_go_to_the_pair_seen_first(m):
    merges = m.train_bpe("hello hello hello", 1)
    assert merges == [("h", "e")], f"h-e, e-l, l-l and l-o all occur 3 times; h-e comes first. Got {merges}"


def test_merges_build_whole_words(m):
    merges = m.train_bpe("hello hello hello", 4)
    assert m.encode("hello", merges) == ["hello"], f"after 4 merges 'hello' is one token; merges={merges}"


def test_training_stops_when_nothing_repeats(m):
    assert m.train_bpe("abcdef", 10) == []


def test_round_trip_on_unseen_text(m):
    merges = m.train_bpe(CORPUS, 40)
    for text in ["the plain train", "spain rains", "zebra!", "", "  double  spaces  "]:
        assert m.decode(m.encode(text, merges)) == text, f"round trip failed for {text!r}"


def test_merges_compress_the_corpus(m):
    merges = m.train_bpe(CORPUS, 40)
    assert len(m.encode(CORPUS, merges)) < len(CORPUS) * 0.6, "40 merges should cut the token count well below 60%"


def test_training_is_deterministic(m):
    assert m.train_bpe(CORPUS, 25) == m.train_bpe(CORPUS, 25)
