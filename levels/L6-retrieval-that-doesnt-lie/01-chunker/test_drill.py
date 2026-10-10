"""The judge for drill 6.1."""

DOC = (
    "Retrieval narrows the search. Ranking orders the results. "
    "Chunk size is a trade-off between context and precision! "
    "Is overlap worth the extra tokens? Usually, a little is. "
    "Evaluate before you tune anything."
)


def test_overlapping_windows(m):
    assert m.chunk_words("a b c d e f g", 3, 1) == ["a b c", "c d e", "e f g"]


def test_windows_without_overlap(m):
    assert m.chunk_words("a b c d e f g", 3) == ["a b c", "d e f", "g"]


def test_no_redundant_trailing_chunk(m):
    assert m.chunk_words("a b c d e f", 3) == ["a b c", "d e f"]
    assert m.chunk_words("a b c d e", 4, 2) == ["a b c d", "c d e"], "stop once a window reaches the end"


def test_short_and_empty_text(m):
    assert m.chunk_words("only two", 5, 2) == ["only two"]
    assert m.chunk_words("   ", 3) == []


def test_bad_settings_raise(m):
    for size, overlap in [(0, 0), (3, 3), (3, -1)]:
        try:
            m.chunk_words("a b c", size, overlap)
        except ValueError:
            continue
        raise AssertionError(f"size={size}, overlap={overlap} should raise ValueError")


def test_every_word_is_covered_in_order(m):
    words = [f"w{i}" for i in range(23)]
    chunks = m.chunk_words(" ".join(words), 5, 2)
    seen = []
    for c in chunks:
        for w in c.split():
            if not seen or int(w[1:]) > int(seen[-1][1:]):
                seen.append(w)
    assert seen == words


def test_sentence_chunks_respect_the_limit_and_keep_sentences_whole(m):
    chunks = m.chunk_sentences(DOC, 70)
    assert len(chunks) >= 3
    for c in chunks:
        assert len(c) <= 70, f"chunk too long: {c!r}"
        assert c[-1] in ".!?", f"chunk ends mid-sentence: {c!r}"
    assert " ".join(chunks) == " ".join(DOC.split()), "joining the chunks gives back the text"


def test_an_oversized_sentence_stands_alone(m):
    long = "This single sentence is much longer than the tiny limit we chose."
    chunks = m.chunk_sentences(f"Short one. {long} Tail.", 20)
    assert chunks == ["Short one.", long, "Tail."]
