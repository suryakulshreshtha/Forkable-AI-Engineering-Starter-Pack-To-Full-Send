"""The judge for drill 6.2."""
import math

from kit.text import words

DOCS = [
    "the quick brown fox jumps over the lazy dog",
    "the lazy cat sleeps all day",
    "a fox and a cat walk into the forest",
    "kubernetes schedules containers onto nodes",
    "the dog barks at the mail carrier every day",
]
TOKENS = [words(d) for d in DOCS]


def test_idf_by_hand(m):
    bm = m.BM25(TOKENS)
    n_the = 4    # 'the' appears in four of five documents
    assert abs(bm.idf("the") - math.log((5 - n_the + 0.5) / (n_the + 0.5) + 1)) < 1e-12
    assert abs(bm.idf("zebra") - math.log(5.5 / 0.5 + 1)) < 1e-12


def test_rare_terms_weigh_more(m):
    bm = m.BM25(TOKENS)
    assert bm.idf("kubernetes") > bm.idf("fox") > bm.idf("the") > 0


def test_search_finds_the_rare_word(m):
    bm = m.BM25(TOKENS)
    top = bm.search(words("the containers"), top_k=1)
    assert top and top[0][0] == 3, f"the only document about containers should win, got {top}"


def test_results_are_sorted_and_cut_to_top_k(m):
    bm = m.BM25(TOKENS)
    hits = bm.search(words("lazy fox"), top_k=3)
    assert len(hits) == 3
    scores = [s for _, s in hits]
    assert scores == sorted(scores, reverse=True)
    assert hits[0][0] == 0, "the doc with both 'lazy' and 'fox' should lead"


def test_unknown_terms_return_nothing(m):
    assert m.BM25(TOKENS).search(["zebra", "quasar"]) == []


def test_term_frequency_saturates(m):
    once = ["cat"] + ["filler"] * 9
    many = ["cat"] * 10
    bm = m.BM25([once, many, ["other"] * 10])
    s1, s10 = bm.score(["cat"], 0), bm.score(["cat"], 1)
    assert s1 < s10 < 3 * s1, f"ten mentions should help, but far less than tenfold ({s1:.3f} vs {s10:.3f})"


def test_short_documents_win_ties_in_frequency(m):
    bm = m.BM25([["cat", "x"], ["cat"] + ["x"] * 20, ["y"] * 5])
    assert bm.score(["cat"], 0) > bm.score(["cat"], 1)


def test_b_zero_turns_off_length_normalisation(m):
    bm = m.BM25([["cat", "x"], ["cat"] + ["x"] * 20, ["y"] * 5], b=0.0)
    assert abs(bm.score(["cat"], 0) - bm.score(["cat"], 1)) < 1e-12


def test_ties_break_by_index(m):
    bm = m.BM25([["a", "b"], ["a", "b"], ["c"]])
    assert [i for i, _ in bm.search(["a"])] == [0, 1]
