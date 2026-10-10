"""The judge for drill 4.2."""
import random

from kit.text import words

TEXT = [
    "the cat sat on the mat",
    "the dog sat on the rug",
    "the cat chased the dog",
    "a dog sat on a mat",
]
CORPUS = [words(s) for s in TEXT]


def test_raw_counts_without_smoothing(m):
    lm = m.BigramLM(alpha=0).fit([["the", "cat", "sat"], ["the", "dog", "sat"]])
    assert lm.prob("<s>", "the") == 1.0
    assert lm.prob("the", "cat") == 0.5
    assert lm.prob("sat", "</s>") == 1.0


def test_fit_returns_self(m):
    lm = m.BigramLM()
    assert lm.fit(CORPUS) is lm


def test_each_row_is_a_probability_distribution(m):
    lm = m.BigramLM(alpha=0.5).fit(CORPUS)
    for prev in ["<s>", "the", "sat", "never-seen"]:
        total = sum(lm.prob(prev, w) for w in lm.vocab)
        assert abs(total - 1.0) < 1e-9, f"P(. | {prev}) sums to {total}"


def test_smoothing_gives_unseen_pairs_some_mass(m):
    lm = m.BigramLM(alpha=1.0).fit(CORPUS)
    assert lm.prob("mat", "cat") > 0
    assert lm.prob("the", "cat") > lm.prob("mat", "cat")


def test_perplexity_prefers_familiar_sentences(m):
    lm = m.BigramLM(alpha=0.1).fit(CORPUS)
    familiar = lm.perplexity(words("the dog sat on the mat"))
    scrambled = lm.perplexity(words("mat the on sat dog the"))
    assert familiar < scrambled, f"familiar={familiar:.2f} should be below scrambled={scrambled:.2f}"


def test_perplexity_of_a_certain_model_is_one(m):
    lm = m.BigramLM(alpha=0).fit([["only", "one", "path"]])
    assert abs(lm.perplexity(["only", "one", "path"]) - 1.0) < 1e-12


def test_generation_is_seeded_and_stays_in_vocabulary(m):
    lm = m.BigramLM(alpha=0.1).fit(CORPUS)
    a = lm.generate(random.Random(0), max_len=12)
    b = lm.generate(random.Random(0), max_len=12)
    assert a == b, "same seed, same sample"
    assert len(a) <= 12 and all(w in lm.vocab and w != "</s>" for w in a)


def test_unsmoothed_model_of_one_sentence_recites_it(m):
    lm = m.BigramLM(alpha=0).fit([["to", "be", "or", "not"]])
    assert lm.generate(random.Random(3)) == ["to", "be", "or", "not"]
