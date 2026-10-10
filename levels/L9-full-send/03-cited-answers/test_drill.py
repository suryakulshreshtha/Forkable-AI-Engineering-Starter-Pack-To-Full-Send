"""The judge for drill 9.3."""
import math

SYNC = {"id": "sync.md#1", "path": "sync.md", "heading": "Pulling updates",
        "text": "Forks drift over time. Add the upstream remote once. "
                "Then fetch upstream and merge it into your main branch. Conflicts are rare."}
CACHE = {"id": "cache.md#1", "path": "cache.md", "heading": "Caching",
         "text": "The cache saves money. Merge requests should include cache tests."}


def test_score_sentence(m):
    assert m.score_sentence({"upstream", "remote"}, "Add the upstream remote once.") == 2 / math.sqrt(4)
    assert m.score_sentence({"cache"}, "Nothing relevant here.") == 0.0
    assert m.score_sentence({"cache"}, "The is a.") == 0.0, "no content words at all"


def test_answers_from_the_best_sentences_with_citations(m):
    out = m.answer("how do I fetch and merge upstream?", [(SYNC, 2.0)], max_sentences=2)
    assert out["citations"] == ["sync.md#1"]
    assert out["answer"] == "Add the upstream remote once. [1] Then fetch upstream and merge it into your main branch. [1]"


def test_reading_order_is_restored(m):
    out = m.answer("upstream remote fetch merge", [(SYNC, 2.0)], max_sentences=3)
    first, second = out["answer"].index("Add the upstream"), out["answer"].index("Then fetch")
    assert first < second, "chosen sentences must appear in their original order"


def test_markers_number_distinct_sources(m):
    out = m.answer("merge cache", [(SYNC, 2.0), (CACHE, 1.0)], max_sentences=3)
    assert out["citations"] == ["sync.md#1", "cache.md#1"], f"got {out['citations']}"
    assert "[2]" in out["answer"] and "[1]" in out["answer"]


def test_max_sentences_is_respected(m):
    out = m.answer("upstream remote fetch merge branch", [(SYNC, 2.0)], max_sentences=1)
    assert out["answer"].count("[1]") == 1


def test_refuses_without_evidence(m):
    out = m.answer("what is the capital of France?", [(SYNC, 2.0), (CACHE, 1.0)])
    assert out == {"answer": m.REFUSAL, "citations": []}


def test_refuses_on_an_empty_query(m):
    assert m.answer("what is it?", [(SYNC, 1.0)]) == {"answer": m.REFUSAL, "citations": []}
    assert m.answer("upstream", []) == {"answer": m.REFUSAL, "citations": []}


def test_min_overlap_raises_the_bar(m):
    loose = m.answer("upstream france", [(SYNC, 1.0)], min_overlap=1)
    strict = m.answer("upstream france", [(SYNC, 1.0)], min_overlap=2)
    assert loose["citations"] == ["sync.md#1"] and strict["citations"] == []
