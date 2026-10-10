"""The judge for drill 9.2."""

CHUNKS = [
    {"id": "setup.md#1", "path": "setup.md", "heading": "Installing Python",
     "text": "Download the installer and run it. Version 3.9 or newer works."},
    {"id": "setup.md#2", "path": "setup.md", "heading": "Editors",
     "text": "Any editor works. Python support helps with highlighting and caching."},
    {"id": "setup.md#3", "path": "setup.md", "heading": "Python versions",
     "text": "Check the Python version with the version flag."},
    {"id": "sync.md#1", "path": "sync.md", "heading": "Pulling updates",
     "text": "Add the upstream remote, fetch it, then merge upstream main into your branch."},
    {"id": "cache.md#1", "path": "cache.md", "heading": "Caching answers",
     "text": "Repeated prompts are served from the cache, which saves time and money."},
]


def test_chunk_tokens_weight_the_heading(m):
    tokens = m.chunk_tokens({"heading": "Pulling Updates", "text": "Fetch the upstream remote."})
    assert tokens == ["pulling", "updates", "pulling", "updates", "fetch", "upstream", "remote"], f"got {tokens}"


def test_search_finds_the_relevant_chunk(m):
    hits = m.Index(CHUNKS).search("how do I merge upstream changes?", k=1)
    assert len(hits) == 1 and hits[0][0]["id"] == "sync.md#1"


def test_a_heading_match_beats_a_passing_mention(m):
    hits = m.Index(CHUNKS).search("caching", k=2)
    assert hits[0][0]["id"] == "cache.md#1"


def test_results_are_sorted_and_limited(m):
    hits = m.Index(CHUNKS).search("python version", k=2)
    assert len(hits) == 2
    assert hits[0][1] >= hits[1][1]
    assert {h[0]["path"] for h in hits} == {"setup.md"}


def test_per_path_cap_adds_diversity(m):
    index = m.Index(CHUNKS)
    capped = index.search("python version editor cache", k=3, max_per_path=1)
    paths = [h[0]["path"] for h in capped]
    assert len(paths) == len(set(paths)), f"each file at most once, got {paths}"
    assert "cache.md" in paths


def test_stopword_only_queries_return_nothing(m):
    assert m.Index(CHUNKS).search("what is the", k=3) == []


def test_no_match_returns_nothing(m):
    assert m.Index(CHUNKS).search("quantum entanglement", k=3) == []
