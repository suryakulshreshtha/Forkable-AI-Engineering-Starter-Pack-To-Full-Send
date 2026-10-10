"""The judge for drill 8.2."""


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def test_keys_ignore_spacing_and_param_order(m):
    a = m.cache_key("What  is\nRAG?", temperature=0.0, model="small")
    b = m.cache_key("What is RAG?", model="small", temperature=0.0)
    assert a == b
    assert isinstance(a, str) and len(a) == 64, "a sha256 hex digest is 64 characters"


def test_keys_respect_real_differences(m):
    base = m.cache_key("What is RAG?", temperature=0.0)
    assert base != m.cache_key("What is RAG?", temperature=0.7), "different params, different answer"
    assert base != m.cache_key("what is rag?", temperature=0.0), "case is a real difference"


def test_lru_evicts_the_least_recently_used(m):
    cache = m.LRUCache(2)
    cache.put("a", 1)
    cache.put("b", 2)
    assert cache.get("a") == 1          # a is now the most recent
    cache.put("c", 3)                   # so b gets evicted
    assert "b" not in cache and "a" in cache and "c" in cache and len(cache) == 2


def test_put_refreshes_an_existing_key(m):
    cache = m.LRUCache(2)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.put("a", 10)
    cache.put("c", 3)
    assert cache.get("a") == 10 and "b" not in cache


def test_entries_expire(m):
    clock = FakeClock()
    cache = m.LRUCache(10, ttl=60, clock=clock)
    cache.put("k", "fresh")
    clock.now += 59
    assert cache.get("k") == "fresh"
    clock.now += 1
    assert cache.get("k") is None and "k" not in cache, "an entry 60s old with ttl=60 is expired"


def test_stats_count_hits_and_misses(m):
    cache = m.LRUCache(4)
    cache.get("nope")
    cache.put("x", 1)
    cache.get("x")
    cache.get("x")
    assert cache.stats == {"hits": 2, "misses": 1, "hit_rate": 2 / 3}


def test_cached_call_computes_once(m):
    calls = []

    def model(prompt, **params):
        calls.append(prompt)
        return f"answer to {prompt}"

    cache = m.LRUCache(8)
    first = m.cached_call(cache, model, "hello  world", temperature=0)
    second = m.cached_call(cache, model, "hello world", temperature=0)
    third = m.cached_call(cache, model, "hello world", temperature=1)
    assert first == second == "answer to hello  world" and third == "answer to hello world"
    assert len(calls) == 2


def test_falsy_answers_are_cached_too(m):
    calls = []

    def shy_model(prompt):
        calls.append(prompt)
        return ""

    cache = m.LRUCache(8)
    m.cached_call(cache, shy_model, "say nothing")
    m.cached_call(cache, shy_model, "say nothing")
    assert len(calls) == 1, "an empty-string answer is still an answer"
