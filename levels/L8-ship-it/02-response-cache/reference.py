"""
Title: Response Cache
XP: 25

## The job
Stop paying twice for the same answer. `cache_key(prompt, **params)` hashes the prompt
(with runs of whitespace collapsed) together with the parameters, so key order does not
matter but any real difference does. `LRUCache(capacity, ttl, clock)` keeps the most
recently used entries, evicts the least recently used when full, expires entries older than
`ttl` seconds, and counts hits and misses. `cached_call` puts them together and must cache
falsy answers such as an empty string.

## Why it matters
Repeated prompts are common in real traffic and in test suites, and a cache is the cheapest
latency and cost win available. The details are where caches go wrong: two prompts that
differ only in spacing should share an entry, two calls with different temperatures should
not, and stale answers must age out.

## Hints
- `json.dumps(..., sort_keys=True)` makes a stable string out of a dict; hash it with `hashlib.sha256(...).hexdigest()`.
- `collections.OrderedDict` has `move_to_end(key)` and `popitem(last=False)`, which is most of an LRU cache.
- Store (value, stored_at) and compare `clock() - stored_at` with the ttl on every get. An entry is expired when its age reaches ttl.
- Use a private sentinel object (not None) to tell "missing" apart from "cached an empty or None answer".
"""
from __future__ import annotations

import hashlib
import json
import time
from collections import OrderedDict

_MISSING = object()


def cache_key(prompt, **params):
    """Stable hex digest of the whitespace-normalised prompt plus sorted params."""
    # >>> fullsend: normalise, serialise with sorted keys, hash
    normalised = " ".join(prompt.split())
    payload = json.dumps({"prompt": normalised, "params": params}, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
    # <<< fullsend


class LRUCache:
    def __init__(self, capacity, ttl=None, clock=time.monotonic):
        if capacity < 1:
            raise ValueError("capacity must be at least 1")
        self.capacity, self.ttl, self.clock = capacity, ttl, clock
        self._data = OrderedDict()
        self.hits = self.misses = 0

    def get(self, key, default=None):
        """Return the cached value (and mark it recently used), or default on a miss or expiry."""
        # >>> fullsend: miss, expired, or hit
        if key not in self._data:
            self.misses += 1
            return default
        value, stored_at = self._data[key]
        if self.ttl is not None and self.clock() - stored_at >= self.ttl:
            del self._data[key]
            self.misses += 1
            return default
        self._data.move_to_end(key)
        self.hits += 1
        return value
        # <<< fullsend

    def put(self, key, value):
        """Store value now; evict least recently used entries beyond capacity."""
        # >>> fullsend: insert or refresh, then evict
        self._data[key] = (value, self.clock())
        self._data.move_to_end(key)
        while len(self._data) > self.capacity:
            self._data.popitem(last=False)
        # <<< fullsend

    def __len__(self):
        return len(self._data)

    def __contains__(self, key):
        return key in self._data

    @property
    def stats(self):
        total = self.hits + self.misses
        return {"hits": self.hits, "misses": self.misses, "hit_rate": self.hits / total if total else 0.0}


def cached_call(cache, fn, prompt, **params):
    """Return fn(prompt, **params), computing it only on a cache miss."""
    # >>> fullsend: key, look up with a sentinel, compute and store on a miss
    key = cache_key(prompt, **params)
    value = cache.get(key, _MISSING)
    if value is _MISSING:
        value = fn(prompt, **params)
        cache.put(key, value)
    return value
    # <<< fullsend
