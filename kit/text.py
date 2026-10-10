"""Small, boring text helpers shared from level 4 onward."""
from __future__ import annotations

import re

_WORD = re.compile(r"[a-z0-9]+")
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+")

STOPWORDS = frozenset(
    """a an and are as at be but by can do does for from has have how i if in into is it its
    me my of on or our so than that the their them then there these they this to too us was
    we were what when where which who why will with you your""".split()
)


def words(text):
    """Lowercase alphanumeric tokens: 'Don't PANIC, 42!' -> ['don', 't', 'panic', '42']."""
    return _WORD.findall(text.lower())


def content_words(text):
    """words() minus common stopwords."""
    return [w for w in words(text) if w not in STOPWORDS]


def sentences(text):
    """Split on ., ! or ? followed by whitespace. Good enough for docs, not for law."""
    return [s.strip() for s in _SENTENCE_BREAK.split(text.strip()) if s.strip()]
