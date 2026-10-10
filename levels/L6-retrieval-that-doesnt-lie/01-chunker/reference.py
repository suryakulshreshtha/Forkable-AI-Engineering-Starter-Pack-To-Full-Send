"""
Title: Chunker
XP: 20

## The job
Cut text into retrievable pieces two ways. `chunk_words(text, size, overlap)` makes windows
of `size` words where neighbouring windows share `overlap` words, and never emits an empty
or redundant trailing chunk. `chunk_sentences(text, max_chars)` packs whole sentences
greedily into chunks of at most `max_chars` characters; a single sentence that is longer
than the limit becomes a chunk of its own rather than being cut mid-sentence.

## Why it matters
Chunking quietly decides what a retrieval system can find. Chunks too small lose the
context that makes them meaningful; chunks too big dilute the match and waste the prompt.
Overlap stops an answer from being split across a boundary where neither half matches.

## Hints
- Words: `text.split()`. Step forward by `size - overlap`, and stop once a window reaches the last word.
- Validate early: size must be positive and 0 <= overlap < size, otherwise raise ValueError.
- `kit.text.sentences` splits text into sentences for you.
- For packing, keep a current chunk. If adding a space and the next sentence would pass max_chars, close the current chunk and start a new one with that sentence.
"""
from __future__ import annotations

from kit.text import sentences


def chunk_words(text, size, overlap=0):
    """Overlapping word windows. Returns a list of strings."""
    # >>> fullsend: validate, then slide a window of `size` words by `size - overlap`
    if size <= 0:
        raise ValueError("size must be positive")
    if not 0 <= overlap < size:
        raise ValueError("overlap must be in [0, size)")
    tokens = text.split()
    if not tokens:
        return []
    step = size - overlap
    chunks = []
    for start in range(0, len(tokens), step):
        chunks.append(" ".join(tokens[start:start + size]))
        if start + size >= len(tokens):
            break
    return chunks
    # <<< fullsend


def chunk_sentences(text, max_chars):
    """Greedily pack whole sentences into chunks of at most max_chars characters."""
    # >>> fullsend: never split a sentence; start a new chunk when the next one will not fit
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    chunks, current = [], ""
    for sentence in sentences(text):
        if not current:
            current = sentence
        elif len(current) + 1 + len(sentence) <= max_chars:
            current += " " + sentence
        else:
            chunks.append(current)
            current = sentence
    if current:
        chunks.append(current)
    return chunks
    # <<< fullsend
