"""
Title: Cited Answers
XP: 35

## The job
Answer only from evidence. `score_sentence(query_terms, sentence)` counts distinct query
terms in the sentence divided by the square root of its content-word count (0.0 with no
overlap). `answer(query, hits, max_sentences, min_overlap)` scores every sentence of every
retrieved chunk, keeps the best few that share at least `min_overlap` query terms, puts them
back in reading order, and tags each with a citation marker like [1]. It returns
`{"answer", "citations"}`, or the fixed `REFUSAL` text with no citations when nothing
qualifies.

## Why it matters
An answer you cannot trace is an answer you cannot trust. This extractive answerer is a
stand-in for a language model, but the contract is the one production systems use: every
claim points at a source, and "I don't know" is a correct output when the sources are
silent. Swapping in a real model later changes the wording, not the contract.

## Hints
- Use `kit.text.content_words` for both the query and each sentence, and `kit.text.sentences` to split chunk text.
- Collect candidates as (score, hit_rank, sentence_position, sentence, chunk_id). Sort by score (desc), hit rank, position to choose; then re-sort the winners by hit rank and position to read naturally.
- Citations are the distinct chunk ids of the chosen sentences in order of first use. A sentence from the second distinct chunk gets the marker [2].
- Refuse when the query has no content words, or when no sentence reaches min_overlap.
"""
from __future__ import annotations

import math

from kit.text import content_words, sentences

REFUSAL = "I don't know. These docs do not cover that."


def score_sentence(query_terms, sentence):
    """|distinct query terms in sentence| / sqrt(|sentence content words|)."""
    # >>> fullsend: overlap normalised by sentence length
    words = content_words(sentence)
    if not words:
        return 0.0
    overlap = len(set(query_terms) & set(words))
    return overlap / math.sqrt(len(words)) if overlap else 0.0
    # <<< fullsend


def answer(query, hits, max_sentences=2, min_overlap=1):
    """hits: [(chunk, score), ...] from Index.search. Returns {"answer": str, "citations": [ids]}."""
    # >>> fullsend: score sentences, pick the best, restore reading order, cite
    terms = set(content_words(query))
    if not terms:
        return {"answer": REFUSAL, "citations": []}
    candidates = []
    for rank, (chunk, _) in enumerate(hits):
        for pos, sentence in enumerate(sentences(chunk["text"])):
            overlap = len(terms & set(content_words(sentence)))
            if overlap >= min_overlap:
                candidates.append((score_sentence(terms, sentence), rank, pos, sentence, chunk["id"]))
    if not candidates:
        return {"answer": REFUSAL, "citations": []}
    chosen = sorted(candidates, key=lambda c: (-c[0], c[1], c[2]))[:max_sentences]
    chosen.sort(key=lambda c: (c[1], c[2]))
    citations, parts = [], []
    for _, _, _, sentence, chunk_id in chosen:
        if chunk_id not in citations:
            citations.append(chunk_id)
        parts.append(f"{sentence} [{citations.index(chunk_id) + 1}]")
    return {"answer": " ".join(parts), "citations": citations}
    # <<< fullsend
