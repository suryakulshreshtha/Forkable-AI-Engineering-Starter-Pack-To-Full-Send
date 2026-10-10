"""
Title: Bigram Language Model
XP: 25

## The job
Build the smallest real language model. `BigramLM.fit(sentences)` counts which token follows
which, wrapping each sentence in `<s>` and `</s>`. `prob(prev, word)` uses add-alpha
smoothing over the vocabulary (every token seen in training, plus `</s>`).
`perplexity(sentence)` scores how surprised the model is. `generate(rng, max_len)` samples
tokens until it draws `</s>` or reaches `max_len`.

## Why it matters
Perplexity is still the headline number in language model papers, and "predict the next
token, then sample" is exactly how chat models write. Smoothing teaches the oldest lesson
in the field: a model that assigns zero probability to anything it has not seen will fail
on the first new sentence it meets.

## Hints
- Store counts as `counts[prev][next]` with a `defaultdict(Counter)`.
- prob = (count(prev, word) + alpha) / (count(prev) + alpha * V), where count(prev) is how many bigrams start with prev and V is the vocabulary size.
- Perplexity = exp(-average log prob) over every transition, including the final one into `</s>`.
- To sample, walk the words in sorted order, adding up probabilities until the total passes `rng.random()`. Sorting keeps a seeded run reproducible.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict

BOS, EOS = "<s>", "</s>"


class BigramLM:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.counts = defaultdict(Counter)
        self.vocab = set()

    def fit(self, sentences):
        """sentences: list of token lists. Returns self."""
        # >>> fullsend: wrap each sentence, count transitions, collect the vocabulary
        for sentence in sentences:
            seq = [BOS] + list(sentence) + [EOS]
            for a, b in zip(seq, seq[1:]):
                self.counts[a][b] += 1
            self.vocab.update(sentence)
        self.vocab.add(EOS)
        return self
        # <<< fullsend

    def prob(self, prev, word):
        """P(word | prev) with add-alpha smoothing."""
        # >>> fullsend: smoothed count ratio
        row = self.counts.get(prev, Counter())
        total = sum(row.values())
        denom = total + self.alpha * len(self.vocab)
        if denom == 0:
            return 0.0
        return (row[word] + self.alpha) / denom
        # <<< fullsend

    def next_distribution(self, prev):
        """{word: P(word | prev)} for every word in the vocabulary."""
        return {w: self.prob(prev, w) for w in sorted(self.vocab)}

    def perplexity(self, sentence):
        """exp of the average negative log probability per transition (including into </s>)."""
        # >>> fullsend: sum log probs over the wrapped sentence
        seq = [BOS] + list(sentence) + [EOS]
        total = 0.0
        for a, b in zip(seq, seq[1:]):
            p = self.prob(a, b)
            if p <= 0:
                return math.inf
            total += math.log(p)
        return math.exp(-total / (len(seq) - 1))
        # <<< fullsend

    def generate(self, rng, max_len=20):
        """Sample tokens after <s> until </s> or max_len tokens. Returns the tokens without markers."""
        # >>> fullsend: draw from next_distribution with rng.random()
        out, prev = [], BOS
        while len(out) < max_len:
            r, cumulative, choice = rng.random(), 0.0, EOS
            for word, p in self.next_distribution(prev).items():
                cumulative += p
                if r < cumulative:
                    choice = word
                    break
            if choice == EOS:
                break
            out.append(choice)
            prev = choice
        return out
        # <<< fullsend
