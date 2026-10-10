"""
Title: Retry and Budget
XP: 25

## The job
Write the guardrails around flaky, metered calls. `retry(fn, attempts, base_delay, sleep,
retry_on, max_delay)` calls `fn()` until it succeeds, sleeping base_delay * 2^i (capped at
max_delay) between attempts, and re-raises the last error when out of attempts. Errors not
in `retry_on` propagate immediately. `TokenBudget` tracks spending and raises
`BudgetExceeded` before going over. `guarded_call` charges the budget for every attempt.

## Why it matters
APIs time out and rate-limit; retrying with backoff is mandatory. But each retry costs as
much as the first try, and an agent that retries inside a loop can quietly multiply your
bill. Charging before the call and refusing to overspend is the difference between a
surprising invoice and a clear error message.

## Hints
- Take `sleep` as a parameter so tests can pass a fake that records delays instead of waiting.
- Delay before retry number i (counting from 0) is min(max_delay, base_delay * 2 ** i). There is no sleep after the final failure.
- Only catch the exception types in `retry_on`; anything else should escape on the first attempt.
- `charge` checks first and only then adds, so a rejected charge leaves `used` unchanged.
- In `guarded_call`, BudgetExceeded is not retryable: retrying will not make money appear.
"""
from __future__ import annotations

import math
import time


class BudgetExceeded(Exception):
    """Raised when a charge would push spending past the limit."""


def estimate_tokens(text):
    """A rough rule of thumb: one token per four characters, rounded up."""
    # >>> fullsend: ceil(len / 4)
    return math.ceil(len(text) / 4)
    # <<< fullsend


class TokenBudget:
    def __init__(self, limit):
        if limit < 0:
            raise ValueError("limit must be non-negative")
        self.limit = limit
        self.used = 0

    @property
    def remaining(self):
        return self.limit - self.used

    def can_afford(self, n):
        # >>> fullsend: would n more fit?
        return self.used + n <= self.limit
        # <<< fullsend

    def charge(self, n):
        """Add n to used, or raise BudgetExceeded without changing anything."""
        # >>> fullsend: reject negatives, check, then spend
        if n < 0:
            raise ValueError("cannot charge a negative amount")
        if not self.can_afford(n):
            raise BudgetExceeded(f"need {n}, only {self.remaining} of {self.limit} left")
        self.used += n
        # <<< fullsend


def retry(fn, attempts=3, base_delay=0.5, sleep=time.sleep, retry_on=(Exception,), max_delay=8.0):
    """Call fn() up to `attempts` times with exponential backoff between failures."""
    # >>> fullsend: loop, catch only retry_on, sleep between attempts, re-raise at the end
    if attempts < 1:
        raise ValueError("attempts must be at least 1")
    for i in range(attempts):
        try:
            return fn()
        except retry_on:
            if i == attempts - 1:
                raise
            sleep(min(max_delay, base_delay * 2 ** i))
    # <<< fullsend


def guarded_call(fn, prompt, budget, attempts=3, base_delay=0.5, sleep=time.sleep):
    """Retry fn(prompt) on TimeoutError or ConnectionError, charging the budget before every attempt."""
    # >>> fullsend: wrap one attempt (charge, then call) and retry it
    def attempt():
        budget.charge(estimate_tokens(prompt))
        return fn(prompt)

    return retry(attempt, attempts, base_delay, sleep, retry_on=(TimeoutError, ConnectionError))
    # <<< fullsend
