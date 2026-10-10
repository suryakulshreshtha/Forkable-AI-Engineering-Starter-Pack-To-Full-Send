"""
Title: Eval Harness
XP: 30

## The job
Build a tiny evaluation harness. A `Case` has an id, an input, an expected value and a
mode: `exact` (after trimming whitespace), `contains` (case-insensitive) or `regex`.
`score(output, case)` returns a bool. `run_eval(fn, cases)` runs every case, counts a crash
as a failure and records its error, and rejects duplicate ids. `compare(old, new)` lists
which cases were fixed and which broke. `gate(old, new, max_broken)` says whether a change
is safe to ship.

## Why it matters
Prompt and model changes are code changes, and a higher average can hide a case that used
to work and now fails. Teams that ship AI features reliably keep a fixed set of cases and
look at the per-case diff, not just the headline score. This is that habit, in fifty lines.

## Hints
- Unknown modes are a bug in the test set: raise ValueError rather than silently failing the case.
- Wrap only the call to fn in try/except, so one crashing case does not stop the run while a broken test set still fails loudly. Record "TypeName: message". An output that is not a string is a failure too.
- `rate` is passed / total, and 0.0 when there are no cases.
- fixed = ids that failed before and pass now; broke = the reverse. Return both sorted.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class Case:
    id: str
    input: str
    expect: str
    mode: str = "exact"


def score(output, case):
    """True when output satisfies the case's expectation."""
    # >>> fullsend: branch on the mode
    if case.mode == "exact":
        return output.strip() == case.expect.strip()
    if case.mode == "contains":
        return case.expect.lower() in output.lower()
    if case.mode == "regex":
        return re.search(case.expect, output) is not None
    raise ValueError(f"unknown mode '{case.mode}' in case {case.id}")
    # <<< fullsend


def run_eval(fn, cases):
    """Return {"passed", "total", "rate", "results": {id: bool}, "errors": {id: str}}."""
    # >>> fullsend: check ids, run each case safely, tally
    ids = [c.id for c in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("case ids must be unique")
    results, errors = {}, {}
    for case in cases:
        try:
            output = fn(case.input)
        except Exception as exc:
            results[case.id] = False
            errors[case.id] = f"{type(exc).__name__}: {exc}"
            continue
        if not isinstance(output, str):
            results[case.id] = False
            errors[case.id] = f"TypeError: expected a string, got {type(output).__name__}"
            continue
        results[case.id] = bool(score(output, case))
    passed = sum(results.values())
    return {
        "passed": passed,
        "total": len(cases),
        "rate": passed / len(cases) if cases else 0.0,
        "results": results,
        "errors": errors,
    }
    # <<< fullsend


def compare(old, new):
    """{"fixed": [...], "broke": [...], "delta": new rate - old rate}."""
    # >>> fullsend: per-case diff of two run_eval reports
    fixed = sorted(i for i, ok in new["results"].items() if ok and not old["results"].get(i, False))
    broke = sorted(i for i, ok in new["results"].items() if not ok and old["results"].get(i, False))
    return {"fixed": fixed, "broke": broke, "delta": new["rate"] - old["rate"]}
    # <<< fullsend


def gate(old, new, max_broken=0):
    """Ship only if no more than max_broken previously passing cases now fail."""
    # >>> fullsend: count what broke
    return len(compare(old, new)["broke"]) <= max_broken
    # <<< fullsend
