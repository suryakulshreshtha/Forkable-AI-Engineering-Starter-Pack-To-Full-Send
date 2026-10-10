"""
Title: Self-Eval
XP: 30

## The job
Grade the whole system. `check_case(result, case)` judges one answer: an `unanswerable`
case passes only if the system refused (no citations); otherwise every `expect_words`
entry must appear in the answer (case-insensitive) and, if `expect_path` is given, at least
one citation must come from that file. It returns `(passed, why)`. `evaluate(ask, golden)`
runs every case and reports totals and per-case results. `to_markdown(report)` renders a
small table you could paste into a pull request.

## Why it matters
Retrieval systems fail in two opposite ways: they miss answers that are in the docs, and
they invent answers that are not. A golden set with both kinds of questions measures both
failures at once, and running it on every change is what turns "it seems better" into
"it is better, and here is the table".

## Hints
- A citation id looks like "docs/guide.md#3"; the file is everything before the last "#".
- Build `why` from the specific failures: "should have refused", "missing words: x, y", "no citation from docs/x.md". Use "" when the case passes.
- `rate` is passed / total and 0.0 for an empty golden set.
- In the markdown table, show a check or a cross per case, the question, and the reason.
"""
from __future__ import annotations


def check_case(result, case):
    """Return (passed, why) for one {"answer", "citations"} result against one golden case."""
    # >>> fullsend: refusal cases first, then words and citation path
    if case.get("unanswerable"):
        if result["citations"]:
            return False, "should have refused"
        return True, ""
    problems = []
    missing = [w for w in case.get("expect_words", []) if w.lower() not in result["answer"].lower()]
    if missing:
        problems.append("missing words: " + ", ".join(missing))
    path = case.get("expect_path")
    if path and not any(c.rsplit("#", 1)[0] == path for c in result["citations"]):
        problems.append(f"no citation from {path}")
    return (not problems), "; ".join(problems)
    # <<< fullsend


def evaluate(ask, golden):
    """ask(question) -> {"answer", "citations"}. Returns {"total", "passed", "rate", "cases"}."""
    # >>> fullsend: run every case through ask and check_case
    cases = []
    for case in golden:
        passed, why = check_case(ask(case["q"]), case)
        cases.append({"q": case["q"], "passed": passed, "why": why})
    passed = sum(c["passed"] for c in cases)
    return {
        "total": len(cases),
        "passed": passed,
        "rate": passed / len(cases) if cases else 0.0,
        "cases": cases,
    }
    # <<< fullsend


def to_markdown(report):
    """A short summary line and a | result | question | why | table."""
    # >>> fullsend: header, separator, one row per case
    lines = [
        f"**{report['passed']} / {report['total']} passed** ({report['rate']:.0%})",
        "",
        "| result | question | why |",
        "| --- | --- | --- |",
    ]
    for case in report["cases"]:
        mark = "✅" if case["passed"] else "❌"
        question = case["q"].replace("|", "/")
        lines.append(f"| {mark} | {question} | {case['why'] or '-'} |")
    return "\n".join(lines)
    # <<< fullsend
