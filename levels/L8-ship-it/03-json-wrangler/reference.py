"""
Title: JSON Wrangler
XP: 25

## The job
Get structured data out of what a model actually wrote. `strip_fences` returns the inside
of the first triple-backtick code block, or the text unchanged. `find_json_span` returns the
first balanced JSON object or array, ignoring brackets inside strings. `remove_trailing_commas`
deletes commas that sit right before a closing bracket, but never inside strings.
`extract_json` chains them and parses, raising ValueError when there is nothing usable.
`validate(obj, schema)` checks required and optional (`name?`) fields and their types.

## Why it matters
"Please reply in JSON" works most of the time, and production code has to survive the rest:
a chatty preamble, a fenced block, a trailing comma, a brace inside a quoted string.
Validating the parsed result against a schema is the second half: valid JSON with the
wrong shape breaks your app just as hard as invalid JSON.

## Hints
- Scan character by character with three pieces of state: a stack of expected closing brackets, whether you are inside a string, and whether the previous character was a backslash.
- A quote character only toggles the in-string state when it is not escaped.
- Mismatched or missing closing brackets mean there is no valid span: raise ValueError.
- For trailing commas: outside strings, when you see a comma, peek past any whitespace; if the next character closes an object or array, drop the comma.
- In `validate`, keys ending in a question mark are optional. Booleans are not integers here either.
"""
from __future__ import annotations

import json
import re

_FENCE = re.compile(r"```[a-zA-Z0-9_-]*[ \t]*\n(.*?)```", re.DOTALL)
_PAIRS = {"{": "}", "[": "]"}


def strip_fences(text):
    """Inside of the first fenced code block, or the original text."""
    # >>> fullsend: regex for a fenced block
    match = _FENCE.search(text)
    return match.group(1) if match else text
    # <<< fullsend


def find_json_span(text):
    """The first balanced {...} or [...] substring. ValueError if there is none."""
    # >>> fullsend: bracket stack plus string and escape tracking
    start = next((i for i, ch in enumerate(text) if ch in _PAIRS), None)
    if start is None:
        raise ValueError("no JSON object or array found")
    stack, in_string, escaped = [], False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch in _PAIRS:
            stack.append(_PAIRS[ch])
        elif ch in "}]":
            if not stack or stack.pop() != ch:
                raise ValueError(f"mismatched bracket at position {i}")
            if not stack:
                return text[start:i + 1]
    raise ValueError("unbalanced JSON: missing closing bracket")
    # <<< fullsend


def remove_trailing_commas(s):
    """Drop commas directly before } or ], outside of strings."""
    # >>> fullsend: string-aware scan with a look-ahead past whitespace
    out, in_string, escaped = [], False, False
    for i, ch in enumerate(s):
        if in_string:
            out.append(ch)
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == ",":
            j = i + 1
            while j < len(s) and s[j] in " \t\r\n":
                j += 1
            if j < len(s) and s[j] in "}]":
                continue
        out.append(ch)
    return "".join(out)
    # <<< fullsend


def extract_json(text):
    """Parse the first JSON value in messy model output, repairing trailing commas."""
    # >>> fullsend: fences, span, parse, repair once, parse again
    candidate = find_json_span(strip_fences(text))
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass
    try:
        return json.loads(remove_trailing_commas(candidate))
    except json.JSONDecodeError as exc:
        raise ValueError(f"found JSON-like text but could not parse it: {exc}") from None
    # <<< fullsend


def _type_ok(value, expected):
    if expected is int:
        return isinstance(value, int) and not isinstance(value, bool)
    if expected is float:
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    return isinstance(value, expected)


def validate(obj, schema):
    """schema: {"name": type, "nickname?": str, ...}. Returns a list of error strings."""
    # >>> fullsend: object check, then each field: present? right type?
    if not isinstance(obj, dict):
        return ["expected an object"]
    errors = []
    for key, expected in schema.items():
        optional = key.endswith("?")
        name = key[:-1] if optional else key
        if name not in obj:
            if not optional:
                errors.append(f"missing field '{name}'")
            continue
        if not _type_ok(obj[name], expected):
            errors.append(f"field '{name}' should be {expected.__name__}")
    return errors
    # <<< fullsend
