"""
Title: Tool Registry
XP: 25

## The job
Build the registry an agent calls tools through. `register(name, fn, params, required,
description)` records a tool, where params maps argument names to Python types and
`required` defaults to every param. `validate(name, args)` returns a list of readable error
strings (unknown tool, missing, unexpected or wrongly typed arguments). `call(name, args)`
never raises: it returns `{"ok": True, "result": ...}` or `{"ok": False, "error": "..."}`.
`describe()` lists tool specs, sorted by name, ready to show a model.

## Why it matters
Models produce arguments that are almost right: a string where you wanted a number, an
extra field, a missing one. Validating before running stops bad calls from touching real
systems, and returning errors as data (instead of crashing) lets the model read its mistake
and try again. Every serious agent framework has this layer.

## Hints
- In Python, True is an int. Reject booleans where an integer or number is expected.
- Accept an int where a float is expected; numbers are numbers.
- Do not run the tool at all if validation finds problems.
- Catch exceptions from the tool itself and report them as "TypeName: message".
- Map Python types to schema words (str to "string", int to "integer", float to "number", bool to "boolean", list to "array", dict to "object") in `describe`.
"""
from __future__ import annotations

TYPE_NAMES = {str: "string", int: "integer", float: "number", bool: "boolean", list: "array", dict: "object"}


def type_ok(value, expected):
    """True when value fits the expected type, with the bool and int-as-float rules."""
    # >>> fullsend: special-case float and int, then isinstance
    if expected is float:
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected is int:
        return isinstance(value, int) and not isinstance(value, bool)
    return isinstance(value, expected)
    # <<< fullsend


class ToolRegistry:
    def __init__(self):
        self._tools = {}

    def register(self, name, fn, params=None, required=None, description=""):
        """Raise ValueError for a duplicate name or a required name missing from params."""
        # >>> fullsend: validate the registration itself, then store it
        if name in self._tools:
            raise ValueError(f"tool '{name}' is already registered")
        params = dict(params or {})
        required = list(params) if required is None else list(required)
        unknown = [r for r in required if r not in params]
        if unknown:
            raise ValueError(f"required names not in params: {unknown}")
        self._tools[name] = {"fn": fn, "params": params, "required": required, "description": description}
        # <<< fullsend

    def names(self):
        return sorted(self._tools)

    def validate(self, name, args):
        """A list of problems; empty means the call is safe to run."""
        # >>> fullsend: unknown tool, non-dict args, missing, unexpected, wrong type
        if name not in self._tools:
            return [f"unknown tool '{name}'"]
        if not isinstance(args, dict):
            return ["arguments must be an object"]
        spec, errors = self._tools[name], []
        for req in spec["required"]:
            if req not in args:
                errors.append(f"missing required argument '{req}'")
        for key, value in args.items():
            if key not in spec["params"]:
                errors.append(f"unexpected argument '{key}'")
            elif not type_ok(value, spec["params"][key]):
                want = TYPE_NAMES.get(spec["params"][key], spec["params"][key].__name__)
                errors.append(f"argument '{key}' should be {want}")
        return errors
        # <<< fullsend

    def call(self, name, args):
        """Validate, run, and wrap the outcome. Never raises."""
        # >>> fullsend: errors as data, results as data
        errors = self.validate(name, args)
        if errors:
            return {"ok": False, "error": "; ".join(errors)}
        try:
            return {"ok": True, "result": self._tools[name]["fn"](**args)}
        except Exception as exc:
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        # <<< fullsend

    def describe(self):
        """[{"name", "description", "params": {arg: type word}, "required"}], sorted by name."""
        # >>> fullsend: one spec per tool, types as schema words
        out = []
        for name in self.names():
            spec = self._tools[name]
            out.append({
                "name": name,
                "description": spec["description"],
                "params": {k: TYPE_NAMES.get(t, t.__name__) for k, t in spec["params"].items()},
                "required": list(spec["required"]),
            })
        return out
        # <<< fullsend
