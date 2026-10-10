"""A validating tool registry for agents. Unlocked by drill 7.1."""
from __future__ import annotations

TYPE_NAMES = {str: "string", int: "integer", float: "number", bool: "boolean", list: "array", dict: "object"}


def type_ok(value, expected):
    if expected is float:
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected is int:
        return isinstance(value, int) and not isinstance(value, bool)
    return isinstance(value, expected)


class ToolRegistry:
    def __init__(self):
        self._tools = {}

    def register(self, name, fn, params=None, required=None, description=""):
        if name in self._tools:
            raise ValueError(f"tool '{name}' is already registered")
        params = dict(params or {})
        required = list(params) if required is None else list(required)
        unknown = [r for r in required if r not in params]
        if unknown:
            raise ValueError(f"required names not in params: {unknown}")
        self._tools[name] = {"fn": fn, "params": params, "required": required, "description": description}

    def names(self):
        return sorted(self._tools)

    def validate(self, name, args):
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

    def call(self, name, args):
        errors = self.validate(name, args)
        if errors:
            return {"ok": False, "error": "; ".join(errors)}
        try:
            return {"ok": True, "result": self._tools[name]["fn"](**args)}
        except Exception as exc:
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    def describe(self):
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
