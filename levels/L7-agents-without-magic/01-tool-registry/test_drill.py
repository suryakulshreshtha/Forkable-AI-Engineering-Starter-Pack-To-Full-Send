"""The judge for drill 7.1."""


def _registry(m, log=None):
    reg = m.ToolRegistry()

    def add(a, b):
        if log is not None:
            log.append(("add", a, b))
        return a + b

    def shout(text, times=1):
        return (text.upper() + "!") * times

    def explode():
        raise RuntimeError("the tool caught fire")

    reg.register("add", add, {"a": float, "b": float}, description="Add two numbers")
    reg.register("shout", shout, {"text": str, "times": int}, required=["text"])
    reg.register("explode", explode, {})
    return reg


def test_names_are_sorted(m):
    assert _registry(m).names() == ["add", "explode", "shout"]


def test_bad_registrations_raise(m):
    reg = _registry(m)
    for attempt in (
        lambda: reg.register("add", lambda: 0, {}),
        lambda: reg.register("odd", lambda x: x, {"x": int}, required=["y"]),
    ):
        try:
            attempt()
        except ValueError:
            continue
        raise AssertionError("duplicate names and unknown required names must raise ValueError")


def test_valid_arguments_pass(m):
    reg = _registry(m)
    assert reg.validate("add", {"a": 1, "b": 2.5}) == [], "an int is a fine number"
    assert reg.validate("shout", {"text": "hi"}) == [], "optional arguments may be left out"


def test_each_kind_of_problem_is_reported(m):
    reg = _registry(m)
    assert reg.validate("nope", {}) == ["unknown tool 'nope'"]
    errors = reg.validate("add", {"a": "1", "c": 3})
    joined = " | ".join(errors)
    assert len(errors) == 3, f"expected 3 problems, got {errors}"
    assert "'b'" in joined and "'c'" in joined and "'a'" in joined
    assert reg.validate("add", [1, 2]) != [], "arguments must be a dict"


def test_booleans_are_not_numbers(m):
    reg = _registry(m)
    assert reg.validate("shout", {"text": "x", "times": True}) != [], "True is technically an int; reject it"
    assert reg.validate("add", {"a": False, "b": 1}) != []


def test_call_runs_valid_tools(m):
    assert _registry(m).call("add", {"a": 2, "b": 3}) == {"ok": True, "result": 5}
    assert _registry(m).call("shout", {"text": "go", "times": 2}) == {"ok": True, "result": "GO!GO!"}


def test_invalid_calls_never_run_the_tool(m):
    log = []
    result = _registry(m, log).call("add", {"a": 1})
    assert result["ok"] is False and "'b'" in result["error"]
    assert log == [], "validation failed, so the tool must not run"


def test_tool_crashes_become_errors(m):
    result = _registry(m).call("explode", {})
    assert result == {"ok": False, "error": "RuntimeError: the tool caught fire"}


def test_describe_for_the_model(m):
    spec = _registry(m).describe()
    assert [s["name"] for s in spec] == ["add", "explode", "shout"]
    assert spec[0] == {
        "name": "add",
        "description": "Add two numbers",
        "params": {"a": "number", "b": "number"},
        "required": ["a", "b"],
    }
    assert spec[2]["params"] == {"text": "string", "times": "integer"} and spec[2]["required"] == ["text"]
