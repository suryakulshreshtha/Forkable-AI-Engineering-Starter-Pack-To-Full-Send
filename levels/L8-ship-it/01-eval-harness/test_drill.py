"""The judge for drill 8.1."""

CASES_SPEC = [
    ("capital", "capital of France?", "Paris", "contains"),
    ("math", "2+2", "4", "exact"),
    ("date", "today's date?", r"\d{4}-\d{2}-\d{2}", "regex"),
    ("greet", "say hi", "hello", "contains"),
]


def _cases(m):
    return [m.Case(*spec) for spec in CASES_SPEC]


def old_model(prompt):
    return {"capital of France?": "It is Paris.", "2+2": " 4 ", "today's date?": "no idea",
            "say hi": "Hello there"}[prompt]


def new_model(prompt):
    return {"capital of France?": "Paris, of course", "2+2": "four", "today's date?": "2026-10-10",
            "say hi": "HELLO!"}[prompt]


def test_scoring_modes(m):
    assert m.score("  Paris ", m.Case("a", "", "Paris")) is True, "exact ignores surrounding whitespace"
    assert m.score("paris", m.Case("a", "", "Paris")) is False, "exact is case-sensitive"
    assert m.score("I think PARIS.", m.Case("b", "", "paris", "contains")) is True
    assert m.score("v1.2.3", m.Case("c", "", r"v\d+\.\d+", "regex")) is True
    assert m.score("version one", m.Case("c", "", r"v\d+", "regex")) is False


def test_unknown_mode_is_a_test_set_bug(m):
    try:
        m.score("x", m.Case("d", "", "x", "vibes"))
    except ValueError:
        return
    raise AssertionError("an unknown mode should raise ValueError")


def test_run_eval_tallies(m):
    report = m.run_eval(old_model, _cases(m))
    assert report["results"] == {"capital": True, "math": True, "date": False, "greet": True}
    assert (report["passed"], report["total"]) == (3, 4) and report["rate"] == 0.75
    assert report["errors"] == {}


def test_crashes_count_as_failures_and_are_recorded(m):
    def flaky(prompt):
        if prompt == "2+2":
            raise TimeoutError("model took too long")
        return old_model(prompt)

    report = m.run_eval(flaky, _cases(m))
    assert report["results"]["math"] is False and report["passed"] == 2
    assert report["errors"] == {"math": "TimeoutError: model took too long"}


def test_non_string_output_fails_the_case(m):
    report = m.run_eval(lambda p: None, [m.Case("x", "q", "a")])
    assert report["results"] == {"x": False} and "x" in report["errors"]


def test_duplicate_ids_are_rejected(m):
    try:
        m.run_eval(old_model, [m.Case("a", "2+2", "4"), m.Case("a", "2+2", "4")])
    except ValueError:
        return
    raise AssertionError("two cases with the same id should raise ValueError")


def test_a_better_average_can_hide_a_regression(m):
    old = m.run_eval(old_model, _cases(m))
    new = m.run_eval(new_model, _cases(m))
    diff = m.compare(old, new)
    assert diff == {"fixed": ["date"], "broke": ["math"], "delta": 0.0}, f"got {diff}"
    assert m.gate(old, new) is False, "something that used to pass now fails"
    assert m.gate(old, new, max_broken=1) is True


def test_empty_case_list(m):
    report = m.run_eval(old_model, [])
    assert report["rate"] == 0.0 and report["total"] == 0
