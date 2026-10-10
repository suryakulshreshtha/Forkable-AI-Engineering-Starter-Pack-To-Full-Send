"""The judge for drill 8.3."""

CHATTY = 'Sure! Here is the data you asked for:\n{"name": "Ada", "langs": ["en", "fr"]}\nLet me know if you need more.'

FENCED = """Of course.

```json
{"ok": true, "items": [1, 2, 3]}
```

Anything else?"""


def test_plain_json(m):
    assert m.extract_json('{"a": 1}') == {"a": 1}
    assert m.extract_json("[1, 2, 3]") == [1, 2, 3]


def test_chatty_preamble_and_epilogue(m):
    assert m.extract_json(CHATTY) == {"name": "Ada", "langs": ["en", "fr"]}


def test_fenced_block(m):
    assert m.strip_fences(FENCED).strip() == '{"ok": true, "items": [1, 2, 3]}'
    assert m.strip_fences("no fences here") == "no fences here"
    assert m.extract_json(FENCED) == {"ok": True, "items": [1, 2, 3]}


def test_brackets_inside_strings_do_not_count(m):
    text = 'result: {"text": "a } tricky { string ]", "n": 2} trailing'
    assert m.find_json_span(text) == '{"text": "a } tricky { string ]", "n": 2}'


def test_escaped_quotes_inside_strings(m):
    text = 'x {"quote": "she said \\"hi}\\" twice", "k": 1} y'
    assert m.extract_json(text) == {"quote": 'she said "hi}" twice', "k": 1}


def test_trailing_commas_are_repaired(m):
    assert m.extract_json('{"a": [1, 2,], "b": 3,}') == {"a": [1, 2], "b": 3}
    assert m.extract_json('[\n  {"x": 1},\n]') == [{"x": 1}]


def test_commas_inside_strings_survive_the_repair(m):
    assert m.remove_trailing_commas('{"s": "x,}", }') == '{"s": "x,}" }'
    assert m.extract_json('{"s": "x,}", }') == {"s": "x,}"}


def test_nothing_usable_raises_value_error(m):
    for text in ["no json at all", '{"a": 1', '{"a": [1}', "{'single': 'quotes'}"]:
        try:
            m.extract_json(text)
        except ValueError:
            continue
        raise AssertionError(f"extract_json({text!r}) should raise ValueError")


def test_schema_validation(m):
    schema = {"name": str, "age": int, "score": float, "nickname?": str}
    assert m.validate({"name": "Ada", "age": 36, "score": 9}, schema) == []
    errors = m.validate({"name": "Ada", "age": True, "nickname": 7}, schema)
    assert sorted(errors) == sorted([
        "field 'age' should be int",
        "missing field 'score'",
        "field 'nickname' should be str",
    ]), f"got {errors}"
    assert m.validate(["not", "an", "object"], schema) == ["expected an object"]
