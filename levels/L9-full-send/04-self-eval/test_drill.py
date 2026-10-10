"""The judge for drill 9.4."""

ANSWERS = {
    "How do I sync my fork?": {"answer": "Add the upstream remote, then merge upstream main. [1]",
                               "citations": ["FORK-ME.md#2"]},
    "What license is used?": {"answer": "The project uses the MIT license. [1]",
                              "citations": ["README.md#9"]},
    "Who won the 1998 World Cup?": {"answer": "I don't know. These docs do not cover that.", "citations": []},
    "What is the tallest mountain?": {"answer": "Everest, probably. [1]", "citations": ["kit/README.md#1"]},
}

GOLDEN = [
    {"q": "How do I sync my fork?", "expect_path": "FORK-ME.md", "expect_words": ["upstream"]},
    {"q": "What license is used?", "expect_path": "LICENSE.md", "expect_words": ["MIT"]},
    {"q": "Who won the 1998 World Cup?", "unanswerable": True},
    {"q": "What is the tallest mountain?", "unanswerable": True},
]


def ask(question):
    return ANSWERS[question]


def test_check_case_passes_a_good_answer(m):
    assert m.check_case(ANSWERS["How do I sync my fork?"], GOLDEN[0]) == (True, "")


def test_check_case_explains_failures(m):
    passed, why = m.check_case(ANSWERS["What license is used?"], GOLDEN[1])
    assert passed is False and why == "no citation from LICENSE.md", f"got {why!r}"
    passed, why = m.check_case({"answer": "Use git.", "citations": ["FORK-ME.md#1"]}, GOLDEN[0])
    assert passed is False and why == "missing words: upstream"


def test_word_matching_ignores_case(m):
    case = {"q": "x", "expect_words": ["MIT", "License"]}
    assert m.check_case({"answer": "mit license", "citations": []}, case) == (True, "")


def test_citation_paths_can_contain_hashes_in_folders(m):
    case = {"q": "x", "expect_path": "notes/c#/intro.md"}
    assert m.check_case({"answer": "", "citations": ["notes/c#/intro.md#4"]}, case)[0] is True


def test_refusals(m):
    assert m.check_case(ANSWERS["Who won the 1998 World Cup?"], GOLDEN[2]) == (True, "")
    assert m.check_case(ANSWERS["What is the tallest mountain?"], GOLDEN[3]) == (False, "should have refused")


def test_evaluate_report(m):
    report = m.evaluate(ask, GOLDEN)
    assert (report["total"], report["passed"], report["rate"]) == (4, 2, 0.5)
    assert [c["passed"] for c in report["cases"]] == [True, False, True, False]
    assert report["cases"][3] == {"q": "What is the tallest mountain?", "passed": False, "why": "should have refused"}


def test_empty_golden_set(m):
    assert m.evaluate(ask, []) == {"total": 0, "passed": 0, "rate": 0.0, "cases": []}


def test_markdown_table(m):
    text = m.to_markdown(m.evaluate(ask, GOLDEN))
    lines = text.splitlines()
    assert "2 / 4" in lines[0]
    table = [l for l in lines if l.startswith("|")]
    assert len(table) == 2 + 4, "a header, a separator and one row per case"
    assert "should have refused" in text and "no citation from LICENSE.md" in text
