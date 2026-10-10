"""The judge for drill 9.1."""
import tempfile
from pathlib import Path

GUIDE = """Intro text before any heading.

# Setup
Install Python. Then run the doctor.

## Running drills
Use the check command.

```bash
# this is a shell comment, not a heading
python fullsend.py check 1.1
```

## Empty section

### Final words
Ship it.
"""


def _tree():
    tmp = tempfile.TemporaryDirectory()
    root = Path(tmp.name)
    files = {
        "README.md": "# Hello\nWelcome.",
        "docs/guide.md": GUIDE,
        "docs/notes.txt": "not markdown",
        ".git/HEAD.md": "# hidden",
        "node_modules/pkg/README.md": "# vendored",
        "docs/.drafts/secret.md": "# draft",
    }
    for rel, text in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return tmp, root


def test_load_docs_finds_markdown_and_skips_junk(m):
    tmp, root = _tree()
    with tmp:
        docs = m.load_docs(root)
        assert [d["path"] for d in docs] == ["README.md", "docs/guide.md"], f"got {[d['path'] for d in docs]}"
        assert docs[0]["text"] == "# Hello\nWelcome."


def test_load_docs_accepts_other_suffixes(m):
    tmp, root = _tree()
    with tmp:
        paths = [d["path"] for d in m.load_docs(root, suffixes=(".txt",))]
        assert paths == ["docs/notes.txt"]


def test_sections_split_at_headings(m):
    sections = m.split_sections({"path": "docs/guide.md", "text": GUIDE})
    headings = [s["heading"] for s in sections]
    assert headings == ["", "Setup", "Running drills", "Final words"], f"got {headings}"
    assert sections[0]["text"] == "Intro text before any heading."
    assert all(s["path"] == "docs/guide.md" for s in sections)


def test_code_fences_hide_fake_headings(m):
    sections = m.split_sections({"path": "g.md", "text": GUIDE})
    running = next(s for s in sections if s["heading"] == "Running drills")
    assert "# this is a shell comment" in running["text"], "a # line inside a code fence is not a heading"


def test_chunks_have_stable_per_file_ids(m):
    sections = [
        {"path": "a.md", "heading": "One", "text": " ".join(f"w{i}" for i in range(10))},
        {"path": "a.md", "heading": "Two", "text": "short section"},
        {"path": "b.md", "heading": "", "text": "other file"},
    ]
    chunks = m.make_chunks(sections, max_words=4, overlap=1)
    assert [c["id"] for c in chunks] == ["a.md#1", "a.md#2", "a.md#3", "a.md#4", "b.md#1"]
    assert chunks[0]["text"] == "w0 w1 w2 w3" and chunks[1]["text"] == "w3 w4 w5 w6"
    assert chunks[3] == {"id": "a.md#4", "path": "a.md", "heading": "Two", "text": "short section"}


def test_the_whole_pipeline(m):
    tmp, root = _tree()
    with tmp:
        sections = [s for d in m.load_docs(root) for s in m.split_sections(d)]
        chunks = m.make_chunks(sections, max_words=50, overlap=10)
        assert {c["path"] for c in chunks} == {"README.md", "docs/guide.md"}
        assert len({c["id"] for c in chunks}) == len(chunks), "ids must be unique"
