"""
Title: Ingest
XP: 30

## The job
Turn a folder of Markdown into chunks. `load_docs(root, suffixes)` returns
`{"path", "text"}` for every matching file, with paths relative to root in forward-slash
form, sorted, skipping hidden folders and `node_modules`. `split_sections(doc)` cuts a
document at Markdown headings (ignoring `#` lines inside fenced code) into
`{"path", "heading", "text"}` sections, dropping empty ones. `make_chunks(sections,
max_words, overlap)` windows each section's text and gives every chunk an id of the form
`path#n`, numbered from 1 within each file.

## Why it matters
Ingestion is where most retrieval systems quietly lose. Index a vendored dependency folder
and every answer cites it; lose the headings and chunks forget what they are about; give
chunks unstable ids and citations stop pointing anywhere. Clean, boring, deterministic
ingestion is what makes everything downstream debuggable.

## Hints
- `Path(root).rglob("*")` walks everything. Skip a file if any folder between root and the file starts with a dot or is named node_modules.
- `path.relative_to(root).as_posix()` gives forward slashes on every operating system.
- A heading line matches `#` to `######` followed by a space. Toggle an in_code flag on lines starting with three backticks and ignore headings while it is set.
- Text before the first heading belongs to a section whose heading is the empty string.
- `kit.retrieval.chunk_words` does the windowing. Keep one counter per path for the ids.
"""
from __future__ import annotations

import re
from pathlib import Path

from kit.retrieval import chunk_words

SKIP_DIRS = {"node_modules", "__pycache__", "venv"}
_HEADING = re.compile(r"^#{1,6}\s+(.*?)\s*#*\s*$")


def load_docs(root, suffixes=(".md",)):
    """[{"path": "docs/guide.md", "text": "..."}], sorted by path."""
    # >>> fullsend: walk, filter by suffix and folder, read as UTF-8
    root = Path(root)
    docs = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        parts = path.relative_to(root).parts[:-1]
        if any(p.startswith(".") or p in SKIP_DIRS for p in parts):
            continue
        docs.append({"path": path.relative_to(root).as_posix(), "text": path.read_text(encoding="utf-8")})
    docs.sort(key=lambda d: d["path"])
    return docs
    # <<< fullsend


def split_sections(doc):
    """Split one doc at headings. Returns [{"path", "heading", "text"}] with non-empty text."""
    # >>> fullsend: walk lines, start a new section at each heading outside code fences
    sections, heading, lines, in_code = [], "", [], False

    def flush():
        text = "\n".join(lines).strip()
        if text:
            sections.append({"path": doc["path"], "heading": heading, "text": text})

    for line in doc["text"].splitlines():
        if line.lstrip().startswith("```"):
            in_code = not in_code
            lines.append(line)
            continue
        match = None if in_code else _HEADING.match(line)
        if match:
            flush()
            heading, lines = match.group(1).strip(), []
        else:
            lines.append(line)
    flush()
    return sections
    # <<< fullsend


def make_chunks(sections, max_words=80, overlap=20):
    """[{"id": "path#n", "path", "heading", "text"}], n counting from 1 per path."""
    # >>> fullsend: window each section, number chunks per path
    chunks, counters = [], {}
    for section in sections:
        for piece in chunk_words(section["text"], max_words, overlap):
            n = counters.get(section["path"], 0) + 1
            counters[section["path"]] = n
            chunks.append({
                "id": f"{section['path']}#{n}",
                "path": section["path"],
                "heading": section["heading"],
                "text": piece,
            })
    return chunks
    # <<< fullsend
