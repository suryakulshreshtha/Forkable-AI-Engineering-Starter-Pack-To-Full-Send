#!/usr/bin/env python3
"""PocketRAG: ask this repository a question, answered by YOUR Level 9 code.

    python levels/L9-full-send/app.py "how do I pull new drills into my fork?"
    python levels/L9-full-send/app.py --eval          # grade against golden.json
    python levels/L9-full-send/app.py --eval --min-score 0.9   # exit 1 if fewer than 90% pass (for CI)
    python levels/L9-full-send/app.py --reference ... # use reference.py instead of your drill.py

The pipeline: 9.1 ingest -> 9.2 retrieve -> 9.3 cited answers, graded by 9.4 self-eval.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kit.text import content_words  # noqa: E402

PARTS = {
    "ingest": "01-ingest",
    "retrieve": "02-retrieve",
    "answer": "03-cited-answers",
    "selfeval": "04-self-eval",
}
SKIP = {"PROGRESS.md"}


def load(part, use_reference):
    path = HERE / PARTS[part] / ("reference.py" if use_reference else "drill.py")
    if not path.exists():
        raise SystemExit(f"{path.relative_to(ROOT)} is missing. Run: python fullsend.py init")
    spec = importlib.util.spec_from_file_location(f"pocketrag_{part}", str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_FENCED = re.compile(r"^```.*?^```[ \t]*$", re.DOTALL | re.MULTILINE)
_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")


def clean_markdown(text):
    """Make Markdown read like prose: drop code blocks and comments, flatten links, tables and bullets.

    This is app-level polish, separate from the drills: answers quote sentences, so
    sentences should not be full of table pipes and code fences.
    """
    text = _COMMENT.sub("", _FENCED.sub("", text))
    text = _LINK.sub(r"\1", text).replace("**", "")
    out = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("|"):
            cells = [c.strip() for c in stripped.strip("|").split("|")]
            if all(set(c) <= set("-: ") for c in cells):
                continue
            stripped = ": ".join(c for c in cells if c)
        elif re.match(r"^([-*]|\d+\.)\s+", stripped):
            stripped = re.sub(r"^([-*]|\d+\.)\s+", "", stripped)
        else:
            out.append(line)
            continue
        if stripped and stripped[-1] not in ".!?:":
            stripped += "."
        out.append(stripped)
    return "\n".join(out)


def build(use_reference=False, max_words=60, overlap=15, k=4):
    ingest = load("ingest", use_reference)
    docs = [d for d in ingest.load_docs(ROOT) if d["path"] not in SKIP]
    for doc in docs:
        doc["text"] = clean_markdown(doc["text"])
    sections = [s for d in docs for s in ingest.split_sections(d)]
    chunks = ingest.make_chunks(sections, max_words, overlap)
    index = load("retrieve", use_reference).Index(chunks)
    answerer = load("answer", use_reference)

    def ask(question):
        hits = index.search(question, k=k, max_per_path=2)
        need = min(2, len(set(content_words(question))))
        return answerer.answer(question, hits, max_sentences=2, min_overlap=max(1, need))

    return ask, len(docs), len(chunks)


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):  # a legacy Windows console cannot show every character
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    parser = argparse.ArgumentParser(description="Ask this repo a question with your PocketRAG.")
    parser.add_argument("question", nargs="*", help="your question")
    parser.add_argument("--eval", action="store_true", help="grade against golden.json")
    parser.add_argument("--reference", action="store_true", help="use the reference implementations")
    parser.add_argument("--min-score", type=float, metavar="RATE",
                        help="with --eval: exit 1 if the pass rate is below RATE (0 to 1), e.g. 0.9")
    parser.add_argument("--max-words", type=int, default=60, help="chunk size in words (default 60)")
    parser.add_argument("--overlap", type=int, default=15, help="words shared by neighbouring chunks (default 15)")
    parser.add_argument("-k", type=int, default=4, help="chunks retrieved per question (default 4)")
    args = parser.parse_args(argv)
    if args.min_score is not None and not 0.0 <= args.min_score <= 1.0:
        parser.error("--min-score must be between 0 and 1")
    if args.min_score is not None and not args.eval:
        parser.error("--min-score only makes sense with --eval")
    if args.max_words < 1 or args.k < 1 or not 0 <= args.overlap < args.max_words:
        parser.error("need --max-words >= 1, -k >= 1 and 0 <= --overlap < --max-words")

    try:
        ask, n_docs, n_chunks = build(args.reference, args.max_words, args.overlap, args.k)
        if args.eval:
            selfeval = load("selfeval", args.reference)
            golden = json.loads((HERE / "golden.json").read_text(encoding="utf-8"))
            report = selfeval.evaluate(ask, golden)
            print(selfeval.to_markdown(report))
            if args.min_score is not None and report["rate"] + 1e-9 < args.min_score:
                print(f"\neval gate failed: {report['rate']:.0%} passed, below the required {args.min_score:.0%}")
                return 1
            return 0
        question = " ".join(args.question).strip()
        if not question:
            parser.print_help()
            return 0
        result = ask(question)
    except NotImplementedError as exc:
        print(f"Your Level 9 code has an unfinished TODO: {exc}")
        print("Clear drills 9.1 to 9.3 (python fullsend.py next), or try --reference to see it working.")
        return 1

    print(result["answer"])
    if result["citations"]:
        print("\nSources:")
        for n, cid in enumerate(result["citations"], 1):
            print(f"  [{n}] {cid}")
    print(f"\n(searched {n_chunks} chunks from {n_docs} Markdown files)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
