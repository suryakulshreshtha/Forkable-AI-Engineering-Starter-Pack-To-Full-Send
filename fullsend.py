#!/usr/bin/env python3
"""fullsend: the one-file runner for Forkable AI Engineering, Starter Pack to Full Send.

Learner commands
  python fullsend.py init                 create your drill.py files (never overwrites)
  python fullsend.py next                 show the next drill you have not cleared
  python fullsend.py check <drill>        run the judge against your drill.py
  python fullsend.py hint <drill> [n|all] reveal hints one at a time
  python fullsend.py status               the board: every drill, cleared or not
  python fullsend.py card                 write PROGRESS.md for your fork
  python fullsend.py reset <drill>        back up your drill.py and start it fresh
  python fullsend.py doctor               check your setup

Maintainer commands
  python fullsend.py build [--check]      regenerate READMEs from reference.py docstrings
  python fullsend.py verify               references pass, stubs fail cleanly, docs fresh
  python fullsend.py new <level> <slug>   scaffold a new drill

A drill can be named as 1.2, L1/02, its slug (slope-finder) or its folder path.
Standard library only. Python 3.9+.
"""
from __future__ import annotations

import _thread
import argparse
import ast
import difflib
import importlib.util
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

__version__ = "1.0.0"

ROOT = Path(__file__).resolve().parent
LEVELS_DIR = ROOT / "levels"
MIN_PY = (3, 9)
UPSTREAM_SLUG = "suryakulshreshtha/Forkable-AI-Engineering-Starter-Pack-To-Full-Send"
UPSTREAM_URL = f"https://github.com/{UPSTREAM_SLUG}.git"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True  # keep drill folders free of __pycache__

LEVEL_DIR = re.compile(r"^L(\d+)-([a-z0-9-]+)$")
DRILL_DIR = re.compile(r"^(\d+)-([a-z0-9-]+)$")
START = re.compile(r"^(?P<indent>[ \t]*)# >>> fullsend(?::\s*(?P<label>.*?))?\s*$")
END = re.compile(r"^[ \t]*# <<< fullsend\s*$")
BOARD_START = "<!-- board:start -->"
BOARD_END = "<!-- board:end -->"

# --------------------------------------------------------------------------- output


def _can_encode(sample: str) -> bool:
    try:
        sample.encode(getattr(sys.stdout, "encoding", None) or "ascii")
        return True
    except (UnicodeEncodeError, LookupError):
        return False


if os.name == "nt":  # switch on ANSI escape handling in modern Windows consoles
    os.system("")

UNICODE = _can_encode("✔✘·▸█░")
SYM_PASS = "✔" if UNICODE else "+"
SYM_FAIL = "✘" if UNICODE else "x"
SYM_TODO = "·" if UNICODE else "."
SYM_ARROW = "▸" if UNICODE else ">"
BAR_FULL, BAR_EMPTY = ("█", "░") if UNICODE else ("#", "-")
COLOR = sys.stdout.isatty() and "NO_COLOR" not in os.environ


def _paint(code: str):
    return lambda text: f"\033[{code}m{text}\033[0m" if COLOR else str(text)


green, red, yellow, cyan, dim, bold = (_paint(c) for c in ("32", "31", "33", "36", "2", "1"))


def bar(done: int, total: int, width: int = 20) -> str:
    filled = 0 if total == 0 else round(width * done / total)
    return BAR_FULL * filled + BAR_EMPTY * (width - filled)


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


# --------------------------------------------------------------------------- model


@dataclass
class Drill:
    level: int
    index: int
    slug: str
    path: Path
    meta: dict
    body: str
    hints: list
    level_title: str = ""

    @property
    def key(self) -> str:
        return f"{self.level}.{self.index}"

    @property
    def title(self) -> str:
        return self.meta.get("Title", self.slug.replace("-", " ").title())

    @property
    def xp(self) -> int:
        try:
            return int(self.meta.get("XP", "10"))
        except ValueError:
            return 10

    @property
    def reference(self) -> Path:
        return self.path / "reference.py"

    @property
    def tests(self) -> Path:
        return self.path / "test_drill.py"

    @property
    def mine(self) -> Path:
        return self.path / "drill.py"


@dataclass
class Level:
    number: int
    slug: str
    path: Path
    title: str
    blurb: str
    drills: list = field(default_factory=list)

    @property
    def xp(self) -> int:
        return sum(d.xp for d in self.drills)


def parse_docstring(source: str, where: Path):
    """Split a reference.py docstring into (meta, body, hints)."""
    try:
        doc = ast.get_docstring(ast.parse(source), clean=True)
    except SyntaxError as exc:
        raise SystemExit(f"{rel(where)}: syntax error: {exc}")
    if not doc:
        raise SystemExit(f"{rel(where)}: reference.py needs a module docstring")
    lines = doc.splitlines()
    meta, i = {}, 0
    while i < len(lines) and lines[i].strip():
        m = re.match(r"^([A-Za-z][\w ]*):\s*(.+)$", lines[i])
        if not m:
            break
        meta[m.group(1).strip()] = m.group(2).strip()
        i += 1
    rest = "\n".join(lines[i:]).strip()
    parts = re.split(r"(?m)^## Hints\s*$", rest, maxsplit=1)
    body, hints = parts[0].strip(), []
    if len(parts) == 2:
        for line in parts[1].splitlines():
            m = re.match(r"^\s*[-*]\s+(.*)$", line)
            if m:
                hints.append(m.group(1).strip())
            elif hints and line.strip():
                hints[-1] += " " + line.strip()
    return meta, body, hints


def discover() -> list:
    levels = []
    if not LEVELS_DIR.is_dir():
        raise SystemExit(f"no levels/ folder next to {rel(Path(__file__))}")
    candidates = [p for p in LEVELS_DIR.iterdir() if p.is_dir() and LEVEL_DIR.match(p.name)]
    for ldir in sorted(candidates, key=lambda p: int(LEVEL_DIR.match(p.name).group(1))):
        lm = LEVEL_DIR.match(ldir.name)
        number = int(lm.group(1))
        title, blurb = lm.group(2).replace("-", " ").title(), ""
        intro = ldir / "_intro.md"
        if intro.exists():
            text = intro.read_text(encoding="utf-8").strip().splitlines()
            if text and text[0].startswith("# "):
                title = text[0][2:].strip()
                text = text[1:]
            blurb = "\n".join(text).strip()
        level = Level(number, lm.group(2), ldir, title, blurb)
        for ddir in sorted(p for p in ldir.iterdir() if p.is_dir() and DRILL_DIR.match(p.name)):
            ref = ddir / "reference.py"
            if not ref.exists():
                continue
            dm = DRILL_DIR.match(ddir.name)
            meta, body, hints = parse_docstring(ref.read_text(encoding="utf-8"), ref)
            level.drills.append(
                Drill(number, int(dm.group(1)), dm.group(2), ddir, meta, body, hints, title)
            )
        levels.append(level)
    return levels


def all_drills(levels) -> list:
    return [d for level in levels for d in level.drills]


def find_drill(levels, spec: str) -> Drill:
    drills = all_drills(levels)
    spec = spec.strip().rstrip("/\\")
    as_path = Path(spec)
    if not as_path.is_absolute():
        as_path = Path.cwd() / as_path
    if as_path.exists():
        target = as_path.resolve()
        for d in drills:
            dp = d.path.resolve()
            if target == dp or dp in target.parents:
                return d
    m = re.match(r"^[Ll]?(\d+)\s*[./\\\-_: ]+\s*(\d+)$", spec)
    if m:
        key = f"{int(m.group(1))}.{int(m.group(2))}"
        for d in drills:
            if d.key == key:
                return d
    for d in drills:
        if spec.lower() in (d.slug, d.path.name):
            return d
    names = [d.slug for d in drills] + [d.key for d in drills]
    close = difflib.get_close_matches(spec, names, n=3)
    hint = f" Did you mean: {', '.join(close)}?" if close else " Try: python fullsend.py status"
    raise SystemExit(f"No drill called '{spec}'.{hint}")


# --------------------------------------------------------------------------- stubs


def make_stub(drill: Drill) -> str:
    """Turn reference.py into a learner stub: marked blocks become TODOs."""
    source = drill.reference.read_text(encoding="utf-8")
    tree = ast.parse(source)
    lines = source.splitlines()
    first = tree.body[0] if tree.body else None
    if not (isinstance(first, ast.Expr) and isinstance(getattr(first, "value", None), ast.Constant)
            and isinstance(first.value.value, str)):
        raise SystemExit(f"{rel(drill.reference)}: docstring must be the first statement")
    doc_lines = [
        f'"""{drill.key} · {drill.title}',
        "",
        *drill.body.replace("\\", "\\\\").splitlines(),
        "",
        f"Check your work:  python fullsend.py check {drill.key}",
        f"Stuck?           python fullsend.py hint {drill.key}",
        '"""',
    ]
    lines[first.lineno - 1 : first.end_lineno] = doc_lines

    out, inside, blocks = [], False, 0
    for number, line in enumerate(lines, 1):
        if inside:
            if START.match(line):
                raise SystemExit(f"{rel(drill.reference)}:{number}: nested fullsend marker")
            if END.match(line):
                inside = False
            continue
        m = START.match(line)
        if m:
            indent, label = m.group("indent"), (m.group("label") or "your code here").strip()
            out.append(f"{indent}# TODO({drill.key}): {label}")
            out.append(f"{indent}raise NotImplementedError({label!r})")
            inside, blocks = True, blocks + 1
            continue
        if END.match(line):
            raise SystemExit(f"{rel(drill.reference)}:{number}: '# <<< fullsend' without a start")
        out.append(line)
    if inside:
        raise SystemExit(f"{rel(drill.reference)}: a '# >>> fullsend' block is never closed")
    if blocks == 0:
        raise SystemExit(f"{rel(drill.reference)}: no '# >>> fullsend' blocks to turn into TODOs")
    header = [
        "# YOUR DRILL. Edit freely: upstream never ships or changes drill.py, so pulling",
        "# updates into your fork will not collide with your work.",
    ]
    return "\n".join(header + out).rstrip() + "\n"


# --------------------------------------------------------------------------- judge


class _Timeout(BaseException):
    """BaseException so a learner's broad `except Exception` cannot swallow it."""


_USE_SIGALRM = hasattr(signal, "SIGALRM")  # macOS and Linux; Windows uses the timer fallback


@contextmanager
def _time_limit(seconds: float):
    """Raise _Timeout in the main thread if the body runs longer than `seconds`."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    if _USE_SIGALRM:
        def _raise(signum, frame):
            raise _Timeout()

        previous = signal.signal(signal.SIGALRM, _raise)
        signal.setitimer(signal.ITIMER_REAL, seconds)
        try:
            yield
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous)
        return

    # Windows has no SIGALRM. A timer thread interrupts the main thread instead, and
    # the flag tells that interrupt apart from a real Ctrl+C.
    fired = threading.Event()

    def _fire():
        fired.set()
        _thread.interrupt_main()

    timer = threading.Timer(seconds, _fire)
    timer.daemon = True
    timer.start()
    try:
        yield
    except KeyboardInterrupt:
        if fired.is_set():
            raise _Timeout() from None
        raise
    finally:
        timer.cancel()


@dataclass
class Outcome:
    name: str
    status: str  # pass | fail | todo | error
    detail: str = ""


_load_counter = 0


def _load(path: Path, prefix: str):
    global _load_counter
    _load_counter += 1
    name = f"_fullsend_{prefix}_{_load_counter}"
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def _where(exc: BaseException, path: Path) -> str:
    frames = [f for f in traceback.extract_tb(exc.__traceback__) if Path(f.filename) == path]
    if not frames:
        return ""
    f = frames[-1]
    return f" (line {f.lineno}: {f.line.strip()})" if f.line else f" (line {f.lineno})"


def judge(drill: Drill, target: Path, timeout: float = 5.0) -> list:
    """Run every test_ function in the drill's judge against the module at `target`.

    After one test hangs, the remaining tests get a much shorter limit, so a single
    infinite loop costs seconds rather than a minute.
    """
    tests_module = _load(drill.tests, "tests")
    tests = [(n, f) for n, f in vars(tests_module).items() if n.startswith("test_") and callable(f)]
    try:
        with _time_limit(timeout):
            module = _load(target, "target")
    except NotImplementedError as exc:
        return [Outcome(n, "todo", str(exc)) for n, _ in tests]
    except _Timeout:
        return [Outcome(n, "error", "importing your file took too long") for n, _ in tests]
    except Exception as exc:  # syntax errors, missing imports, ...
        why = f"could not import {target.name}: {type(exc).__name__}: {exc}"
        return [Outcome(n, "error", why) for n, _ in tests]

    outcomes, limit = [], timeout
    for name, fn in tests:
        try:
            with _time_limit(limit):
                fn(module)
            outcomes.append(Outcome(name, "pass"))
        except NotImplementedError as exc:
            outcomes.append(Outcome(name, "todo", str(exc) or "not implemented yet"))
        except AssertionError as exc:
            detail = str(exc) or "assertion failed" + _where(exc, drill.tests)
            outcomes.append(Outcome(name, "fail", detail))
        except _Timeout:
            outcomes.append(Outcome(name, "error", f"no answer after {limit:g}s: infinite loop?"))
            limit = min(timeout, 1.0)
        except Exception as exc:
            detail = f"{type(exc).__name__}: {exc}{_where(exc, target)}"
            outcomes.append(Outcome(name, "error", detail))
    return outcomes


def cleared(outcomes) -> bool:
    return bool(outcomes) and all(o.status == "pass" for o in outcomes)


def drill_state(drill: Drill):
    """Return ('new'|'cleared'|'started'|'untouched', passed, total)."""
    if not drill.mine.exists():
        return "new", 0, 0
    outcomes = judge(drill, drill.mine)
    passed = sum(o.status == "pass" for o in outcomes)
    if cleared(outcomes):
        return "cleared", passed, len(outcomes)
    # Some drills hand you working pieces, so a fresh stub can pass a test or two.
    # "Untouched" means the file still matches the stub, whatever the judge says.
    try:
        pristine = drill.mine.read_text(encoding="utf-8").replace("\r\n", "\n") == make_stub(drill)
    except (OSError, UnicodeDecodeError):
        pristine = False
    return ("untouched" if pristine else "started"), passed, len(outcomes)


def pretty_test_name(name: str) -> str:
    return name[len("test_"):].replace("_", " ") if name.startswith("test_") else name


def print_outcomes(outcomes) -> None:
    for o in outcomes:
        label = pretty_test_name(o.name)
        if o.status == "pass":
            print(f"  {green(SYM_PASS)} {label}")
        elif o.status == "todo":
            print(f"  {dim(SYM_TODO)} {label} {dim('- TODO: ' + o.detail)}")
        else:
            print(f"  {red(SYM_FAIL)} {label} {dim('- ' + o.detail)}")


# --------------------------------------------------------------------------- learner commands


def cmd_init(args) -> int:
    levels = discover()
    made = kept = 0
    for d in all_drills(levels):
        if d.mine.exists():
            kept += 1
            continue
        d.mine.write_text(make_stub(d), encoding="utf-8")
        made += 1
    print(f"{green(SYM_PASS)} created {made} drill.py file(s); left {kept} existing one(s) alone.")
    if made:
        print(f"Next: {bold('python fullsend.py next')}")
    return 0


def first_unfinished(levels):
    for d in all_drills(levels):
        state, passed, total = drill_state(d)
        if state != "cleared":
            return d, state, passed, total
    return None, "done", 0, 0


def brief(drill: Drill) -> str:
    """The body up to the second section heading: enough to start work."""
    chunks = re.split(r"(?m)^## ", drill.body)
    text = chunks[0] + ("## " + chunks[1] if len(chunks) > 1 else "")
    return text.strip()


def cmd_next(args) -> int:
    levels = discover()
    d, state, passed, total = first_unfinished(levels)
    if d is None:
        print(green(bold("Every drill cleared. Full send achieved.")))
        print("Run python fullsend.py card, push your fork, then build something of your own.")
        return 0
    if state == "new":
        print(yellow("You have no drill.py files yet. Run: python fullsend.py init"))
        return 1
    print(f"{cyan(SYM_ARROW)} {bold(d.key + ' · ' + d.title)}  {dim(d.level_title)}  {dim('+' + str(d.xp) + ' XP')}")
    if total:
        print(dim(f"  {passed}/{total} tests passing so far"))
    print()
    print(brief(d))
    print()
    print(f"  edit   {rel(d.mine)}")
    print(f"  check  python fullsend.py check {d.key}")
    print(f"  hint   python fullsend.py hint {d.key}")
    return 0


def cmd_check(args) -> int:
    levels = discover()
    d = find_drill(levels, args.drill)
    target = d.reference if args.reference else d.mine
    if not target.exists():
        print(yellow(f"{rel(target)} does not exist yet. Run: python fullsend.py init"))
        return 1
    print(f"{cyan(SYM_ARROW)} {bold(d.key + ' · ' + d.title)}  {dim(rel(target))}")
    started = time.perf_counter()
    outcomes = judge(d, target)
    print_outcomes(outcomes)
    passed = sum(o.status == "pass" for o in outcomes)
    took = time.perf_counter() - started
    if cleared(outcomes):
        print(green(bold(f"\n  CLEARED  +{d.xp} XP")) + dim(f"  ({passed}/{len(outcomes)} in {took:.2f}s)"))
        if not args.reference:
            nxt = _after(levels, d)
            print(dim(f"  commit it:  git add {rel(d.mine)} && git commit -m \"clear {d.key}\""))
            if nxt:
                print(dim(f"  next up:    python fullsend.py check {nxt.key}  ({nxt.title})"))
        return 0
    print(f"\n  {passed}/{len(outcomes)} passing" + dim(f"  ·  stuck? python fullsend.py hint {d.key}"))
    return 1


def _after(levels, drill: Drill):
    drills = all_drills(levels)
    i = drills.index(drill)
    return drills[i + 1] if i + 1 < len(drills) else None


def cmd_hint(args) -> int:
    d = find_drill(discover(), args.drill)
    if not d.hints:
        print("This drill has no hints. The reference.py next to it is the last resort.")
        return 0
    which = args.n or "1"
    if which == "all":
        picks = list(enumerate(d.hints, 1))
    else:
        try:
            n = int(which)
        except ValueError:
            raise SystemExit("hint number must be an integer or 'all'")
        if not 1 <= n <= len(d.hints):
            raise SystemExit(f"{d.key} has hints 1 to {len(d.hints)}")
        picks = [(n, d.hints[n - 1])]
    for n, text in picks:
        print(f"{yellow('hint ' + str(n) + '/' + str(len(d.hints)))}  {text}")
    last = picks[-1][0]
    if last < len(d.hints):
        print(dim(f"more: python fullsend.py hint {d.key} {last + 1}"))
    else:
        print(dim(f"out of hints. Last resort: {rel(d.reference)}"))
    return 0


def collect_board(levels, reference: bool = False):
    rows = []
    for level in levels:
        states = []
        for d in level.drills:
            if reference:
                outcomes = judge(d, d.reference)
                ok = cleared(outcomes)
                states.append((d, "cleared" if ok else "started", sum(o.status == "pass" for o in outcomes), len(outcomes)))
            else:
                states.append((d, *drill_state(d)))
        rows.append((level, states))
    return rows


def cmd_status(args) -> int:
    levels = discover()
    rows = collect_board(levels, reference=args.reference)
    total_xp = sum(lv.xp for lv in levels)
    earned = sum(d.xp for _, states in rows for d, s, *_ in states if s == "cleared")
    n_cleared = sum(1 for _, states in rows for _, s, *_ in states if s == "cleared")
    n_all = sum(len(states) for _, states in rows)
    if args.summary:
        print_utf8(markdown_board(rows, earned, total_xp, n_cleared, n_all))
        return 0
    print(bold("FULL SEND BOARD") + dim("  (checking reference.py)" if args.reference else ""))
    for level, states in rows:
        done = sum(1 for _, s, *_ in states if s == "cleared")
        marks = " ".join(_mark(s) for _, s, *_ in states)
        print(f"  L{level.number} {level.title:<34} {bar(done, len(states), 10)} {marks}")
    print()
    print(f"  {bold(str(earned))}/{total_xp} XP  ·  {n_cleared}/{n_all} drills  {bar(n_cleared, n_all)}")
    if any(s == "new" for _, states in rows for _, s, *_ in states):
        print(yellow("  some drill.py files are missing: python fullsend.py init"))
    else:
        print(dim("  next: python fullsend.py next"))
    return 0


def _mark(state: str) -> str:
    return {
        "cleared": green(SYM_PASS),
        "started": yellow(SYM_FAIL),
        "untouched": dim(SYM_TODO),
        "new": dim("?"),
    }[state]


def _md_bar(done: int, total: int, width: int) -> str:
    """Markdown is always UTF-8, so it gets real block characters whatever the terminal supports."""
    filled = 0 if total == 0 else round(width * done / total)
    return "█" * filled + "░" * (width - filled)


def print_utf8(text: str) -> None:
    """Print text that contains emoji, even on a terminal whose encoding cannot show them."""
    try:
        print(text)
    except UnicodeEncodeError:
        sys.stdout.flush()
        sys.stdout.buffer.write((text + "\n").encode("utf-8"))
        sys.stdout.buffer.flush()


def markdown_board(rows, earned, total_xp, n_cleared, n_all) -> str:
    pct = 0 if n_all == 0 else round(100 * n_cleared / n_all)
    out = [
        f"**XP {earned} / {total_xp}** · **{n_cleared} / {n_all} drills cleared** · {pct}%",
        "",
        f"`{_md_bar(n_cleared, n_all, 30)}`",
        "",
        "| Level | Progress | Drills |",
        "| --- | --- | --- |",
    ]
    for level, states in rows:
        done = sum(1 for _, s, *_ in states if s == "cleared")
        cells = " ".join(("✅" if s == "cleared" else "🟨" if s == "started" else "⬜") for _, s, *_ in states)
        out.append(f"| L{level.number} · {level.title} | `{_md_bar(done, len(states), 10)}` {done}/{len(states)} | {cells} |")
    return "\n".join(out)


def cmd_card(args) -> int:
    levels = discover()
    rows = collect_board(levels)
    total_xp = sum(lv.xp for lv in levels)
    earned = sum(d.xp for _, states in rows for d, s, *_ in states if s == "cleared")
    n_cleared = sum(1 for _, states in rows for _, s, *_ in states if s == "cleared")
    n_all = sum(len(states) for _, states in rows)
    text = "\n".join([
        "# My Full Send progress",
        "",
        f"_Written by `python fullsend.py card` on {date.today().isoformat()}. "
        "Every ✅ means the judge passed against my own `drill.py`._",
        "",
        markdown_board(rows, earned, total_xp, n_cleared, n_all),
        "",
        "Legend: ✅ cleared · 🟨 in progress · ⬜ not started",
        "",
    ])
    out = ROOT / "PROGRESS.md"
    out.write_text(text, encoding="utf-8")
    print(f"{green(SYM_PASS)} wrote {rel(out)}: {earned}/{total_xp} XP, {n_cleared}/{n_all} drills")
    print(dim("  commit it so your fork shows where you are: git add PROGRESS.md && git commit -m card"))
    return 0


def cmd_reset(args) -> int:
    d = find_drill(discover(), args.drill)
    if d.mine.exists():
        backup = d.mine.with_name("drill.py.bak")
        shutil.copyfile(d.mine, backup)
        print(dim(f"backed up your work to {rel(backup)}"))
    d.mine.write_text(make_stub(d), encoding="utf-8")
    print(f"{green(SYM_PASS)} fresh {rel(d.mine)}")
    return 0


def _git(git: str, *cmd) -> str:
    try:
        done = subprocess.run([git, *cmd], cwd=ROOT, capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return ""
    return done.stdout.strip()


def cmd_doctor(args) -> int:
    problems = 0

    def line(ok: bool, text: str, fix: str = ""):
        nonlocal problems
        if not ok:
            problems += 1
        print(f"  {green(SYM_PASS) if ok else red(SYM_FAIL)} {text}" + (dim(f"  -> {fix}") if fix and not ok else ""))

    print(bold("fullsend doctor") + dim(f"  v{__version__}"))
    py = sys.version_info
    line(py >= MIN_PY, f"Python {py.major}.{py.minor}.{py.micro}", "install Python 3.9 or newer")
    git = shutil.which("git")
    in_repo = bool(git) and (ROOT / ".git").exists()
    origin = _git(git, "remote", "get-url", "origin") if in_repo else ""
    maintainer = UPSTREAM_SLUG.lower() in origin.lower()

    levels = discover()
    drills = all_drills(levels)
    line(bool(drills), f"{len(levels)} levels, {len(drills)} drills found")
    have = sum(d.mine.exists() for d in drills)
    if not maintainer:
        line(have == len(drills), f"{have}/{len(drills)} drill.py files present", "python fullsend.py init")
    try:
        probe = ROOT / ".fullsend-write-test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        line(True, "repository folder is writable")
    except OSError:
        line(False, "repository folder is writable", "clone somewhere you can write")
    line(git is not None, "git is installed", "install git to save progress in your fork")
    if maintainer:
        print(f"  {cyan(SYM_ARROW)} this clone points at the upstream repository (maintainer mode)")
        tracked = [p for p in _git(git, "ls-files").splitlines() if p.endswith("/drill.py")]
        line(not tracked, "no drill.py files are committed upstream",
             "git rm --cached <path> for each, then commit" if tracked else "")
        if have:
            print(dim(f"    {have} local drill.py file(s) exist here; never commit them upstream"))
    elif in_repo:
        remotes = _git(git, "remote").split()
        line("upstream" in remotes, "an 'upstream' remote is set (for pulling new drills)",
             f"git remote add upstream {UPSTREAM_URL}")
    print()
    print(green("All good. python fullsend.py next") if problems == 0 else yellow(f"{problems} thing(s) to fix above."))
    return 0 if problems == 0 else 1


# --------------------------------------------------------------------------- maintainer commands


def drill_readme(levels, d: Drill) -> str:
    drills = all_drills(levels)
    i = drills.index(d)
    level = next(lv for lv in levels if lv.number == d.level)
    nav = []
    if i > 0:
        p = drills[i - 1]
        nav.append(f"[← {p.key} {p.title}]({os.path.relpath(p.path, d.path).replace(os.sep, '/')}/)")
    nav.append("[level overview](../)")
    if i + 1 < len(drills):
        n = drills[i + 1]
        nav.append(f"[{n.key} {n.title} →]({os.path.relpath(n.path, d.path).replace(os.sep, '/')}/)")
    position = level.drills.index(d) + 1
    out = [
        "<!-- generated by `python fullsend.py build` from reference.py; edit the docstring there -->",
        f"# {d.key} · {d.title}",
        "",
        f"`Level {d.level}: {d.level_title}` · `+{d.xp} XP` · `drill {position} of {len(level.drills)}`",
        "",
        d.body,
        "",
        "## Run it",
        "",
        "```bash",
        f"python fullsend.py check {d.key}     # judge your drill.py",
        f"python fullsend.py hint {d.key}      # one hint at a time",
        "```",
        "",
    ]
    if d.hints:
        out += ["<details><summary>All hints (try the drill first)</summary>", ""]
        out += [f"{k}. {h}" for k, h in enumerate(d.hints, 1)]
        out += ["", "</details>", ""]
    out += [
        "| File | Role |",
        "| --- | --- |",
        "| `drill.py` | Yours. Created by `python fullsend.py init`. |",
        "| `test_drill.py` | The judge. Read it: the tests are the spec. |",
        "| `reference.py` | One worked answer. Peek after an honest attempt. |",
        "",
        " · ".join(nav),
        "",
    ]
    return "\n".join(out)


def level_readme(level: Level) -> str:
    out = [
        "<!-- generated by `python fullsend.py build` from _intro.md and the drills -->",
        f"# Level {level.number} · {level.title}",
        "",
        level.blurb,
        "",
        "| Drill | Title | XP |",
        "| --- | --- | --- |",
    ]
    for d in level.drills:
        out.append(f"| [{d.key}]({d.path.name}/) | {d.title} | {d.xp} |")
    out += ["", f"**Level total: {level.xp} XP**", "", "Start: `python fullsend.py check "
            f"{level.drills[0].key if level.drills else level.number}`", ""]
    return "\n".join(out)


def root_board(levels) -> str:
    total = sum(lv.xp for lv in levels)
    count = len(all_drills(levels))
    out = [
        BOARD_START,
        f"_{len(levels)} levels · {count} drills · {total} XP. Generated by `python fullsend.py build`._",
        "",
        "| Level | Drills | XP |",
        "| --- | --- | --- |",
    ]
    for lv in levels:
        names = " · ".join(f"[{d.key} {d.title}]({rel(d.path)}/)" for d in lv.drills)
        out.append(f"| [L{lv.number} · {lv.title}]({rel(lv.path)}/) | {names} | {lv.xp} |")
    out.append(BOARD_END)
    return "\n".join(out)


def planned_files(levels) -> dict:
    files = {}
    for lv in levels:
        files[lv.path / "README.md"] = level_readme(lv)
        for d in lv.drills:
            files[d.path / "README.md"] = drill_readme(levels, d)
    readme = ROOT / "README.md"
    if readme.exists():
        text = readme.read_text(encoding="utf-8")
        if BOARD_START in text and BOARD_END in text:
            head, rest = text.split(BOARD_START, 1)
            _, tail = rest.split(BOARD_END, 1)
            files[readme] = head + root_board(levels) + tail
    return files


def cmd_build(args) -> int:
    levels = discover()
    stale = []
    for path, content in planned_files(levels).items():
        current = path.read_text(encoding="utf-8") if path.exists() else None
        if current != content:
            stale.append(path)
            if not args.check:
                path.write_text(content, encoding="utf-8")
    if args.check:
        for p in stale:
            print(f"  {red(SYM_FAIL)} out of date: {rel(p)}")
        if stale:
            print(yellow("run: python fullsend.py build"))
        return 1 if stale else 0
    print(f"{green(SYM_PASS)} build: {len(stale)} file(s) updated")
    return 0


def cmd_verify(args) -> int:
    levels = discover()
    failures = warnings = 0
    print(bold("fullsend verify"))
    with tempfile.TemporaryDirectory() as tmp:
        for d in all_drills(levels):
            issues, notes = [], []
            src = d.reference.read_text(encoding="utf-8")
            doc = ast.get_docstring(ast.parse(src), clean=True) or ""
            if "Title" not in d.meta:
                issues.append("docstring has no 'Title:' line")
            if not d.hints:
                issues.append("docstring has no '## Hints' list")
            if "\\" in doc or '"""' in doc:
                issues.append("docstring must not contain backslashes or triple quotes")
            if not d.tests.exists():
                issues.append("missing test_drill.py")
            else:
                ref = judge(d, d.reference)
                for o in ref:
                    if o.status != "pass":
                        issues.append(f"reference fails {pretty_test_name(o.name)}: {o.detail}")
                if not ref:
                    issues.append("test_drill.py defines no test_ functions")
                try:
                    stub_path = Path(tmp) / f"stub_{d.level}_{d.index}" / "drill.py"
                    stub_path.parent.mkdir()
                    stub_path.write_text(make_stub(d), encoding="utf-8")
                    stub = judge(d, stub_path)
                    if any(o.status == "error" and o.detail.startswith("could not import") for o in stub):
                        issues.append(f"stub does not import cleanly: {stub[0].detail}")
                    elif cleared(stub):
                        issues.append("the untouched stub already passes every test")
                    free = [pretty_test_name(o.name) for o in stub if o.status == "pass"]
                    if free:
                        notes.append("passes on the empty stub: " + ", ".join(free))
                except SystemExit as exc:
                    issues.append(str(exc))
            status = green(SYM_PASS) if not issues else red(SYM_FAIL)
            print(f"  {status} {d.key:<5} {d.title}")
            for msg in issues:
                print(f"      {red(msg)}")
            for msg in notes:
                print(f"      {dim('note: ' + msg)}")
            failures += bool(issues)
            warnings += bool(notes)
    print(bold("docs"))
    build_args = argparse.Namespace(check=True)
    docs_stale = cmd_build(build_args)
    if not docs_stale:
        print(f"  {green(SYM_PASS)} generated READMEs are current")
    failures += docs_stale
    print()
    if failures:
        print(red(f"verify: {failures} problem(s)"))
        return 1
    print(green(f"verify: all {len(all_drills(levels))} drills healthy") + (dim(f"  ({warnings} with notes)") if warnings else ""))
    return 0


REFERENCE_TEMPLATE = '''"""
Title: {title}
XP: 15

## The job
Describe what the learner builds, in two to four sentences.

## Why it matters
One paragraph that connects this drill to real AI engineering work.

## Hints
- A gentle nudge.
- A stronger nudge.
- Nearly the answer.
"""
from __future__ import annotations


def solve(x):
    """Explain the contract: inputs, outputs, edge cases."""
    # >>> fullsend: return the answer
    return x
    # <<< fullsend
'''

TEST_TEMPLATE = '''"""The judge for drill {key}. Each test receives `m`, the module under test."""


def test_solves_the_basic_case(m):
    assert m.solve(2) == 2, "solve(2) should be 2"
'''


def cmd_new(args) -> int:
    levels = discover()
    level = next((lv for lv in levels if lv.number == args.level), None)
    if level is None:
        raise SystemExit(f"no level {args.level}. Levels: {', '.join(str(lv.number) for lv in levels)}")
    if not re.match(r"^[a-z0-9]+(-[a-z0-9]+)*$", args.slug):
        raise SystemExit("slug must be lowercase words joined by hyphens, like 'beam-search'")
    index = max((d.index for d in level.drills), default=0) + 1
    path = level.path / f"{index:02d}-{args.slug}"
    path.mkdir()
    title = args.slug.replace("-", " ").title()
    (path / "reference.py").write_text(REFERENCE_TEMPLATE.format(title=title), encoding="utf-8")
    (path / "test_drill.py").write_text(TEST_TEMPLATE.format(key=f"{level.number}.{index}"), encoding="utf-8")
    print(f"{green(SYM_PASS)} scaffolded {rel(path)}")
    print("  1. write reference.py (wrap learner work in '# >>> fullsend: label' ... '# <<< fullsend')")
    print("  2. write test_drill.py (each test_ function receives the module as m)")
    print("  3. python fullsend.py build && python fullsend.py verify")
    return 0


# --------------------------------------------------------------------------- entry


def main(argv=None) -> int:
    if sys.version_info < MIN_PY:
        print(f"fullsend needs Python {MIN_PY[0]}.{MIN_PY[1]}+; you have {sys.version.split()[0]}")
        return 2
    parser = argparse.ArgumentParser(
        prog="fullsend.py", description="Runner for Forkable AI Engineering, Starter Pack to Full Send."
    )
    parser.add_argument("--version", action="version", version=f"fullsend {__version__}")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("init", help="create your drill.py files")
    sub.add_parser("next", help="show the next drill to clear")
    p = sub.add_parser("check", help="judge one drill")
    p.add_argument("drill")
    p.add_argument("--reference", action="store_true", help="judge reference.py instead of drill.py")
    p = sub.add_parser("hint", help="reveal a hint")
    p.add_argument("drill")
    p.add_argument("n", nargs="?", help="hint number, or 'all'")
    p = sub.add_parser("status", help="the board")
    p.add_argument("--reference", action="store_true", help="judge every reference.py")
    p.add_argument("--summary", action="store_true", help="print a markdown board (for CI summaries)")
    sub.add_parser("card", help="write PROGRESS.md")
    p = sub.add_parser("reset", help="restart a drill from a fresh stub")
    p.add_argument("drill")
    sub.add_parser("doctor", help="check your setup")
    p = sub.add_parser("build", help="regenerate READMEs")
    p.add_argument("--check", action="store_true", help="only report stale files")
    sub.add_parser("verify", help="maintainer health check")
    p = sub.add_parser("new", help="scaffold a drill")
    p.add_argument("level", type=int)
    p.add_argument("slug")

    args = parser.parse_args(argv)
    handlers = {
        "init": cmd_init, "next": cmd_next, "check": cmd_check, "hint": cmd_hint,
        "status": cmd_status, "card": cmd_card, "reset": cmd_reset, "doctor": cmd_doctor,
        "build": cmd_build, "verify": cmd_verify, "new": cmd_new,
    }
    if not args.command:
        parser.print_help()
        return 0
    return handlers[args.command](args)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\ninterrupted")
        sys.exit(130)
    except BrokenPipeError:  # e.g. `python fullsend.py status | head`
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        sys.exit(0)
