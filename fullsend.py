#!/usr/bin/env python3
"""fullsend: the one-file runner for Forkable AI Engineering, Starter Pack to Full Send.

Learner commands
  python fullsend.py init                 create your drill.py files (never overwrites)
  python fullsend.py next                 show the next drill you have not cleared
  python fullsend.py check <drill>        run the judge against your drill.py
  python fullsend.py watch <drill>        re-run the judge every time you save drill.py
  python fullsend.py hint <drill> [n|all] reveal hints one at a time
  python fullsend.py status [--json]      the board: every drill, cleared or not
  python fullsend.py card [--badge]       write PROGRESS.md (and a badge file) for your fork
  python fullsend.py sync [--dry-run]     pull new drills from upstream into your fork
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
import itertools
import json
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
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

__version__ = "1.1.0"

ROOT = Path(__file__).resolve().parent
LEVELS_DIR = ROOT / "levels"
MIN_PY = (3, 9)
UPSTREAM_SLUG = "suryakulshreshtha/Forkable-AI-Engineering-Starter-Pack-To-Full-Send"
UPSTREAM_URL = f"https://github.com/{UPSTREAM_SLUG}.git"
SOFT_TIMEOUT = 5.0  # seconds one test may run before the judge interrupts it
HARD_GRACE = 2.0  # extra seconds before a judge process that went silent is killed
WORKER = "_judge-worker"  # hidden subcommand: the child process that runs one judge

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True  # keep drill folders free of __pycache__

LEVEL_DIR = re.compile(r"^L(\d+)-([a-z0-9-]+)$")
DRILL_DIR = re.compile(r"^(\d+)-([a-z0-9-]+)$")
START = re.compile(r"^(?P<indent>[ \t]*)# >>> fullsend(?::\s*(?P<label>.*?))?\s*$")
END = re.compile(r"^[ \t]*# <<< fullsend\s*$")
BOARD_START = "<!-- board:start -->"
BOARD_END = "<!-- board:end -->"
SHAPE_START = "<!-- shape:start -->"
SHAPE_END = "<!-- shape:end -->"
MERMAID_INIT = (
    "%%{init: {'theme':'base','themeVariables':{'primaryColor':'#f6f8fa',"
    "'primaryTextColor':'#1f2328','primaryBorderColor':'#1f883d','lineColor':'#6e7781',"
    "'fontSize':'14px'}}}%%"
)

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


_load_counter = itertools.count(1)


def _load(path: Path, prefix: str):
    name = f"_fullsend_{prefix}_{next(_load_counter)}"
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


def _soft_timeout() -> float:
    """Seconds one test may run before the judge interrupts it (FULLSEND_TIMEOUT overrides)."""
    try:
        return float(os.environ.get("FULLSEND_TIMEOUT", SOFT_TIMEOUT))
    except ValueError:
        return SOFT_TIMEOUT


def _inprocess() -> bool:
    """FULLSEND_INPROCESS=1 judges inside this process, which is handy with a debugger."""
    return bool(os.environ.get("FULLSEND_INPROCESS"))


def _run_tests(tests_path: Path, target: Path, timeout: float, start: int, emit) -> None:
    """Run the judge's tests in this process and report each step through emit(e=..., ...).

    Events: tests (every test name), start (index), done (index, name, status, detail).
    Python-level loops are interrupted after `timeout`; one long C call (sum(range(10**13)))
    cannot be, which is why judge() normally runs this in a child process it can kill.
    """
    tests_module = _load(tests_path, "tests")
    tests = [(n, f) for n, f in vars(tests_module).items() if n.startswith("test_") and callable(f)]
    emit(e="tests", names=[n for n, _ in tests])

    def finish_rest(status: str, detail: str) -> None:
        for i in range(start, len(tests)):
            emit(e="done", i=i, name=tests[i][0], status=status, detail=detail)

    try:
        with _time_limit(timeout):
            module = _load(target, "target")
    except NotImplementedError as exc:
        return finish_rest("todo", str(exc))
    except _Timeout:
        return finish_rest("error", "importing your file took too long")
    except SystemExit:
        return finish_rest("error", f"{target.name} called exit() while importing")
    except Exception as exc:  # syntax errors, missing imports, ...
        return finish_rest("error", f"could not import {target.name}: {type(exc).__name__}: {exc}")

    limit = timeout
    for i in range(start, len(tests)):
        name, fn = tests[i]
        emit(e="start", i=i)
        status, detail = "pass", ""
        try:
            with _time_limit(limit):
                fn(module)
        except NotImplementedError as exc:
            status, detail = "todo", str(exc) or "not implemented yet"
        except AssertionError as exc:
            status, detail = "fail", str(exc) or "assertion failed" + _where(exc, tests_path)
        except _Timeout:
            status, detail = "error", f"no answer after {limit:g}s: infinite loop?"
            limit = min(timeout, 1.0)  # after one hang the rest get a much shorter limit
        except SystemExit:
            status, detail = "error", "your code called exit()"
        except Exception as exc:
            status, detail = "error", f"{type(exc).__name__}: {exc}{_where(exc, target)}"
        emit(e="done", i=i, name=name, status=status, detail=detail)


def judge_inprocess(drill: Drill, target: Path, timeout: float | None = None) -> list:
    """Judge inside this process. No hard kill: see judge_isolated."""
    timeout = _soft_timeout() if timeout is None else timeout
    names, results = [], {}

    def emit(**event):
        if event["e"] == "tests":
            names.extend(event["names"])
        elif event["e"] == "done":
            results[event["i"]] = Outcome(event["name"], event["status"], event["detail"])

    _run_tests(drill.tests, target, timeout, 0, emit)
    return [results[i] for i in range(len(names)) if i in results]


def _worker_main(argv) -> int:
    """The child process: run the tests from --start and log events to --out as JSON lines."""
    parser = argparse.ArgumentParser(prog=f"fullsend.py {WORKER}")
    parser.add_argument("--tests", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--timeout", type=float, required=True)
    parser.add_argument("--start", type=int, default=0)
    a = parser.parse_args(argv)
    with open(a.out, "a", encoding="utf-8") as fh:
        def emit(**event):
            fh.write(json.dumps(event) + "\n")
            fh.flush()

        try:
            _run_tests(Path(a.tests), Path(a.target), a.timeout, a.start, emit)
        except BaseException as exc:  # the judge itself is broken, not the learner's code
            emit(e="crash", detail=f"{type(exc).__name__}: {exc}")
            return 3
    return 0


def _stop(proc) -> None:
    """Kill the judge process, and on POSIX anything it started."""
    try:
        if os.name == "posix":
            os.killpg(proc.pid, signal.SIGKILL)
        else:
            proc.kill()
    except OSError:
        pass
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass


def _read_events(path: Path, offset: int):
    """Return (events, new_offset) for the complete JSON lines written after `offset`."""
    try:
        with open(path, "rb") as fh:
            fh.seek(offset)
            data = fh.read()
    except OSError:
        return [], offset
    end = data.rfind(b"\n")
    if end < 0:
        return [], offset
    events = []
    for line in data[: end + 1].splitlines():
        try:
            events.append(json.loads(line.decode("utf-8")))
        except ValueError:
            pass
    return events, offset + end + 1


def judge_isolated(drill: Drill, target: Path, timeout: float | None = None, show_output: bool = False) -> list:
    """Run the judge in a child process, so one runaway test can be killed outright.

    The child logs progress to a file. If it goes silent for longer than the per-test
    limit plus a grace period, it is killed, only that test is marked as an error, and
    a fresh child carries on with the tests after it.
    """
    timeout = _soft_timeout() if timeout is None else timeout
    names, results = None, {}
    start = 0
    sys.stdout.flush()
    sys.stderr.flush()
    with tempfile.TemporaryDirectory() as tmp:
        attempt = 0
        while True:
            attempt += 1
            hard = timeout + HARD_GRACE
            progress = Path(tmp) / f"progress{attempt}.jsonl"
            progress.write_text("", encoding="utf-8")
            errlog = Path(tmp) / f"stderr{attempt}.txt"
            cmd = [sys.executable, str(Path(__file__).resolve()), WORKER,
                   "--tests", str(drill.tests), "--target", str(target), "--out", str(progress),
                   "--timeout", repr(timeout), "--start", str(start)]
            popen_kwargs = {"stdin": subprocess.DEVNULL}
            if os.name == "posix":
                popen_kwargs["start_new_session"] = True
            err_fh = None
            if not show_output:  # learners' print() calls only show up in `check`
                err_fh = open(errlog, "wb")
                popen_kwargs.update(stdout=subprocess.DEVNULL, stderr=err_fh)
            state = {"current": None, "last": time.monotonic(), "note": ""}

            def absorb(events):
                nonlocal names
                for ev in events:
                    state["last"] = time.monotonic()
                    kind = ev.get("e")
                    if kind == "tests" and names is None:
                        names = list(ev["names"])
                    elif kind == "start":
                        state["current"] = ev["i"]
                    elif kind == "done":
                        results[ev["i"]] = Outcome(ev["name"], ev["status"], ev.get("detail", ""))
                        state["current"] = None
                    elif kind == "crash":
                        state["note"] = ev.get("detail", "")

            killed = False
            proc = subprocess.Popen(cmd, **popen_kwargs)
            try:
                offset = 0
                while True:
                    finished = proc.poll() is not None
                    events, offset = _read_events(progress, offset)
                    absorb(events)
                    if finished:
                        break
                    if time.monotonic() - state["last"] > hard:
                        _stop(proc)
                        killed = True
                        events, offset = _read_events(progress, offset)
                        absorb(events)
                        break
                    time.sleep(0.02)
            finally:
                if proc.poll() is None:
                    _stop(proc)
                if err_fh is not None:
                    err_fh.close()

            tail = ""
            if not show_output:
                try:
                    tail = errlog.read_text(encoding="utf-8", errors="replace").strip().splitlines()[-1]
                except (OSError, IndexError):
                    tail = ""
            if names is None:
                why = state["note"] or tail or f"the judge process exited with code {proc.returncode} before listing tests"
                return [Outcome("judge", "error", why)]
            missing = [i for i in range(start, len(names)) if i not in results]
            if not missing:
                break
            crash = f"your code crashed the judge process (exit code {proc.returncode})" + (f": {tail}" if tail else "")
            current = state["current"]
            if current is not None and current in missing:
                why = f"no answer after {hard:g}s: infinite loop? (the judge stopped it)" if killed else crash
                results[current] = Outcome(names[current], "error", why)
                start = current + 1
                timeout = min(timeout, 1.0)  # after one hang the rest get a much shorter limit
                if start >= len(names):
                    break
                continue
            why = "importing your file took too long" if killed else crash
            for i in missing:
                results[i] = Outcome(names[i], "error", why)
            break
    return [results[i] for i in range(len(names)) if i in results]


def judge(drill: Drill, target: Path, timeout: float | None = None, show_output: bool = False) -> list:
    """Run every test_ function in the drill's judge against the module at `target`.

    Normally this happens in a child process, so a drill that never returns is killed
    instead of freezing the runner. FULLSEND_INPROCESS=1 judges in this process instead.
    """
    if _inprocess():
        return judge_inprocess(drill, target, timeout)
    return judge_isolated(drill, target, timeout, show_output)


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


def _check_once(levels, d: Drill, target: Path, reference: bool) -> int:
    print(f"{cyan(SYM_ARROW)} {bold(d.key + ' · ' + d.title)}  {dim(rel(target))}")
    started = time.perf_counter()
    outcomes = judge(d, target, show_output=True)
    print_outcomes(outcomes)
    passed = sum(o.status == "pass" for o in outcomes)
    took = time.perf_counter() - started
    if cleared(outcomes):
        print(green(bold(f"\n  CLEARED  +{d.xp} XP")) + dim(f"  ({passed}/{len(outcomes)} in {took:.2f}s)"))
        if not reference:
            nxt = _after(levels, d)
            print(dim("  commit it:"))
            print(dim(f"    git add {rel(d.mine)}"))
            print(dim(f"    git commit -m \"clear {d.key}\""))
            if nxt:
                print(dim(f"  next up:    python fullsend.py check {nxt.key}  ({nxt.title})"))
        return 0
    print(f"\n  {passed}/{len(outcomes)} passing" + dim(f"  ·  stuck? python fullsend.py hint {d.key}"))
    return 1


def cmd_check(args) -> int:
    levels = discover()
    d = find_drill(levels, args.drill)
    target = d.reference if args.reference else d.mine
    if not target.exists():
        print(yellow(f"{rel(target)} does not exist yet. Run: python fullsend.py init"))
        return 1
    return _check_once(levels, d, target, args.reference)


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
    def work(d: Drill):
        if reference:
            outcomes = judge(d, d.reference)
            ok = cleared(outcomes)
            return ("cleared" if ok else "started", sum(o.status == "pass" for o in outcomes), len(outcomes))
        return drill_state(d)

    drills = all_drills(levels)
    if _inprocess() or len(drills) < 2:  # in-process judging relies on the main thread's timer
        results = [work(d) for d in drills]
    else:  # each judge is its own child process, so they can run side by side
        pool = ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 2))
        try:
            results = list(pool.map(work, drills))
        except BaseException:
            pool.shutdown(wait=False, cancel_futures=True)
            raise
        pool.shutdown()
    by_key = {d.key: r for d, r in zip(drills, results)}
    return [(level, [(d, *by_key[d.key]) for d in level.drills]) for level in levels]


def cmd_status(args) -> int:
    levels = discover()
    rows = collect_board(levels, reference=args.reference)
    total_xp = sum(lv.xp for lv in levels)
    earned = sum(d.xp for _, states in rows for d, s, *_ in states if s == "cleared")
    n_cleared = sum(1 for _, states in rows for _, s, *_ in states if s == "cleared")
    n_all = sum(len(states) for _, states in rows)
    if args.json and args.summary:
        raise SystemExit("choose one of --json and --summary")
    if args.json:
        print_utf8(json.dumps(board_json(rows, earned, total_xp, n_cleared, n_all), indent=2, ensure_ascii=False))
        return 0
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


def board_json(rows, earned, total_xp, n_cleared, n_all) -> dict:
    """The board as plain data, for scripts, dashboards and badges."""
    return {
        "xp": {"earned": earned, "total": total_xp},
        "drills": {"cleared": n_cleared, "total": n_all},
        "levels": [
            {
                "level": level.number,
                "title": level.title,
                "drills": [
                    {"key": d.key, "title": d.title, "xp": d.xp, "state": state, "passed": passed, "total": total}
                    for d, state, passed, total in states
                ],
            }
            for level, states in rows
        ],
    }


def badge_json(earned, total_xp, n_cleared, n_all) -> dict:
    """A shields.io endpoint payload: https://shields.io/badges/endpoint-badge"""
    pct = 0 if n_all == 0 else 100 * n_cleared / n_all
    if n_cleared == 0:
        color = "lightgrey"
    elif pct < 34:
        color = "orange"
    elif pct < 67:
        color = "yellow"
    elif pct < 100:
        color = "yellowgreen"
    else:
        color = "brightgreen"
    return {"schemaVersion": 1, "label": "full send", "message": f"{n_cleared}/{n_all} drills · {earned} XP", "color": color}


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
    files = "PROGRESS.md"
    if args.badge:
        badge = ROOT / "progress-badge.json"
        badge.write_text(json.dumps(badge_json(earned, total_xp, n_cleared, n_all), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"{green(SYM_PASS)} wrote {rel(badge)} (a shields.io endpoint file: see FORK-ME.md)")
        files += " progress-badge.json"
    print(dim("  commit it so your fork shows where you are:"))
    print(dim(f"    git add {files}"))
    print(dim("    git commit -m \"progress card\""))
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


def cmd_watch(args) -> int:
    levels = discover()
    d = find_drill(levels, args.drill)
    target = d.reference if args.reference else d.mine
    if not target.exists():
        print(yellow(f"{rel(target)} does not exist yet. Run: python fullsend.py init"))
        return 1
    print(dim(f"watching {rel(target)}: save the file to re-run the judge, Ctrl+C to stop"))

    def stamp():
        try:
            return target.stat().st_mtime_ns, d.tests.stat().st_mtime_ns
        except OSError:
            return None

    seen = None
    while True:
        now = stamp()
        if now is not None and now != seen:
            time.sleep(0.15)  # let an editor finish writing before the judge reads the file
            seen = stamp()
            print(dim(f"\n--- {time.strftime('%H:%M:%S')} ---"))
            code = _check_once(levels, d, target, args.reference)
            sys.stdout.flush()  # when piped (| tee), results would otherwise wait for the next run
            if code == 0 and args.exit_on_clear:
                return 0
        time.sleep(0.4)


def _git(git: str, *cmd) -> str:
    try:
        done = subprocess.run([git, *cmd], cwd=ROOT, capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return ""
    return done.stdout.strip()


def _git_run(git: str, *cmd, timeout: int = 120):
    """Run git in the repository and return (exit code, stdout, stderr)."""
    try:
        done = subprocess.run([git, *cmd], cwd=ROOT, capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=timeout)
    except (OSError, subprocess.SubprocessError) as exc:
        return 127, "", str(exc)
    return done.returncode, done.stdout.strip(), done.stderr.strip()


def cmd_sync(args) -> int:
    git = shutil.which("git")
    if not git:
        print(red("git is not installed. Install it, then run this again."))
        return 1
    if not (ROOT / ".git").exists():
        print(yellow("This folder is not a git clone, so there is nothing to sync."))
        return 1
    origin = _git(git, "remote", "get-url", "origin")
    if UPSTREAM_SLUG.lower() in origin.lower():
        print("This clone points at the upstream repository itself, so there is nothing to sync. Use: git pull")
        return 0
    if not args.dry_run and _git(git, "status", "--porcelain", "--untracked-files=no"):
        print(yellow("You have uncommitted changes. Commit them first, then sync:"))
        print(dim("  git add -A"))
        print(dim("  git commit -m \"save my work\""))
        return 1
    upstream_url = os.environ.get("FULLSEND_UPSTREAM", UPSTREAM_URL)
    if "upstream" not in _git(git, "remote").split():
        if args.dry_run:
            print(dim(f"would run: git remote add upstream {upstream_url}"))
            print(dim("then fetch upstream and merge any new commits (re-run without --dry-run)"))
            return 0
        code, _, err = _git_run(git, "remote", "add", "upstream", upstream_url)
        if code:
            print(red(f"could not add the upstream remote: {err}"))
            return 1
        print(f"{green(SYM_PASS)} added remote 'upstream' -> {upstream_url}")
    print(dim("fetching upstream ..."))
    code, _, err = _git_run(git, "fetch", "upstream")
    if code:
        print(red(f"git fetch upstream failed: {err}"))
        return 1
    ref = f"upstream/{args.branch}"
    if _git_run(git, "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}")[0]:
        print(red(f"upstream has no branch called '{args.branch}' (try: --branch <name>)"))
        return 1
    code, count, _ = _git_run(git, "rev-list", "--count", f"HEAD..{ref}")
    behind = int(count) if code == 0 and count.isdigit() else 0
    if behind == 0:
        print(f"{green(SYM_PASS)} already up to date with {ref}")
        return 0
    print(f"{behind} new commit(s) on {ref}:")
    for line in _git_run(git, "log", "--oneline", "-n", "15", f"HEAD..{ref}")[1].splitlines():
        print(f"  {line}")
    if args.dry_run:
        print(dim("dry run: nothing was merged. Run: python fullsend.py sync"))
        return 0
    before = {d.key for d in all_drills(discover())}
    code, out, err = _git_run(git, "merge", "--no-edit", ref)
    if code:
        conflicts = _git_run(git, "diff", "--name-only", "--diff-filter=U")[1].splitlines()
        if conflicts:
            print(red("The merge stopped on conflicts in:"))
            for name in conflicts:
                print(f"  {name}")
            print("Fix them, then: git add <files>  and  git commit   (or undo it all with: git merge --abort)")
            print(dim("Conflicts usually mean a file upstream owns (test_drill.py, reference.py, a README) was edited here."))
        else:
            print(red(f"git merge failed: {err or out}"))
        return 1
    print(f"{green(SYM_PASS)} merged {ref}")
    cmd_init(args)
    new = [d for d in all_drills(discover()) if d.key not in before]
    if new:
        print(f"{len(new)} new drill(s): " + ", ".join(f"{d.key} {d.title}" for d in new))
    return 0


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
             f"python fullsend.py sync   (or: git remote add upstream {UPSTREAM_URL})")
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


def _mermaid_text(text: str) -> str:
    """Make a title safe inside a double-quoted Mermaid label."""
    return text.replace("&", "&amp;").replace('"', "#quot;").replace("<", "&lt;").replace(">", "&gt;")


def shape_graph(levels) -> str:
    """A Mermaid flowchart of the levels in order, built from the drills so it cannot drift."""
    out = [SHAPE_START]
    if levels:
        out += ["```mermaid", MERMAID_INIT, "flowchart TB"]
        for lv in levels:
            n = len(lv.drills)
            label = f"L{lv.number} · {_mermaid_text(lv.title)}<br/><sub>{n} drill{'s' if n != 1 else ''} · {lv.xp} XP</sub>"
            out.append(f'  L{lv.number}["{label}"]')
        for a, b in zip(levels, levels[1:]):
            out.append(f"  L{a.number} --> L{b.number}")
        out.append(f"  class L{levels[-1].number} capstone")
        out.append("  classDef capstone fill:#dafbe1,stroke:#1f883d,stroke-width:2px")
        out.append("```")
    out.append(SHAPE_END)
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
        for start, end, block in (
            (BOARD_START, BOARD_END, root_board(levels)),
            (SHAPE_START, SHAPE_END, shape_graph(levels)),
        ):
            if start in text and end in text:
                head, rest = text.split(start, 1)
                _, tail = rest.split(end, 1)
                text = head + block + tail
        files[readme] = text
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
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == WORKER:  # internal: the judge's child process
        return _worker_main(argv[1:])
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
    p = sub.add_parser("watch", help="re-run the judge whenever you save")
    p.add_argument("drill")
    p.add_argument("--reference", action="store_true", help="watch reference.py instead of drill.py")
    p.add_argument("--exit-on-clear", action="store_true", help="stop watching once the drill is cleared")
    p = sub.add_parser("hint", help="reveal a hint")
    p.add_argument("drill")
    p.add_argument("n", nargs="?", help="hint number, or 'all'")
    p = sub.add_parser("status", help="the board")
    p.add_argument("--reference", action="store_true", help="judge every reference.py")
    p.add_argument("--summary", action="store_true", help="print a markdown board (for CI summaries)")
    p.add_argument("--json", action="store_true", help="print the board as JSON")
    p = sub.add_parser("card", help="write PROGRESS.md")
    p.add_argument("--badge", action="store_true", help="also write progress-badge.json (a shields.io endpoint file)")
    p = sub.add_parser("sync", help="pull new drills from upstream")
    p.add_argument("--dry-run", action="store_true", help="show what would be merged, change nothing")
    p.add_argument("--branch", default="main", help="upstream branch to merge (default: main)")
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
        "init": cmd_init, "next": cmd_next, "check": cmd_check, "watch": cmd_watch, "hint": cmd_hint,
        "status": cmd_status, "card": cmd_card, "sync": cmd_sync, "reset": cmd_reset, "doctor": cmd_doctor,
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
