"""Tests for the runner itself (fullsend.py). Standard library only.

    python -m unittest discover -s tests -v

The drills have their own judges (test_drill.py). These tests cover the machinery around
them: parsing a reference.py, making stubs, finding drills, judging safely, and syncing.
Each CLI test builds a tiny throwaway repository, so nothing here touches your drill.py files.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import fullsend  # noqa: E402

REFERENCE_DOUBLE = '''"""
Title: Double It
XP: 10

## The job
Write `double(x)` that returns twice `x`.

## Why it matters
A first drill for the tests.

## Hints
- Multiply.
- Use the star operator,
  which can wrap onto a second line.
"""
from __future__ import annotations


def double(x):
    """Twice x."""
    # >>> fullsend: multiply by two
    return x * 2
    # <<< fullsend
'''

TESTS_DOUBLE = '''def test_doubles_two(m):
    assert m.double(2) == 4, "double(2) should be 4"


def test_doubles_zero(m):
    assert m.double(0) == 0, "double(0) should be 0"
'''

REFERENCE_SPIN = '''"""
Title: Spin
XP: 5

## The job
Write `spin()` that returns 1.

## Why it matters
Gives the tests something to hang on.

## Hints
- Return one.
"""
from __future__ import annotations


def spin():
    # >>> fullsend: return one
    return 1
    # <<< fullsend
'''

TESTS_SPIN = '''def test_spin_returns_one(m):
    assert m.spin() == 1, "spin() should be 1"
'''


def write_fake_repo(dest: Path) -> Path:
    """A two-drill repository that carries its own copy of the runner."""
    shutil.copy(ROOT / "fullsend.py", dest / "fullsend.py")
    level = dest / "levels" / "L1-basics"
    for slug, ref, tests in (("01-double", REFERENCE_DOUBLE, TESTS_DOUBLE), ("02-spin", REFERENCE_SPIN, TESTS_SPIN)):
        folder = level / slug
        folder.mkdir(parents=True)
        (folder / "reference.py").write_text(ref, encoding="utf-8")
        (folder / "test_drill.py").write_text(tests, encoding="utf-8")
    (level / "_intro.md").write_text("# Basics\n\nTwo tiny drills.\n", encoding="utf-8")
    return dest


class TempCase(unittest.TestCase):
    def make_tmp(self) -> Path:
        tmp = tempfile.mkdtemp(prefix="fullsend-test-")
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        return Path(tmp)


class ParseDocstringTests(unittest.TestCase):
    def test_splits_meta_body_and_hints(self):
        meta, body, hints = fullsend.parse_docstring(REFERENCE_DOUBLE, Path("reference.py"))
        self.assertEqual(meta, {"Title": "Double It", "XP": "10"})
        self.assertIn("## The job", body)
        self.assertNotIn("## Hints", body)

    def test_hint_continuation_lines_join_the_previous_hint(self):
        _, _, hints = fullsend.parse_docstring(REFERENCE_DOUBLE, Path("reference.py"))
        self.assertEqual(hints, ["Multiply.", "Use the star operator, which can wrap onto a second line."])

    def test_a_missing_docstring_is_an_error(self):
        with self.assertRaises(SystemExit):
            fullsend.parse_docstring("x = 1\n", Path("reference.py"))

    def test_a_syntax_error_is_reported_not_raised(self):
        with self.assertRaises(SystemExit):
            fullsend.parse_docstring("def broken(:\n", Path("reference.py"))


class MakeStubTests(TempCase):
    def stub_for(self, source: str) -> str:
        folder = self.make_tmp()
        (folder / "reference.py").write_text(source, encoding="utf-8")
        meta, body, hints = fullsend.parse_docstring(source, folder / "reference.py")
        drill = fullsend.Drill(1, 1, "double", folder, meta, body, hints, "Basics")
        return fullsend.make_stub(drill)

    def test_marked_blocks_become_todos(self):
        stub = self.stub_for(REFERENCE_DOUBLE)
        self.assertIn("# TODO(1.1): multiply by two", stub)
        self.assertIn("raise NotImplementedError('multiply by two')", stub)
        self.assertNotIn("return x * 2", stub, "the worked answer must not leak into the stub")

    def test_the_stub_carries_the_brief_and_the_commands(self):
        stub = self.stub_for(REFERENCE_DOUBLE)
        self.assertIn("## The job", stub)
        self.assertIn("python fullsend.py check 1.1", stub)
        self.assertNotIn("## Hints", stub, "hints stay behind `fullsend.py hint`")

    def test_the_stub_is_valid_python(self):
        compile(self.stub_for(REFERENCE_DOUBLE), "drill.py", "exec")

    def test_a_reference_without_blocks_is_rejected(self):
        source = REFERENCE_DOUBLE.replace("    # >>> fullsend: multiply by two\n", "").replace("    # <<< fullsend\n", "")
        with self.assertRaises(SystemExit):
            self.stub_for(source)

    def test_an_unclosed_block_is_rejected(self):
        with self.assertRaises(SystemExit):
            self.stub_for(REFERENCE_DOUBLE.replace("    # <<< fullsend\n", ""))

    def test_an_end_marker_without_a_start_is_rejected(self):
        with self.assertRaises(SystemExit):
            self.stub_for(REFERENCE_DOUBLE.replace("    # >>> fullsend: multiply by two\n", ""))

    def test_nested_blocks_are_rejected(self):
        nested = REFERENCE_DOUBLE.replace("    return x * 2\n", "    # >>> fullsend: inner\n    return x * 2\n")
        with self.assertRaises(SystemExit):
            self.stub_for(nested)


class FindDrillTests(TempCase):
    def setUp(self):
        self.repo = write_fake_repo(self.make_tmp())
        patcher = mock.patch.object(fullsend, "LEVELS_DIR", self.repo / "levels")
        patcher.start()
        self.addCleanup(patcher.stop)
        self.levels = fullsend.discover()

    def key(self, spec):
        return fullsend.find_drill(self.levels, spec).key

    def test_every_spelling_finds_the_same_drill(self):
        folder = str(self.repo / "levels" / "L1-basics" / "01-double")
        for spec in ("1.1", "L1/01", "l1-1", "double", "01-double", folder):
            with self.subTest(spec=spec):
                self.assertEqual(self.key(spec), "1.1")

    def test_a_typo_suggests_a_neighbour(self):
        with self.assertRaises(SystemExit) as caught:
            fullsend.find_drill(self.levels, "dobule")
        self.assertIn("double", str(caught.exception))

    def test_discovery_reads_titles_and_xp(self):
        drills = fullsend.all_drills(self.levels)
        self.assertEqual([(d.key, d.title, d.xp) for d in drills], [("1.1", "Double It", 10), ("1.2", "Spin", 5)])
        self.assertEqual(self.levels[0].title, "Basics")


class CliCase(TempCase):
    """Runs the real command line against a throwaway repository."""

    def setUp(self):
        self.repo = write_fake_repo(self.make_tmp())
        self.gitconfig = self.repo.parent / (self.repo.name + ".gitconfig")
        self.gitconfig.write_text(
            "[user]\n\tname = Test\n\temail = test@example.invalid\n"
            "[commit]\n\tgpgsign = false\n[init]\n\tdefaultBranch = main\n",
            encoding="utf-8",
        )
        self.env = dict(os.environ, FULLSEND_TIMEOUT="1", GIT_CONFIG_GLOBAL=str(self.gitconfig),
                        GIT_CONFIG_NOSYSTEM="1", NO_COLOR="1", PYTHONIOENCODING="utf-8")
        self.env.pop("FULLSEND_INPROCESS", None)

    def run_cli(self, *args, cwd=None, extra_env=None, timeout=60):
        env = dict(self.env, **(extra_env or {}))
        return subprocess.run([sys.executable, str((cwd or self.repo) / "fullsend.py"), *args],
                              cwd=cwd or self.repo, env=env, capture_output=True, text=True,
                              encoding="utf-8", timeout=timeout)

    def drill_path(self, slug="01-double", repo=None):
        return (repo or self.repo) / "levels" / "L1-basics" / slug / "drill.py"

    def solve(self, slug="01-double", repo=None):
        ref = self.drill_path(slug, repo).with_name("reference.py")
        self.drill_path(slug, repo).write_text(ref.read_text(encoding="utf-8"), encoding="utf-8")

    def write_drill(self, body, slug="01-double"):
        self.drill_path(slug).write_text(body, encoding="utf-8")


class CommandTests(CliCase):
    def test_init_creates_drills_and_never_overwrites(self):
        first = self.run_cli("init")
        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertTrue(self.drill_path().exists() and self.drill_path("02-spin").exists())
        self.drill_path().write_text("# mine\n", encoding="utf-8")
        again = self.run_cli("init")
        self.assertIn("left 2 existing", again.stdout)
        self.assertEqual(self.drill_path().read_text(encoding="utf-8"), "# mine\n")

    def test_check_fails_on_a_fresh_stub_and_clears_with_the_answer(self):
        self.run_cli("init")
        fresh = self.run_cli("check", "1.1")
        self.assertEqual(fresh.returncode, 1)
        self.assertIn("TODO", fresh.stdout)
        self.solve()
        done = self.run_cli("check", "1.1")
        self.assertEqual(done.returncode, 0, done.stdout)
        self.assertIn("CLEARED", done.stdout)
        self.assertIn("git add levels/L1-basics/01-double/drill.py", done.stdout)
        self.assertNotIn("&&", done.stdout, "the commit hint must work in PowerShell 5.1 too")

    def test_check_before_init_says_what_to_run(self):
        result = self.run_cli("check", "1.1")
        self.assertEqual(result.returncode, 1)
        self.assertIn("python fullsend.py init", result.stdout)

    def test_a_wrong_answer_fails_with_the_tests_own_message(self):
        self.run_cli("init")
        self.write_drill("def double(x):\n    return x + 2\n")
        result = self.run_cli("check", "1.1")
        self.assertEqual(result.returncode, 1)
        self.assertIn("double(0) should be 0", result.stdout)

    def test_status_json_reports_each_drill(self):
        self.run_cli("init")
        self.solve()
        board = json.loads(self.run_cli("status", "--json").stdout)
        self.assertEqual(board["drills"], {"cleared": 1, "total": 2})
        self.assertEqual(board["xp"], {"earned": 10, "total": 15})
        states = {d["key"]: d["state"] for d in board["levels"][0]["drills"]}
        self.assertEqual(states, {"1.1": "cleared", "1.2": "untouched"})

    def test_status_rejects_json_with_summary(self):
        self.run_cli("init")
        self.assertEqual(self.run_cli("status", "--json", "--summary").returncode, 1)

    def test_card_writes_progress_and_an_optional_badge(self):
        self.run_cli("init")
        self.solve()
        result = self.run_cli("card", "--badge")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("1 / 2 drills cleared", (self.repo / "PROGRESS.md").read_text(encoding="utf-8"))
        badge = json.loads((self.repo / "progress-badge.json").read_text(encoding="utf-8"))
        self.assertEqual(badge["schemaVersion"], 1)
        self.assertEqual(badge["message"], "1/2 drills · 10 XP")

    def test_card_without_badge_writes_no_badge_file(self):
        self.run_cli("init")
        self.run_cli("card")
        self.assertFalse((self.repo / "progress-badge.json").exists())

    def test_reset_backs_up_then_restarts(self):
        self.run_cli("init")
        self.write_drill("# my attempt\n")
        self.run_cli("reset", "1.1")
        self.assertEqual((self.drill_path().with_name("drill.py.bak")).read_text(encoding="utf-8"), "# my attempt\n")
        self.assertIn("NotImplementedError", self.drill_path().read_text(encoding="utf-8"))

    def test_hints_come_one_at_a_time(self):
        one = self.run_cli("hint", "1.1").stdout
        self.assertIn("hint 1/2", one)
        self.assertNotIn("star operator", one)
        self.assertIn("star operator", self.run_cli("hint", "1.1", "2").stdout)

    def test_build_check_notices_stale_docs_and_build_fixes_them(self):
        self.assertEqual(self.run_cli("build", "--check").returncode, 1)
        self.assertEqual(self.run_cli("build").returncode, 0)
        self.assertEqual(self.run_cli("build", "--check").returncode, 0)

    def test_verify_passes_a_healthy_repo_and_catches_a_broken_reference(self):
        self.run_cli("build")
        self.assertEqual(self.run_cli("verify").returncode, 0)
        ref = self.repo / "levels" / "L1-basics" / "01-double" / "reference.py"
        ref.write_text(ref.read_text(encoding="utf-8").replace("x * 2", "x * 3"), encoding="utf-8")
        broken = self.run_cli("verify")
        self.assertEqual(broken.returncode, 1)
        self.assertIn("reference fails", broken.stdout)

    def test_new_scaffolds_a_drill_that_verify_accepts_once_built(self):
        self.assertEqual(self.run_cli("new", "1", "triple").returncode, 0)
        self.run_cli("build")
        self.assertEqual(self.run_cli("verify").returncode, 0, self.run_cli("verify").stdout)


class JudgeSafetyTests(CliCase):
    """A drill that misbehaves must cost the learner seconds, never the whole run."""

    def setUp(self):
        super().setUp()
        self.run_cli("init")

    def test_a_python_loop_is_interrupted_with_a_clear_message(self):
        self.write_drill("def spin():\n    while True:\n        pass\n", "02-spin")
        result = self.run_cli("check", "1.2")
        self.assertEqual(result.returncode, 1)
        self.assertIn("infinite loop", result.stdout)

    def test_one_long_c_call_is_killed_not_waited_for(self):
        self.write_drill("def spin():\n    return sum(range(10**13))\n", "02-spin")
        started = time.monotonic()
        result = self.run_cli("check", "1.2", timeout=40)
        self.assertLess(time.monotonic() - started, 25, "the judge should have killed the child by now")
        self.assertEqual(result.returncode, 1)
        self.assertIn("the judge stopped it", result.stdout)

    def test_a_hang_while_importing_is_killed_too(self):
        self.write_drill("total = sum(range(10**13))\ndef spin():\n    return 1\n", "02-spin")
        result = self.run_cli("check", "1.2", timeout=40)
        self.assertEqual(result.returncode, 1)
        self.assertIn("importing your file took too long", result.stdout)

    def test_a_crash_only_fails_the_test_it_happened_in(self):
        self.write_drill("import os\n\ndef double(x):\n    if x == 2:\n        os._exit(3)\n    return x * 2\n")
        out = self.run_cli("check", "1.1").stdout
        self.assertIn("crashed the judge process (exit code 3)", out)
        self.assertIn("doubles zero", out.split("crashed")[1], "the next test must still run")
        self.assertIn("1/2 passing", out)

    def test_exit_inside_a_drill_is_an_error_not_a_dead_runner(self):
        self.write_drill("def double(x):\n    raise SystemExit(2)\n")
        result = self.run_cli("check", "1.1")
        self.assertEqual(result.returncode, 1)
        self.assertIn("called exit()", result.stdout)

    def test_print_output_from_a_drill_still_reaches_the_terminal(self):
        self.write_drill("def double(x):\n    print('hello-from-the-learner')\n    return x * 2\n")
        self.assertIn("hello-from-the-learner", self.run_cli("check", "1.1").stdout)

    def test_the_in_process_fallback_gives_the_same_verdicts(self):
        self.solve()
        result = self.run_cli("check", "1.1", extra_env={"FULLSEND_INPROCESS": "1"})
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("CLEARED", result.stdout)

    def test_a_hung_drill_does_not_block_the_board(self):
        self.write_drill("def spin():\n    return sum(range(10**13))\n", "02-spin")
        self.solve("01-double")
        board = json.loads(self.run_cli("status", "--json", timeout=40).stdout)
        states = {d["key"]: d["state"] for d in board["levels"][0]["drills"]}
        self.assertEqual(states, {"1.1": "cleared", "1.2": "started"})


class WatchTests(CliCase):
    def test_watch_reruns_on_save_and_stops_when_cleared(self):
        self.run_cli("init")
        log = self.repo / "watch.log"
        with open(log, "w", encoding="utf-8") as fh:
            proc = subprocess.Popen([sys.executable, str(self.repo / "fullsend.py"), "watch", "1.1", "--exit-on-clear"],
                                    cwd=self.repo, env=self.env, stdout=fh, stderr=subprocess.STDOUT)
        self.addCleanup(lambda: proc.poll() is None and proc.kill())
        deadline = time.monotonic() + 20
        while "0/2 passing" not in log.read_text(encoding="utf-8") and time.monotonic() < deadline:
            time.sleep(0.1)
        self.assertIn("0/2 passing", log.read_text(encoding="utf-8"), "the first run should happen on its own")
        time.sleep(0.05)
        self.solve()
        proc.wait(timeout=20)
        self.assertEqual(proc.returncode, 0)
        self.assertIn("CLEARED", log.read_text(encoding="utf-8"))


@unittest.skipUnless(shutil.which("git"), "git is not installed")
class SyncTests(CliCase):
    def git(self, *args, cwd):
        done = subprocess.run(["git", *args], cwd=cwd, env=self.env, capture_output=True, text=True, encoding="utf-8")
        self.assertEqual(done.returncode, 0, f"git {' '.join(args)}: {done.stderr}")
        return done.stdout.strip()

    def setUp(self):
        super().setUp()
        base = self.repo.parent
        self.upstream_wc = self.repo  # the fake repo plays the original project
        self.git("init", "-q", cwd=self.upstream_wc)
        self.git("checkout", "-q", "-b", "main", cwd=self.upstream_wc)
        self.git("add", "-A", cwd=self.upstream_wc)
        self.git("commit", "-q", "-m", "v1", cwd=self.upstream_wc)
        self.upstream_bare = base / (self.repo.name + "-upstream.git")
        self.git("clone", "-q", "--bare", str(self.upstream_wc), str(self.upstream_bare), cwd=base)
        self.fork = base / (self.repo.name + "-fork")
        self.git("clone", "-q", str(self.upstream_bare), str(self.fork), cwd=base)
        self.addCleanup(shutil.rmtree, self.fork, ignore_errors=True)
        self.addCleanup(shutil.rmtree, self.upstream_bare, ignore_errors=True)
        self.env["FULLSEND_UPSTREAM"] = str(self.upstream_bare)

    def upstream_adds_a_drill(self):
        self.run_cli("new", "1", "triple", cwd=self.upstream_wc)
        self.run_cli("build", cwd=self.upstream_wc)
        self.git("add", "-A", cwd=self.upstream_wc)
        self.git("commit", "-q", "-m", "add triple", cwd=self.upstream_wc)
        self.git("push", "-q", str(self.upstream_bare), "main", cwd=self.upstream_wc)

    def sync(self, *args):
        return self.run_cli("sync", *args, cwd=self.fork)

    def test_a_fork_pulls_new_drills_and_creates_only_the_new_drill_py(self):
        self.run_cli("init", cwd=self.fork)
        mine = self.drill_path(repo=self.fork)
        mine.write_text("# my work in progress\n", encoding="utf-8")
        self.git("add", "-A", cwd=self.fork)
        self.git("commit", "-q", "-m", "my work", cwd=self.fork)
        self.upstream_adds_a_drill()
        result = self.sync()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("1 new drill(s): 1.3 Triple", result.stdout)
        self.assertTrue(self.drill_path("03-triple", self.fork).exists())
        self.assertEqual(mine.read_text(encoding="utf-8"), "# my work in progress\n")
        self.assertIn("upstream", self.git("remote", cwd=self.fork).split())

    def test_dry_run_changes_nothing(self):
        self.upstream_adds_a_drill()
        before = self.git("rev-parse", "HEAD", cwd=self.fork)
        result = self.sync("--dry-run")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.git("rev-parse", "HEAD", cwd=self.fork), before)
        self.assertNotIn("upstream", self.git("remote", cwd=self.fork).split())

    def test_second_sync_says_up_to_date(self):
        self.upstream_adds_a_drill()
        self.sync()
        self.assertIn("already up to date", self.sync().stdout)

    def test_uncommitted_changes_stop_the_sync_before_anything_happens(self):
        (self.fork / "levels" / "L1-basics" / "_intro.md").write_text("# Basics\n\nedited\n", encoding="utf-8")
        result = self.sync()
        self.assertEqual(result.returncode, 1)
        self.assertIn("uncommitted changes", result.stdout)
        self.assertNotIn("upstream", self.git("remote", cwd=self.fork).split())

    def test_a_conflict_is_explained_and_can_be_aborted(self):
        intro = Path("levels") / "L1-basics" / "_intro.md"
        (self.fork / intro).write_text("# Basics\n\nfork wording\n", encoding="utf-8")
        self.git("commit", "-q", "-am", "fork edit", cwd=self.fork)
        (self.upstream_wc / intro).write_text("# Basics\n\nupstream wording\n", encoding="utf-8")
        self.git("commit", "-q", "-am", "upstream edit", cwd=self.upstream_wc)
        self.git("push", "-q", str(self.upstream_bare), "main", cwd=self.upstream_wc)
        result = self.sync()
        self.assertEqual(result.returncode, 1)
        self.assertIn("conflicts", result.stdout)
        self.assertIn("_intro.md", result.stdout)
        self.assertIn("git merge --abort", result.stdout)
        self.git("merge", "--abort", cwd=self.fork)

    def test_the_upstream_clone_itself_has_nothing_to_sync(self):
        self.git("remote", "set-url", "origin", f"https://example.invalid/{fullsend.UPSTREAM_SLUG}.git", cwd=self.fork)
        result = self.sync()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("nothing to sync", result.stdout)
        self.assertNotIn("upstream", self.git("remote", cwd=self.fork).split())


class CapstoneAppTests(unittest.TestCase):
    """The PocketRAG command line, run for real against this repository's own docs."""

    def run_app(self, *args, encoding=None):
        env = dict(os.environ, NO_COLOR="1")
        if encoding:
            env["PYTHONIOENCODING"] = encoding
        return subprocess.run([sys.executable, str(ROOT / "levels" / "L9-full-send" / "app.py"), "--reference", *args],
                              cwd=ROOT, env=env, capture_output=True, timeout=120)

    def test_the_golden_set_passes_with_the_reference_implementation(self):
        done = self.run_app("--eval", "--min-score", "1.0")
        self.assertEqual(done.returncode, 0, done.stdout.decode("utf-8", "replace"))

    def test_the_gate_fails_when_retrieval_gets_worse(self):
        done = self.run_app("--eval", "--max-words", "6", "--overlap", "1", "-k", "1", "--min-score", "0.9")
        self.assertEqual(done.returncode, 1)
        self.assertIn("eval gate failed", done.stdout.decode("utf-8", "replace"))

    def test_a_bad_threshold_is_a_usage_error(self):
        self.assertEqual(self.run_app("--eval", "--min-score", "2").returncode, 2)
        self.assertEqual(self.run_app("--min-score", "0.5").returncode, 2)

    def test_a_legacy_windows_console_encoding_does_not_crash_the_report(self):
        done = self.run_app("--eval", encoding="cp1252")
        self.assertEqual(done.returncode, 0, done.stderr.decode("utf-8", "replace"))

    def test_it_answers_with_a_citation_and_refuses_off_topic_questions(self):
        answered = self.run_app("what does verify check?").stdout.decode("utf-8", "replace")
        self.assertIn("Sources:", answered)
        refused = self.run_app("how do I bake sourdough bread?").stdout.decode("utf-8", "replace")
        self.assertNotIn("Sources:", refused)


if __name__ == "__main__":
    unittest.main()
