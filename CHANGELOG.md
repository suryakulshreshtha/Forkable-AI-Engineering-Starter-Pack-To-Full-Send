# Changelog

Notable changes to Full Send. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and versions are tagged in git.

## [Unreleased]

### Added

- A flowchart of the nine levels in the README, with drill counts and XP per level. `python fullsend.py build` regenerates it from the drills between the `shape` markers, and `verify` and CI fail if it is out of date.

## [1.1.0] - 2026-10-12

### Added

- `python fullsend.py sync` pulls new drills and fixes from upstream into your fork in one step: it adds the `upstream` remote if needed, fetches, merges, then runs `init`. `--dry-run` shows what would be merged and changes nothing.
- `python fullsend.py watch <drill>` re-runs the judge every time you save `drill.py`. `--exit-on-clear` stops once the drill is cleared.
- `python fullsend.py status --json` prints the board as JSON for scripts.
- `python fullsend.py card --badge` also writes `progress-badge.json`, a shields.io endpoint file for a live progress badge in your fork's README.
- Tests for the runner itself in `tests/`, standard library only: `python -m unittest discover -s tests -v`.
- `levels/L9-full-send/app.py --eval --min-score 0.9` exits with an error when the pass rate is too low, so CI can gate on it. `--max-words`, `--overlap` and `-k` expose the chunking and retrieval settings.
- `AGENTS.md` so AI coding assistants pick up the coaching rules without setup.
- A dev container and an "Open in GitHub Codespaces" badge.
- Dependabot for the GitHub Actions the workflow uses.

### Changed

- The judge runs each drill in a child process. A drill that never returns is killed after the per-test limit plus two seconds, only the test it hung in is marked as an error, and the remaining tests still run. `FULLSEND_INPROCESS=1` restores the old in-process judge, which is useful with a debugger. `FULLSEND_TIMEOUT` sets the per-test limit in seconds (default 5).
- `status` and `card` judge drills side by side.
- The commit hint printed by `check` and `card` is two commands on separate lines, so it also works in Windows PowerShell 5.1.
- CI runs on Ubuntu, macOS and Windows, adds the runner tests and the capstone eval gate, and cancels superseded runs.

### Fixed

- A drill making one long call that Python cannot interrupt, such as `sum(range(10**13))`, froze `check`, `status` and `verify` indefinitely.
- `exit()` inside a drill ended the whole runner. It is now reported as an error on that test.
- `app.py --eval` crashed with a `UnicodeEncodeError` when the terminal used a legacy encoding such as cp1252, which is common on Windows.
- `watch` and `check` output could appear late when piped, for example into `tee`.

## [1.0.0] - 2026-10-11

- Nine levels, twenty-eight drills, from linear algebra to a cited-answer capstone.
- `fullsend.py`: init, next, check, hint, status, card, reset, doctor, build, verify, new.
- Fork-safe design: learner files are generated locally and never shipped upstream.
- PocketRAG capstone with a golden evaluation set.
