# Fork Me

This repository is designed to be forked and worked in. Here is how to do that well.

## Why forking works here

Upstream ships blueprints: briefs, judges and reference answers. It never ships `drill.py`. Those files are created on your machine by `python fullsend.py init`, and from then on they belong to your fork. Because upstream never touches them, pulling new drills or fixes from upstream does not collide with your work.

## Fork and start

1. On GitHub, press **Fork**. Keep the name or choose your own.
2. Clone your fork and set up the drills:

```bash
git clone https://github.com/<you>/Forkable-AI-Engineering-Starter-Pack-To-Full-Send.git
cd Forkable-AI-Engineering-Starter-Pack-To-Full-Send
python fullsend.py init
python fullsend.py doctor
```

Prefer a browser? Open your fork (not the original) in a GitHub Codespace. The included dev container runs `init` and `doctor` for you.

## Save your progress

Commit each drill when the judge clears it. The check command prints the exact line to run.

```bash
git add levels/L1-numbers-that-learn/01-dot-and-matmul/drill.py
git commit -m "clear 1.1"
git push
```

Your commit history becomes a timeline of what you built and when.

## Stay in sync

To pull new drills and fixes into your fork, commit your work and run one command:

```bash
python fullsend.py sync --dry-run   # optional: list what is new, change nothing
python fullsend.py sync
```

`sync` adds the original repository as a remote called `upstream` the first time, fetches it, merges `upstream/main`, then runs `init` so any new drills get their `drill.py`. It refuses to start while you have uncommitted changes, and it never overwrites your `drill.py` files.

If you would rather do it by hand, the same steps are:

```bash
git remote add upstream https://github.com/suryakulshreshtha/Forkable-AI-Engineering-Starter-Pack-To-Full-Send.git   # once
git fetch upstream
git merge upstream/main
python fullsend.py init     # creates drill.py files for any new drills; never overwrites yours
```

You will only see conflicts if you edited a file upstream also owns, such as a `test_drill.py` or a `reference.py`. Keep your changes inside `drill.py` files and new files of your own, and merges stay clean.

## Show it off

```bash
python fullsend.py card     # writes PROGRESS.md from the judge's verdicts
git add PROGRESS.md
git commit -m "progress card"
git push
```

Link `PROGRESS.md` from your profile or résumé. Every check mark in it was earned by running the judge against your own code.

Want a live badge in your fork's README? Add `--badge` and commit the extra file:

```bash
python fullsend.py card --badge   # also writes progress-badge.json
git add PROGRESS.md progress-badge.json
git commit -m "progress card"
git push
```

Then add this line to your README, replacing `<you>` with your GitHub username (and the repository name if you renamed your fork):

```text
![Full Send progress](https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2F<you>%2FForkable-AI-Engineering-Starter-Pack-To-Full-Send%2Fmain%2Fprogress-badge.json)
```

The badge shows whatever `progress-badge.json` says on your `main` branch, so re-run `card --badge` and push to update it.

The included GitHub Actions workflow also posts your board to each run's summary page. GitHub disables workflows in new forks until you open the **Actions** tab and enable them.

## Re-run on every save

```bash
python fullsend.py watch 1.1
```

`watch` judges the drill again each time you save `drill.py`. Add `--exit-on-clear` if you want it to stop when the judge prints CLEARED.

## Redo a drill

To start a drill over from scratch, run `reset`. It saves your current attempt as `drill.py.bak` and gives you a fresh stub.

```bash
python fullsend.py reset 2.1     # saves your current attempt as drill.py.bak, then starts fresh
```

## Make it your own

Teaching a team or a class? Fork it as your course repository and add drills with `python fullsend.py new <level> <slug>`. The guide is in [CONTRIBUTING.md](CONTRIBUTING.md). If you write something good, consider sending it upstream.
