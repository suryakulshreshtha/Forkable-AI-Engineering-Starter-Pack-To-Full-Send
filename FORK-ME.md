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

## Save your progress

Commit each drill when the judge clears it. The check command prints the exact line to run.

```bash
git add levels/L1-numbers-that-learn/01-dot-and-matmul/drill.py
git commit -m "clear 1.1"
git push
```

Your commit history becomes a timeline of what you built and when.

## Stay in sync

To pull new drills and fixes into your fork, add the original repository as a second remote called `upstream`, once:

```bash
git remote add upstream https://github.com/suryakulshreshtha/Forkable-AI-Engineering-Starter-Pack-To-Full-Send.git
```

Then, whenever you want new drills or fixes:

```bash
git fetch upstream
git merge upstream/main
python fullsend.py init     # creates drill.py files for any new drills; never overwrites yours
```

You will only see conflicts if you edited a file upstream also owns, such as a `test_drill.py` or a `reference.py`. Keep your changes inside `drill.py` files and new files of your own, and merges stay clean.

## Show it off

```bash
python fullsend.py card     # writes PROGRESS.md from the judge's verdicts
git add PROGRESS.md && git commit -m "progress card" && git push
```

Link `PROGRESS.md` from your profile or résumé. Every check mark in it was earned by running the judge against your own code.

The included GitHub Actions workflow also posts your board to each run's summary page. GitHub disables workflows in new forks until you open the **Actions** tab and enable them.

## Redo a drill

To start a drill over from scratch, run `reset`. It saves your current attempt as `drill.py.bak` and gives you a fresh stub.

```bash
python fullsend.py reset 2.1     # saves your current attempt as drill.py.bak, then starts fresh
```

## Make it your own

Teaching a team or a class? Fork it as your course repository and add drills with `python fullsend.py new <level> <slug>`. The guide is in [CONTRIBUTING.md](CONTRIBUTING.md). If you write something good, consider sending it upstream.
