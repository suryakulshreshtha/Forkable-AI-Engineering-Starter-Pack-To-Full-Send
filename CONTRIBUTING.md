# Contributing

Thanks for wanting to make this better. New drills, sharper tests, clearer briefs and bug fixes are all welcome.

## Add a drill in five steps

Use `python fullsend.py new` to add a new drill. Write its reference and its tests, then run `build` and `verify`. The verify command will check that references pass, stubs fail cleanly and docs are current.

```bash
python fullsend.py new 6 hybrid-search       # scaffolds levels/L6-.../04-hybrid-search/
# 1. write reference.py: docstring brief, then code with marked learner blocks
# 2. write test_drill.py: every test_ function receives the module as m
python fullsend.py build                      # regenerates READMEs from your docstring
python fullsend.py verify                     # must end with "all N drills healthy"
```

## The reference.py contract

The docstring is the single source for the drill's brief, its README and its hints:

```text
"""
Title: Hybrid Search
XP: 25

## The job
What the learner builds, with the exact function names and return shapes.

## Why it matters
Where this shows up in real AI engineering work.

## Hints
- A gentle nudge.
- A stronger one.
- Nearly the answer.
"""
```

The `verify` command checks that every reference passes its tests, that every generated stub imports and fails cleanly, and that the generated READMEs are current. In detail, it enforces these rules:

- The docstring has a `Title:` line and a `## Hints` list, and `## Hints` comes last.
- No backslashes or triple quotes inside the docstring (it is copied into the learner's stub).
- The reference passes every test.
- The generated stub imports cleanly and does not pass every test.
- Generated READMEs are up to date (run `build`).

Mark learner work inside function bodies only:

```python
def idf(self, term):
    # >>> fullsend: ln((N - n + 0.5) / (n + 0.5) + 1)
    n_t = self.df.get(term, 0)
    return math.log((self.n - n_t + 0.5) / (n_t + 0.5) + 1.0)
    # <<< fullsend
```

The label after `fullsend:` becomes the learner's `TODO` and the message of the `NotImplementedError`, so make it a useful nudge rather than a restatement of the function name.

## Writing good tests

- **The tests are the spec.** A learner should be able to clear the drill from the brief plus the test file.
- **Fail with a reason.** Give asserts a message that says what was expected and why.
- **Standard library only.** No NumPy, no pytest-only features in test bodies. Use seeded `random.Random` for anything random.
- **Fast.** Each drill should run in well under a second.
- **One idea per test.** Name tests as sentences: `test_ties_go_to_the_pair_seen_first`.

## Originality rules

This project must stay free of copied material. Before you open a pull request:

- Write prose, code and tests yourself. Do not paste from books, courses, blog posts or other repositories.
- If you adapt something under a compatible license, say so in the pull request and credit it in the drill.
- Credit the ideas: when a drill implements a published method, name the paper or author in the brief.
- If an AI assistant helped, review every line as if you wrote it, because you are the one submitting it.

More on this in [ORIGINALITY.md](ORIGINALITY.md).

## Testing the runner

The drills have their own judges. The runner, `fullsend.py`, and the capstone app have tests in `tests/`, standard library only:

```bash
python -m unittest discover -s tests -v
```

Run them whenever you touch `fullsend.py` or `levels/L9-full-send/app.py`. CI also runs the capstone evaluation as a gate (`app.py --eval --reference --min-score 1.0`), so a change to any Markdown file that makes PocketRAG answer worse fails the build. Fix the wording, or change `golden.json` on purpose, rather than lowering the bar to silence it.

## Opening a pull request from your fork

Your fork holds your own `drill.py` answers, and those must never reach upstream (a CI check enforces this). Make each pull request from a fresh branch based on upstream:

```bash
git fetch upstream
git switch -c my-new-drill upstream/main
# make your changes, commit, push the branch, open the pull request
```

## For maintainers

Keep the upstream repository free of `drill.py` files. If you want to work through the drills yourself, do it in a separate clone and never push those files. `python fullsend.py doctor` recognises the upstream clone and checks this for you.

## House rules

Be kind, be specific, assume good faith. Critique code, not people. Maintainers may close contributions that do not meet the rules above, and will say why.
