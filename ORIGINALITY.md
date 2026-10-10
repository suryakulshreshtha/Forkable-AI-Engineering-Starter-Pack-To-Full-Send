# Originality and Provenance

This file records where this project came from, what it owes to others, and how it stays clean. If you ever need to answer "did you copy this?", point here.

## Statement

Forkable AI Engineering: Starter Pack to Full Send is an independently written curriculum. All prose, drill designs, tests, reference code, the `fullsend.py` runner, the `kit/` modules, the level structure and the naming were written for this repository. No files, text, lesson outlines or code were copied from any other course, book or repository.

## The genre, and what inspired it

"Learn AI engineering by building it from scratch, in a public repository" is a well-established format with many good examples. One of them, [rohitg00/ai-engineering-from-scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (MIT licensed), prompted this project: its public README was read once to understand the format, and this repository was then designed deliberately differently from a blank page. Nothing from that repository was copied.

| | This repository |
| --- | --- |
| Unit of work | Drills: one `reference.py` that generates the brief, README, hints and learner stub |
| Learner workflow | Edit a generated `drill.py`, judged by `fullsend.py check` |
| Fork model | Learner files are generated locally and never shipped upstream, so syncing never conflicts |
| Dependencies | Python standard library only, by rule |
| Progress | `PROGRESS.md` written from the judge's verdicts |
| Capstone | A cited-answer engine that answers questions about this repository |
| Scope | Nine focused levels, with growth tracked in [ROADMAP.md](ROADMAP.md) |

## Ideas we build on

The drills implement published, widely taught methods. Implementing an algorithm from its description is not copying anyone's implementation, but the people who invented these ideas deserve credit:

- Gradient descent with momentum: B. T. Polyak (1964)
- Backpropagation: D. Rumelhart, G. Hinton and R. Williams (1986)
- Byte-pair encoding: P. Gage (1994); for subword tokenization, R. Sennrich, B. Haddow and A. Birch (2016)
- The distributional hypothesis: Z. Harris (1954) and J. R. Firth (1957)
- Attention: D. Bahdanau, K. Cho and Y. Bengio (2015); scaled dot-product attention, A. Vaswani et al. (2017)
- Nucleus (top-p) sampling: A. Holtzman et al. (2020)
- BM25: S. Robertson and colleagues; see Robertson and Zaragoza (2009)
- Reciprocal rank fusion: G. Cormack, C. Clarke and S. Büttcher (2009)

## How the first version was made

The first version was drafted with the help of an AI assistant, working from a blank page with the brief "make an original, fork-friendly AI engineering curriculum". Every drill was then checked by the repository's own `verify` command: each reference passes its judge and each generated stub fails cleanly.

## Keeping it clean

Contributors agree to the originality rules in [CONTRIBUTING.md](CONTRIBUTING.md): write your own prose and code, credit ideas, and declare anything adapted from a compatible license. Maintainers reject contributions that do not.

If you believe something here was copied from your work, open an issue with links to both. It will be investigated promptly and, if confirmed, removed or properly credited.

## License

MIT. See [LICENSE](LICENSE).
