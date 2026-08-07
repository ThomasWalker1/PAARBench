# Contributing to PAARBench

A submission is **an adaptation method with its hyperparameters chosen by the fixed
selection protocol** (`tunable:` in `method.yaml`), evaluated under that protocol,
with its result record included. Everything about *how* to write one is in
[`methods/README.md`](methods/README.md) — that is the document to read, and this one
does not duplicate it. What follows is the part that is about the pull request rather
than about the method.

## Before you start: what you need that is not in this repository

| artifact | how to get it | needed for |
|---|---|---|
| base + adapter checkpoints (~1.1 GB) | `scripts/download_checkpoints.py all` | everything |
| **goal files** `data/pushobj_eval/val_<shape>/plan_targets.pkl` | [`scripts/download_targets.py`](scripts/download_targets.py) → `ThomasWalker1/paarbench-data` | everything |
| training trajectories (~1 GB) | `scripts/download_data.py` | only methods that use offline data |

The goal files are the awkward one: they define each setting's episodes, they are not in git
and not in either Hub release, and nothing here regenerates them. Any run without them fails
up front with the list of missing files. If you cannot obtain them, open an issue rather
than working around it — a submission evaluated on episodes you generated yourself is not
comparable to anything on the board.

## Check your stack before you trust your method

The frozen baseline reproduces exactly, so use it as a self-test:

```bash
.venv/bin/python -m pytest -q                                    # ~6 s, no GPU
.venv/bin/python scripts/validate_method.py my_method            # no GPU
# ~2 min: one shape, 2 episodes. Should score whatever frozen scores.
.venv/bin/python scripts/eval_column.py --setting pushobj --cohort selection \
    --tag smoke --n-evals 2 --gpus 0
```

If a column reports `MEAN: incomplete` while also reporting `0 failed`, your outputs are not
landing where the harness reads them — that is a plumbing problem, not a method problem.

A full submission on `pushobj` is roughly: the standard selection sweep (3–9 cells
initially, plus any boundary expansions), plus one column per test cohort, plus the
frozen columns you need to pair against (see below). A batched method's column is ~2
min on 4 GPUs; an episode-isolated method's is `n_evals` times that.

## You will need to run the frozen baseline too

Every metric is defined against a frozen column on the *same* episodes, and the frozen
per-episode records are not distributed. So before your rows can have continuous columns:

```bash
.venv/bin/python scripts/evaluate.py --frozen --setting pushobj
```

That records **Frozen (Batched)**. If your method declares `requires_episode_isolation: true`,
you also need **Frozen (Individual)** — the two evaluation modes do not agree, and pairing
across them folds that difference into your method's effect:

```bash
.venv/bin/python scripts/evaluate.py --frozen --isolated --setting pushobj --per-gpu 4
```

## Do not regenerate `LEADERBOARD.md`

`scripts/leaderboard.py --out LEADERBOARD.md` recomputes the continuous columns from
per-episode records under `eval_outputs/`, which is git-ignored. If you have only your own
method and the frozen baseline on disk — which is the normal case — regenerating blanks
`median dist Δ`, `catastrophe`, `compounding`, `regret`, `adapt s/replan` and `peak MB` for
**every other method**, and shifts `vs frozen` for the episode-isolated rows. The script now
refuses to do that, and says which rows it would have damaged.

Instead: run `scripts/leaderboard.py --setting <s>` to stdout, and copy *your* row into the
existing table, leaving every other row byte-identical.

## What goes in the pull request

- your `methods/<name>/` directory, including a `README.md`
- the `results/<name>/<setting>.json` records `evaluate.py` produced
- your row(s) added to `LEADERBOARD.md`
- **no** changes to `results/frozen/*.json` or `results/frozen_isolated/*.json`. If your
  re-run of either baseline differs from the committed record, say so in the PR — that is a
  finding, not a file to overwrite.

If your method ships tests of its own, name the file so it cannot collide with another
method's (`test_<method>_*.py`); CI collects `methods/` as well as `tests/`.

### If you had to change the harness

`methods/README.md` says a method should need nothing outside its own directory, and that a
case where it does is a bug in the benchmark. That is still the rule. But if you hit a
blocker that had to be fixed for your submission to exist at all, put each fix in **its own
commit**, with the evidence for the diagnosis in the message, so it can be reviewed, split
off, or reverted independently of the method. Say in the PR description which commits are
the submission and which are the fixes.

## How submissions are verified

CI has no GPU. It runs `validate_method.py`, the GPU-free suite, and a specific check that
no selection rule can reach a test cohort — so it verifies that your submission is
*well-formed*, and cannot verify your numbers. Maintainers re-run submitted results before
merging. Two things follow:

1. Your record must be reproducible from what you committed: the `tunable` axes (or
   authored `params` if none), and the settings you declare.
2. Determinism matters. The planner is deterministic, and this is load-bearing — the paired
   metrics compare method against frozen episode by episode. If your method introduces
   nondeterminism, it must be seeded from something you declare.

## Reporting a problem instead of fixing it

If the interface failed to absorb your method, that is worth an issue on its own: say what
you needed and what you had to do instead. The extension point is supposed to bend.
