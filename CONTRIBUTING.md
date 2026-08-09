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
| base + adapter checkpoints | `scripts/download_checkpoints.py all` (see [`docs/CHECKPOINTS.md`](docs/CHECKPOINTS.md)) | everything |
| **Push goal files** `data/pushobj_eval/val_<shape>/plan_targets.pkl` | [`scripts/download_targets.py`](scripts/download_targets.py) → `ThomasWalker1/paarbench-data` | push settings |
| **Maze episode corpora** `data/maze_eval/*/seed_*.pkl` | tracked in git (or `scripts/generate_maze_targets.py`) | maze settings |
| Maze MuJoCo + Drive data | [`docs/MAZE.md`](docs/MAZE.md), verify with `MAZE_ARTIFACTS.sha256` | maze settings |
| training trajectories | `scripts/download_data.py` (push); maze via [`docs/MAZE.md`](docs/MAZE.md) | only methods that use offline data |

Push goal files define each push setting's episodes; they are not in git, and nothing
here regenerates them. Maze corpora are small and tracked. Any run without the required
targets fails up front with the list of missing files. If you cannot obtain Push goal
files, open an issue rather than working around it — a submission evaluated on episodes
you generated yourself is not comparable to anything on the board.

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

## Frozen baselines are already in the tree

Every continuous metric is defined against a frozen column on the *same* episodes.
Held-out `episodes.jsonl` for published methods — including **Frozen (Batched)** and
**Frozen (Individual)** — are tracked under `eval_outputs/` (see
[`docs/EVAL_RECORDS.md`](docs/EVAL_RECORDS.md)). You do **not** need to re-run frozen just
to populate continuous columns on a fresh clone.

You may still re-run frozen as a self-test of your stack. If your method declares
`requires_episode_isolation: true`, pair against **Frozen (Individual)** — the two
evaluation modes do not agree, and pairing across them folds that difference into your
method's effect:

```bash
.venv/bin/python scripts/evaluate.py --frozen --setting pushobj
.venv/bin/python scripts/evaluate.py --frozen --isolated --setting pushobj --per-gpu 4
# Maze (see docs/MAZE.md):
.venv/bin/python scripts/evaluate.py --frozen --setting maze_medium --gpus 0,1,2,3,4,5,6,7
.venv/bin/python scripts/evaluate.py --frozen --isolated --setting maze_medium \
  --gpus 0,1,2,3,4,5,6,7 --per-gpu 4
```

## Regenerating `LEADERBOARD.md`

`scripts/leaderboard.py --out LEADERBOARD.md` recomputes continuous columns from
`eval_outputs/**/episodes.jsonl`. With the tracked published records present, a full
regenerate is safe once your method's held-out episodes are on disk too.

If some published records are missing locally, regenerating blanks those rows' continuous
columns. The script refuses to overwrite `LEADERBOARD.md` in that case. Safer for a PR:
run `scripts/leaderboard.py --setting <s>` to stdout and copy *your* row into the existing
table, or commit your `episodes.jsonl` files and regenerate with every method present.

## What goes in the pull request

- your `methods/<name>/` directory, including a `README.md`
- the `results/<name>/<setting>.json` records `evaluate.py` produced
- your held-out `eval_outputs/<name>/<setting>/test*/**/episodes.jsonl` files (not
  selection sweeps, planner logs, or `plan_targets.pkl`)
- your row(s) added to `LEADERBOARD.md` (or a regenerate that keeps other rows intact)
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
