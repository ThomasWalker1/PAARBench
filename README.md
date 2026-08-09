# PAARBench — PlanActAdaptRepeatBench

PAARBench evaluates test-time adaptation (TTA) strategies for latent world models in
closed-loop control. A submission declares up to two tunable hyperparameters in
`method.yaml`; the harness runs a fixed selection protocol (3-point grid per axis on
the selection cohort, up to two boundary expansions, then freeze) before evaluating
once on held-out test cohorts.

The benchmark reports a metric frontier rather than a single rank. Alongside success
rate it measures paired final-distance change, catastrophe rate, compounding error,
regret, adaptation latency, peak memory, and selection cost. See
[`LEADERBOARD.md`](LEADERBOARD.md) for recorded comparisons.

## Setup

PAARBench currently targets Linux, Python 3.9, and an NVIDIA driver compatible with
CUDA 12.1. Install [uv](https://docs.astral.sh/uv/), then run:

```bash
uv sync --extra dev
.venv/bin/python -c "import torch; print(torch.cuda.is_available())"
.venv/bin/python -m pytest -q
```

Base and adapter checkpoints are hosted outside git. Their expected paths and hashes
are documented in
[`docs/CHECKPOINTS.md`](docs/CHECKPOINTS.md); the exact artifact inventory is pinned in
[`CHECKPOINTS.sha256`](CHECKPOINTS.sha256).

Download only the base JEPA models needed by benchmark settings, only method-specific
adapter checkpoints, or the complete artifact set:

```bash
.venv/bin/python scripts/download_checkpoints.py settings
.venv/bin/python scripts/download_checkpoints.py methods
.venv/bin/python scripts/download_checkpoints.py all
```

Evaluating an adaptation method requires both groups, so use `all` on a new checkout.
Each download is filtered to its group and verified against `CHECKPOINTS.sha256`.
PushObj/PushT evaluation targets — the goal files defining those settings' episodes —
are **not** tracked in git. Fetch them from Hugging Face Hub and verify against
`TARGETS.sha256`:

```bash
.venv/bin/python scripts/download_targets.py
```

Maze episode corpora under `data/maze_eval/` are tracked in git. Maze MuJoCo setup,
Drive-sourced trajectory staging, and adapter training are documented in
[`docs/MAZE.md`](docs/MAZE.md).

A run missing any goal file fails immediately, naming the files, instead of launching one
process per shape and letting each die separately. See
[`docs/CHECKPOINTS.md`](docs/CHECKPOINTS.md).

Standard evaluation does not require the training dataset. To reproduce training or
develop a method that explicitly uses offline trajectories, download the separately
versioned and verified PushObj/PushT dataset (maze offline data is staged per
[`docs/MAZE.md`](docs/MAZE.md)):

```bash
.venv/bin/python scripts/download_data.py
```

See [`docs/DATASET.md`](docs/DATASET.md) for its contents and provenance.

## Settings

Settings and cohort seeds are fixed in [`paarbench/settings.py`](paarbench/settings.py).

| id | base checkpoint | shapes | selection | held-out test |
|---|---|---|---|---|
| `pushobj` | `pushobj_shape_shift` | T, L, Z, + | seed 0 | seeds 100, 200, 300 |
| `pushobj_shift` | `pushobj_shape_shift` | I, small_tee, square | inherited from `pushobj` | seeds 100, 200, 300 |
| `pusht` | `pusht_visual_shift` | T, L, Z | seed 0 | seeds 100, 200, 300 |
| `maze_medium` | `mediummaze_dynamics_shift` | medium maze | seed 0 | seeds 100, 200, 300 |
| `maze_medium_low_density` | same as `maze_medium` | density scale 0.2 | inherited | seeds 100, 200, 300 |
| `maze_medium_high_damping` | same as `maze_medium` | damping scale 20 | inherited | seeds 100, 200, 300 |
| `maze_diverse` | `mediummaze_dynamics_shift` | held-out DiverseMaze layouts | inherited from `maze_medium` | seeds 100, 200, 300 |

`pushobj_shift` is a held-out shape condition, not a separate environment. It has no
selectable cohort; methods use the parameters selected on `pushobj`.

The maze shifts share the PointMaze Medium base and parameters selected on
`maze_medium`. Their evaluation episodes are immutable, generated target corpora
rather than runtime samples. See
[`docs/MAZE.md`](docs/MAZE.md) for MuJoCo setup, artifact staging, target generation,
and evaluation commands.

## Evaluate a method

Run the full protocol—standard hyperparameter selection on the selection cohort,
parameter freeze, held-out evaluation, and result record—with:

```bash
.venv/bin/python scripts/evaluate.py adajepa \
  --setting pushobj --gpus 0,1,2,3
```

The command writes a compact record to `results/<method>/<setting>.json`. Held-out
per-episode records (`eval_outputs/<method>/<setting>/test*/**/episodes.jsonl`) are
tracked so continuous leaderboard columns regenerate from a fresh clone; selection
sweeps, planner logs, and target copies stay git-ignored (see
[`docs/EVAL_RECORDS.md`](docs/EVAL_RECORDS.md)). Interrupted columns resume from
completed units.

For debugging one fixed configuration without running the submission protocol:

```bash
.venv/bin/python scripts/eval_column.py \
  --setting pushobj --cohort selection --tag debug --gpus 0
```

Regenerate the comparison table from checked-in result records with:

```bash
.venv/bin/python scripts/leaderboard.py --out LEADERBOARD.md
```

## Add an adaptation strategy

Copy `methods/_template/` and implement one adapter, one manifest, and optionally
`tunable` axes. No benchmark registry or planner changes are required.

```bash
cp -r methods/_template methods/my_method
.venv/bin/python scripts/validate_method.py my_method
```

The adapter lifecycle and contribution requirements are documented in
[`methods/README.md`](methods/README.md). Mutable methods must declare
`requires_episode_isolation: true` so updates cannot mix unrelated episodes.

## Repository layout

| path | purpose |
|---|---|
| `paarbench/` | settings, adapter protocol, selection harness, schema, metrics, runner |
| `methods/` | self-contained adaptation strategies and contribution template |
| `planning/`, `plan.py` | closed-loop MPC evaluation |
| `env/`, `datasets/`, `models/` | PushObj/PushT/PointMaze simulation and world-model components |
| `scripts/evaluate.py` | protocol-enforcing submission evaluation |
| `scripts/leaderboard.py` | comparison table generated from `results/` |
| `docs/MAZE.md` | maze install, artifact staging, training, and evaluation |
| `tests/` | GPU-free protocol and interface checks |

The planner is fixed within each setting family (Push*: goal horizon 25; maze: 50),
with 100 gradient-descent steps, zero action initialization, and no action noise.
Determinism is required for paired episode comparisons.
