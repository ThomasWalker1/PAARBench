# PAARBench — PlanActAdaptRepeatBench

PAARBench evaluates test-time adaptation (TTA) strategies for latent world models in
closed-loop control. A submission consists of both an adaptation method and its
hyperparameter-selection rule: the rule may inspect only a fixed selection cohort,
then its frozen choice is evaluated once on held-out cohorts.

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

Base checkpoints, adapter checkpoints, datasets, and evaluation targets are too large
for git. Their expected paths and hashes are documented in
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

## Settings

Settings and cohort seeds are fixed in [`paarbench/settings.py`](paarbench/settings.py).

| id | base checkpoint | shapes | selection | held-out test |
|---|---|---|---|---|
| `pushobj` | `pushobj_shape_shift` | T, L, Z, + | seed 300 | seeds 100, 200, 400 |
| `pushobj_shift` | `pushobj_shape_shift` | I, small_tee, square | inherited from `pushobj` | seed 100 |
| `pusht` | `pusht_visual_shift` | T, L, Z | seed 100 | seeds 200, 400 |

`pushobj_shift` is a held-out shape condition, not a separate environment. It has no
selectable cohort; methods use the parameters selected on `pushobj`.

## Evaluate a method

Run the full protocol—selection, parameter freeze, held-out evaluation, and result
record—with:

```bash
.venv/bin/python scripts/evaluate.py adajepa \
  --setting pushobj --gpus 0,1,2,3
```

The command writes a compact record to `results/<method>/<setting>.json`. Raw episode
records and planner logs go under `eval_outputs/`, which is intentionally ignored by
git. Interrupted columns resume from completed units.

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

Copy `methods/_template/` and implement one adapter, one manifest, and optionally one
selection rule. No benchmark registry or planner changes are required.

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
| `env/`, `datasets/`, `models/` | PushObj/PushT simulation and world-model components |
| `scripts/evaluate.py` | protocol-enforcing submission evaluation |
| `scripts/leaderboard.py` | comparison table generated from `results/` |
| `tests/` | GPU-free protocol and interface checks |

The planner is fixed across methods (goal horizon 25, 100 gradient-descent steps,
zero action initialization, and no action noise). Determinism is required for paired
episode comparisons.
