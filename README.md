# PAARBench — PlanActAdaptRepeatBench

A benchmark for **test-time adaptation (TTA) of latent world models**, scored so that the
hyperparameter-selection rule is part of the submission rather than hidden inside it.

> **Status: M0.** Scaffold and a reproducing frozen baseline. The adapter interface (M1),
> the output schema and metrics (M2), and the submission harness (M3) are not built yet.
> `docs/PLAN.md` is the design document and the source of every §-reference below.

## Why

Scoring a TTA method by success rate at its best-found hyperparameter conceals that the
hyperparameter dominates the method, that a wrong one *reverses* the adaptation signal
rather than merely wasting it, and that the damage compounds over the control loop. Within
a single 16-cell $(\eta, K)$ grid on one setting, closed-loop success spans 0.010 → 0.690
and median final distance-to-goal spans 12.7×. A per-episode oracle over that same grid
reaches 0.815 against the best *fixed* cell's 0.690 — headroom no current method captures.

So: **a submission is an adaptation method together with its hyperparameter-selection
rule.** The rule may read only the selection cohort; it is then frozen and run once on
held-out cohorts, and that is what the leaderboard reports (§3).

**The contribution is the protocol, not the environments.** The suite is thin (§2.3) and
the base models are not yet redistributable (§2.4).

**This is not a populated leaderboard, and is not trying to be one yet.** The goal is
infrastructure that makes proposing, implementing and evaluating a TTA method easy; entries
accumulate over time. The bar that follows from that is concrete: adding a method should
mean writing one adapter, one selection rule, and one config — and touching nothing else.

## Setup

Requires an NVIDIA driver supporting CUDA 12.1 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --python 3.9
.venv/bin/python -c "import torch; print(torch.cuda.is_available())"   # must be True
```

The cu121 torch index is pinned explicitly in `pyproject.toml`. Do not unpin it: default
resolution picks a build for a newer CUDA than many drivers support, and
`torch.cuda.is_available()` then silently returns `False` instead of erroring (§9).

PointMaze additionally needs `uv sync --extra pointmaze` (mujoco-py, d4rl) — but that
setting is currently disabled pending §2.3.

Base checkpoints and evaluation targets are not tracked in git. They are staged under
`checkpoints/` as inherited artifacts; `docs/CHECKPOINTS.md` is the manifest, and the
training code that reproduces them is deliberately deferred (§2.4).

## Running an evaluation column

A **column** is one (setting, cohort) evaluation over every shape in the setting. It is
the unit of evaluation cost, and §3 prices a submission's selection rule in columns.

```bash
.venv/bin/python scripts/eval_column.py \
    --setting pushobj --cohort selection --tag frozen --gpus 0,1,2,3
```

Shapes run one per GPU slot; a shape that already has `logs.json` is skipped, so an
interrupted column resumes. The training dataset is staged onto tmpfs first — the SMB
mount drops connections under a fan-out (§9). Results land in
`eval_outputs/<tag>/<setting>/<cohort>/`.

## Settings

Declared in `paarbench/settings.py`; cohort seeds are fixed by the benchmark, never by a
submission.

| id | base | shapes | selection | test | frozen |
|---|---|---|---|---|---|
| `pushobj` | `pushobj_shape_shift` | T, L, Z, + | seed 300 (n=200) | 100/200/400 (n=600) | 0.490 sel / 0.485 test |
| `pushobj_shift` | same | I, small_tee, square | — | seed 100 (n=150) | 0.293 |
| `pusht` | `pusht_visual_shift` | T, L, Z | seed 100 (n=150) | 200/400 | 0.360 |
| `pointmaze` | `pointmaze` | — | seed 300 | 0/1/2 | 0.880 / 0.733 — **disabled**, §2.3 |

Realistically that is **2 environments and 1 shift condition**. `pushobj_shift` is a
distribution-shift condition on the `pushobj` base, not a fourth environment.

The planner config is shared and fixed across methods (goal horizon 25, 100 GD steps,
`sample_type: zero`, `action_noise: 0`). **The planner being deterministic is
load-bearing** — it is what licenses per-episode paired comparison between methods. Do not
introduce planner stochasticity without also adding a noise-floor control.

## Layout

```
plan.py  planning/  env/  models/  datasets/  metrics/  conf/
    Ported from the predecessor at a pinned commit, kept as top-level modules so the
    port stays diffable against it.  See PROVENANCE.md.
paarbench/
    New benchmark code: settings registry, dataset staging, and (M1-M3) the adapter
    protocol, output schema, metrics, and submission harness.
scripts/eval_column.py
    Column driver.
docs/PLAN.md
    Design document.  Every decision in it has a recorded reason; if you change one,
    change the reason too.
docs/reference/
    Predecessor configs kept as the specification the M1 adapter port must match.
```

## Milestones

See `docs/PLAN.md` §8. Each has an acceptance criterion; do not advance without it.

- **M0 — scaffold.** *Accept:* a frozen PushObj column runs end to end and reports
  success 0.490 on seed 300.
- **M1 — the interface.** One `TestTimeAdapter` protocol; port online gradient TTA and the
  amortized hypernetwork onto it; collapse `planning/mpc.py`'s two hardcoded branches.
- **M2 — output schema + metrics.** One canonical per-episode/per-replan record; all seven
  §4 metrics with CIs.
- **M3 — harness + protocol.** Cohort separation enforced in code, selection cost recorded.
- **M4 — baselines.** Further methods, ongoing rather than a gate. Each port doubles as a
  usability test of M1/M3.
- **M5 — settings.** PointMaze and base-model distribution are deferred by decision (§2.3,
  §2.4); adding the deformable domain is the schedulable piece.
