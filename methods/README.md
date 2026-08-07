# Contributing a method

A method is a **self-contained directory in here**. Adding one means creating that
directory and nothing else — no registry to append to, no planner to edit, no harness
file to touch. If you find yourself needing to change something outside your own
directory to make your method work, that is a bug in the benchmark; please open an issue
saying what you needed, because the interface is supposed to absorb it.

## The shape of a method

```
methods/my_method/
    method.yaml     required   metadata, frozen hyperparameters, and tunable axes
    adapter.py      required   your TestTimeAdapter implementation
    README.md       expected   what it is, and what it reports
```

Start by copying the template:

```bash
cp -r methods/_template methods/my_method
```

## 1. Write the adapter

Implement four hooks. The planner calls them; you never call the planner.

```python
class MyAdapter:
    def __init__(self, wm, preprocessor, **params):
        self.wm, self.preprocessor = wm, preprocessor
        self.guard = BaseWeightGuard(wm)          # snapshot before mutating anything

    def on_episode_start(self, obs_0, goal):
        self.guard.restore()                       # required: leave no trace
        return {}

    def on_transition(self, obs_0, actions, rollout_obs, frameskip):
        ...                                        # buffer what you saw
        return {}

    def before_plan(self, obs):
        ...                                        # apply/refresh your correction
        return {}

    def metrics(self):
        return {}                                  # per-replan scalars to record
```

Full documentation of each hook and its tensor shapes lives in
[`paarbench/adapter.py`](../paarbench/adapter.py).

Three rules that the harness enforces rather than trusts:

- **`on_episode_start` must fully restore the base model.** Use `BaseWeightGuard`. A test
  asserts the base weights are bit-identical across an episode boundary for every
  registered method; if yours leaks, it fails CI. This is not pedantry — a method that
  quietly inherits the previous episode's correction produces results that look real.
- **You never see episode outcome.** No hook is passed success or distance-to-goal, and
  there is no back channel. An episode's outcome is only knowable once it is over;
  hyperparameter selection belongs in `method.yaml`, not in the adapter.
- **Everything is batched — unless you say otherwise.** The whole cohort is planned as
  one batch, so every observation has a leading batch dimension and all episodes step in
  lockstep. That is fine for a method whose state is naturally batched (a per-episode
  correction emitted from a per-episode context, say).

  It is **wrong** for a method that owns mutable shared state — model weights plus an
  optimizer. One gradient step would average 50 unrelated episodes into a single
  correction, which is a different method from per-episode adaptation, and it fails
  quietly: you get a plausible number that is simply not what you meant to measure. If
  that describes your method, set

  ```yaml
  requires_episode_isolation: true
  ```

  and the harness fans out one process per episode instead of one per shape. It costs
  `n_evals`× the processes, so use `--per-gpu 4` or similar; do not set it unless
  batching genuinely changes what your method computes.

## 2. Declare it

**Shipping a checkpoint?** Write its path relative to the repository root. Any param key
ending in `_path` or `_dir` whose value names a real repo-relative file is resolved for
you before your adapter is constructed. This exists because Hydra chdirs into the run
directory before the planner is built, so the obvious relative path would otherwise
resolve somewhere under `eval_outputs/` and fail.

`method.yaml`:

```yaml
name: my_method                 # must equal the directory name
display_name: "My Method (what it does)"
description: >
  One paragraph. What it adapts, and what it is trying to fix.
reference: "Author et al., 2025, arXiv:..."      # optional

adapter: adapter:MyAdapter      # <module>:<attribute>, resolved in this directory
settings: [pushobj, pusht]      # which settings you claim to support

params:                         # fixed defaults; selection overrides tunable keys
  lr: 5.0e-4
  steps: 10
```

## 3. Declare tunable hyperparameters (if any)

**A submission is a method together with how its hyperparameters were chosen.** The
benchmark fixes that procedure so selection cost is comparable across methods:

1. Declare **at most two** tunable axes under `tunable:` in `method.yaml`.
2. On the **selection cohort** only, evaluate a **3-point grid** per axis (3 cells for
   one axis, 3×3 for two).
3. If the best cell lies on a grid boundary, **expand** that axis outward (log/linear
   step for continuous values; adjacent pool members for discrete axes). Repeat at most
   **twice** (default `max_expansions: 2`).
4. **Freeze** the best configuration and evaluate **once** on each held-out test cohort.

Every evaluated grid cell is counted as **selection cost** and reported next to your
score. The harness enforces cohort separation structurally: selection never sees test
seeds.

### Continuous axes (1 or 2)

```yaml
tunable:
  objective: success            # optional: success (default), median_distance_delta,
                                # catastrophe_rate, or compounding_slope
  axes:
    pred_lr:
      initial: [5.0e-4, 2.0e-3, 1.0e-2]
      scale: log
    steps:
      initial: [1, 5, 10]
      scale: linear
```

### Discrete axes (e.g. checkpoint epoch)

Map a semantic axis to an adapter parameter with `maps_to`:

```yaml
tunable:
  axes:
    training_epoch:
      scale: discrete
      initial: [1, 2, 3]
      pool_by_setting:
        pushobj: [1, 2, 3, 4, 5]
        pusht: [1, 2, 3, 4]
      maps_to:
        param: checkpoint_path
        by_setting:
          pushobj: checkpoints/.../hyper_lora_epoch_{}.pth
          pusht: checkpoints/.../hyper_lora_epoch_{}.pth
```

Implementation: [`paarbench/tunable.py`](../paarbench/tunable.py).

### No tunable hyperparameters

Omit `tunable` entirely. The `params` in your `method.yaml` are used as-is at a cost of
zero columns — but only if they truly are not hyperparameters you chose by inspecting
benchmark episodes.

### Which zero you are claiming

| what the record says | what you are claiming |
|---|---|
| `0` | the method has no hyperparameters. Only the built-in `frozen` arm reports this. |
| `0 (authored)` | it has hyperparameters, and they were **authored, not selected** — no column was run to choose them. |
| `3`, `9`, … | the standard protocol ran that many columns on the selection cohort. |
| `unknown` | selection was skipped (`--skip-selection`), so the tuning happened somewhere this record cannot price. |

`0 (authored)` is legitimate for a method whose settings come from the original paper.
If your authored values came from a search you ran elsewhere, say so in your `README.md`.

## 4. Check and run it

```bash
.venv/bin/python scripts/validate_method.py my_method       # no GPU needed
.venv/bin/python scripts/evaluate.py my_method --setting pushobj --gpus 0,1,2,3
```

`evaluate.py` runs the standard selection protocol on the selection cohort, freezes what
it returns, then evaluates once on each test cohort and writes a result record. That
record is what populates the leaderboard.

## 5. Open a pull request

Include your directory and the result record `evaluate.py` produced. CI re-runs
`validate_method.py`, and rejects any selection rule that reaches for a test cohort.

## The frozen baseline

`frozen` is built in, not a directory here: it is `NullAdapter`, whose hooks do nothing.
It appears on every setting as two leaderboard arms:

- **Frozen (Batched)** — `scripts/evaluate.py --frozen` — one process per shape cohort.
- **Frozen (Individual)** — `scripts/evaluate.py --frozen --isolated` — one process per
  episode, tagged `frozen_isolated` on disk.

Batched methods pair against Batched; episode-isolated methods pair against Individual.
The modes are not interchangeable even for NullAdapter. You do not need to add either
arm, and you cannot override them.

## Carrying different weights per setting

A method's architecture is usually shared across domains; its trained weights are not.
Declare per-setting overrides rather than duplicating the method:

```yaml
params:
  checkpoint_path: checkpoints/my_method/pushobj/best.pth
  rank: 2

params_by_setting:
  pusht:
    checkpoint_path: checkpoints/my_method/pusht/best.pth
```

`params_by_setting` is layered on top of `params` for that setting only, and every key
in it must name a setting you declared in `settings`. Duplicating the method instead
would split its results across two leaderboard rows for no reason.
