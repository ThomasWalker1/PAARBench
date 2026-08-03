# Contributing a method

A method is a **self-contained directory in here**. Adding one means creating that
directory and nothing else — no registry to append to, no planner to edit, no harness
file to touch. If you find yourself needing to change something outside your own
directory to make your method work, that is a bug in the benchmark; please open an issue
saying what you needed, because the interface is supposed to absorb it.

## The shape of a method

```
methods/my_method/
    method.yaml     required   metadata, and the frozen hyperparameters
    adapter.py      required   your TestTimeAdapter implementation
    selection.py    optional   how those hyperparameters were chosen
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
  a method that needs outcomes to tune belongs in `selection.py`, below.
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

params:                         # the frozen hyperparameters, passed to __init__
  lr: 5.0e-4
  steps: 10

selection: selection:MyRule     # optional; omit if you have no hyperparameters
```

## 3. Declare how you chose the hyperparameters

This is the part the benchmark exists for. **A submission is a method together with its
hyperparameter-selection rule**, because a score at a best-found hyperparameter says
more about the search than about the method.

Your rule may read **only the selection cohort**, and it is handed an object that has no
way to reach a test cohort:

```python
class MyRule:
    def select(self, harness):
        best, least_harm = None, float("inf")
        for lr in (1e-4, 5e-4, 1e-3):
            result = harness.run(lr=lr)            # one column on the selection cohort
            if result.median_distance_delta < least_harm:
                best, least_harm = {"lr": lr}, result.median_distance_delta
        return best                                # frozen, then run once on test
```

Every `harness.run` is counted, and the count is reported next to your score as your
**selection cost**. That is deliberate: a method needing a 16-cell sweep pays for it
visibly, and a method with no hyperparameters pays nothing. Off-the-shelf rules
Each result exposes `success` plus paired `median_distance_delta`,
`catastrophe_rate`, and `compounding_slope` values measured against the harness-owned
frozen column on that same selection cohort. `GridSearch` accepts one of those names
as its `objective` (success is the default); the paired-harm objectives are minimized.
The frozen reference is never handed to the rule and never reaches a test cohort.
Off-the-shelf rules (`GridSearch`, `FixedParams`) are in
[`paarbench/selection.py`](../paarbench/selection.py).

**If your method has no hyperparameters, omit `selection` entirely.** The `params` in
your `method.yaml` are used as-is at a cost of zero columns.

### Which zero you are claiming

Omitting `selection` is honest for a method that genuinely has nothing to tune, and
misleading for one whose hyperparameters you simply wrote down. The leaderboard
distinguishes the two, so pick the one that is true of your submission:

| what the record says | what you are claiming |
|---|---|
| `0` | the method has no hyperparameters. Only the built-in `frozen` arm reports this. |
| `0 (authored)` | it has hyperparameters, and they were **authored, not selected** — no column was run to choose them. |
| `4`, `16`, … | a declared rule ran that many columns on the selection cohort. |
| `unknown` | a rule is declared but was skipped (`--skip-selection`), so the tuning happened somewhere this record cannot price. |

`0 (authored)` is a legitimate thing to submit — a method whose settings come from the
original paper has not tuned on this benchmark, and that is worth stating. What it is
not is the same claim as `frozen`'s `0`, which is why the column does not render them
identically. If your authored values came from a search you ran elsewhere, say so in
your `README.md`; the column can only price what the harness watched.

To move from `0 (authored)` to a real cost, write the rule that would have chosen those
values and let it run. That is also the cheapest bug-finder in the repo: running
`EpochSelection` for real is what caught a HyperJEPA checkpoint transcribed from the
wrong column of a results table (`docs/CHECKPOINTS.md`).

## 4. Check and run it

```bash
.venv/bin/python scripts/validate_method.py my_method       # no GPU needed
.venv/bin/python scripts/evaluate.py my_method --setting pushobj --gpus 0,1,2,3
```

`evaluate.py` runs your selection rule on the selection cohort, freezes what it returns,
then evaluates once on each test cohort and writes a result record. That record is what
populates the leaderboard.

## 5. Open a pull request

Include your directory and the result record `evaluate.py` produced. CI re-runs
`validate_method.py`, and rejects any selection rule that reaches for a test cohort.

## The frozen baseline

`frozen` is built in, not a directory here: it is `NullAdapter`, whose hooks do nothing.
It is reported on every setting as the do-nothing reference that every metric is measured
against. You do not need to add it, and you cannot override it.

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
