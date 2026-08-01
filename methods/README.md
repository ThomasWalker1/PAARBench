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

Full documentation of each hook, its exact tensor shapes, and why it is shaped that way:
[`paarbench/adapter.py`](../paarbench/adapter.py) and
[`docs/ADAPTER_PROTOCOL.md`](../docs/ADAPTER_PROTOCOL.md).

Three rules that the harness enforces rather than trusts:

- **`on_episode_start` must fully restore the base model.** Use `BaseWeightGuard`. A test
  asserts the base weights are bit-identical across an episode boundary for every
  registered method; if yours leaks, it fails CI. This is not pedantry — a method that
  quietly inherits the previous episode's correction produces results that look real.
- **You never see episode outcome.** No hook is passed success or distance-to-goal, and
  there is no back channel. An episode's outcome is only knowable once it is over;
  a method that needs outcomes to tune belongs in `selection.py`, below.
- **Everything is batched.** The whole cohort is planned as one batch, so every
  observation has a leading batch dimension and all episodes step in lockstep. If your
  method adapts per-episode, keep its state batched.

## 2. Declare it

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
        best, best_score = None, -1.0
        for lr in (1e-4, 5e-4, 1e-3):
            result = harness.run(lr=lr)            # one column on the selection cohort
            if result.success > best_score:
                best, best_score = {"lr": lr}, result.success
        return best                                # frozen, then run once on test
```

Every `harness.run` is counted, and the count is reported next to your score as your
**selection cost**. That is deliberate: a method needing a 16-cell sweep pays for it
visibly, and a method with no hyperparameters pays nothing. Off-the-shelf rules
(`GridSearch`, `FixedParams`) are in [`paarbench/selection.py`](../paarbench/selection.py).

**If your method has no hyperparameters, omit `selection` entirely.** The `params` in
your `method.yaml` are used as-is at a cost of zero columns.

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
