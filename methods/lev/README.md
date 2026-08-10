# LEV — closed-form latent offset with a held-out veto

Every other row on this board corrects the **weights**. `adajepa`, `adajepa_v2` and
`pad` walk an optimizer over them online; `hyperlora` and `static_lora` have a network
trained offline emit them. LEV corrects neither. The predictor is used exactly as it
was loaded, and the correction is an **offset added to what it outputs**, solved in
closed form from the transitions the controller has already executed this episode.

No gradient. No optimizer. No trained checkpoint. Nothing carried from one replan to
the next.

## What it does

For every executed model step the method has two things: what the frozen predictor said
the next latent would be (`x`), and what the encoder says it actually was (`y`). Per
latent channel, pooled over patches, it fits

    y - x  ≈  a·x + b

by ridge regression and adds that back to the predictor's output while the planner
solves. This is the disturbance observer of offset-free MPC, moved into a learned
latent space — with two things bolted on that classical control does not need and a
world model does.

**The fit is arithmetic.** Standardizing `x` per channel makes the slope and intercept
columns of the design orthogonal, so the normal equations are diagonal: the "fit" is six
sums and a division, with no matrix solve and no iteration. `ridge` shrinks the slope
only — the intercept is the pure offset and is the part with the most evidence behind
it. So the axis interpolates between ordinary least squares (`ridge → 0`) and the
classical per-channel offset (`ridge → ∞`), and the grid spans both.

**It is scaled by evidence it was not fitted on.** Because the buffer holds sufficient
statistics rather than latents, refitting with one step held out is a re-sum of a
handful of `(B, C)` tensors — no re-encoding, no forward pass, no iteration. LEV does
that for every step in the buffer, scores each resulting correction on the step it was
held out of, and compares it there against the frozen predictor. The fraction of
held-out squared error it removes,

    κ = clamp(1 − SSE_corrected / SSE_frozen, 0, 1),

multiplies the correction. A correction that cannot beat doing nothing out of sample is
multiplied by zero, and the planner solves through the frozen model. Nothing about this
is tuned; it is read off numbers already in memory, at a cost of no forward passes.

`adajepa_v2`'s `brake_ratio` is the same idea at its coarsest: one held-out sample, one
threshold, a binary reset. κ is the continuous, all-folds version, and it is cheaper
than the brake rather than more expensive.

**It decays over the planning horizon.** The fit and the veto are both one-step
quantities. The planner rolls the predictor forward 25 (Push) or 50 (maze) model steps
open loop, feeding each corrected prediction back in as the next step's context.
Trusting a one-step-validated offset equally at depth 25 is exactly what the
`compounding` column measures. LEV applies the correction at depth `t` with weight
`κ · decay^t`, so the planner falls back to the model it was trained with as it
extrapolates past its evidence.

## What it reports

`before_plan` runs no forward pass at all. Everything — one encoder pass per bounding
frame, at most `num_hist` predictor passes — happens in `on_transition`, once per
executed step, and is reduced immediately to six numbers per channel. The buffer holds
statistics, not tensors.

Nothing in the model is mutable, and the correction for episode *i* is solved from
episode *i*'s own statistics, so a batched cohort computes exactly what a per-episode
process would. **LEV does not declare `requires_episode_isolation`** — unlike all three
gradient methods, which need one process per episode.

Per-replan scalars: `lev/kappa` (mean trust scale), `lev/kappa_zero_frac` (episodes
planning frozen this replan), `lev/offset_norm`, `lev/slope_norm`, `lev/folds`,
`lev/buffer_size`.

## Hyperparameters

Two axes, nine initial cells — the same selection cost as `adajepa` and `adajepa_v2`, so
the rows are priced identically.

| axis | grid | what it does |
|---|---|---|
| `ridge` | 0.1, 1.0, 10.0 (log) | slope shrinkage; large ⇒ pure offset |
| `decay` | 0.8, 0.9, 1.0 (discrete) | per-model-step horizon trust decay |

`decay: 1.0` is in the grid deliberately: that cell is the method *without* its horizon
mechanism, so the mechanism has to win the cell rather than be assumed. Same reason
`adajepa_v2` keeps `horizon: 1` in its grid.

The trust scale κ is **not** an axis. It is recomputed at every replan from the
leave-one-out fit, which is the claim the method is making: it decides how far to trust
itself from evidence rather than from a constant chosen on a selection cohort.

`buffer_size` is authored at 15 executed model steps (three replans at the harness's
fixed `n_taken_actions` of 5), not tuned. An axis spent there would leave one of the two
mechanisms untested. For reference, `adajepa` and `hyperlora` buffer 5 and `adajepa_v2`
buffers 8; a longer buffer is free here because it stores six numbers per channel per
step rather than latents.

## Results

Held-out cohorts, `scripts/evaluate.py`, selection cost 9 columns on every setting.

| setting | success | vs frozen | rank | catastrophe | compounding | s/replan | peak MB |
|---|---|---|---|---|---|---|---|
| `pushobj` | 0.675 | +0.175 | 3 | **4.2%** | **+0.00** | 0.102 | 897 |
| `pushobj_shift` | 0.389 | +0.089 | 3 | **5.6%** | **+0.00** | 0.103 | 897 |
| `pusht` | 0.413 | +0.064 | 4 | 13.9% | **+0.00** | 0.098 | 897 |
| `maze_medium` | 0.820 | +0.047 | 4 | 0.0% | −0.00 | 0.125 | 603 |
| `maze_medium_low_density` | 0.793 | +0.047 | 3 | 22.7% | **+0.00** | 0.099 | 603 |
| `maze_medium_high_damping` | 0.727 | +0.160 | 4 | 0.0% | −0.00 | 0.096 | 603 |
| `maze_diverse` | 0.720 | +0.127 | 2 | 10.5% | **+0.00** | 0.109 | 603 |

LEV beats frozen on all seven and is generally third or fourth on success, behind the
two AdaJEPA arms. What it wins is the rest of the frontier. Compounding is `+0.00` with
a `[+0.00, +0.00]` interval on every setting — the correction is re-solved rather than
accumulated, so there is nothing to drift — against `+0.24`, `+0.97`, `+1.23` and
`+3.87` for the gradient arms. On the Push settings catastrophe is roughly a third of
theirs (4.2% against 17.1% and 14.4% on `pushobj`), and adaptation costs 0.10 s/replan
against 0.24–0.30 while running batched, so one process per shape rather than one per
episode.

### The selection grid is not separable

Worth knowing before reading a hyperparameter conclusion off this method. Two full runs
of the protocol were made, differing only in how many frames are handed to the visual
encoder per call — a change worth 5e-4 in the encoder output and nothing at all in the
method. **It flipped the selected cell on all seven settings.** Selection cost fell from
12–17 columns to 9, because in the first run the winner sat on a grid boundary
everywhere and in the second it sat in the interior everywhere. `maze_medium_high_damping`
moved 0.773 → 0.727, which is the difference between the top row of that table and
fourth.

The 3×3 surface is flat relative to run-to-run noise. The frozen-relative gains, the
compounding column and the Push catastrophe rates reproduced across both runs; the
choice of cell, and any single setting's rank, did not.

## What it cannot do, and what to distrust

- **The correction is affine, per latent channel.** It can remove a systematic bias or
  a gain drift in the predictor's output and nothing else. It is *not* limited by patch
  pooling, which an earlier draft of this file claimed: every base model the benchmark
  ships emits a single pooled visual token, so the latent is `(B, T, 1, 394)` and there
  is no spatial structure to lose.
- **The leave-one-out folds are single executed model steps, and adjacent steps within
  one chunk are consecutive states.** Holding one out is genuinely out-of-sample for the
  fit, but it is not independent, so κ is optimistic — biased towards applying the
  correction. It is a shrinkage estimate, not a hypothesis test, and it should be read
  that way.
- **The correction is inactive for the first two replans** (`before_plan` runs before
  the first `on_transition`, and the veto needs two steps to hold one out). LEV plans
  frozen until then rather than applying an unvalidated correction; on a 20-replan
  episode that is one replan of adaptation given up.
- **The one-step fit is evaluated at rollout depth 0 only.** Deeper in the rollout the
  predictor's input is itself corrected, which is a distribution the fit never saw.
  `decay` is a blunt instrument against that, not a solution.
- **`concat_dim == 1` only.** All four benchmark base models use it. A
  token-concatenated model raises rather than silently splitting the latent wrongly.
- **Peak memory is high for a method that adapts this cheaply**, at 897 MB on Push and
  603 MB on maze against frozen's 157. It is the visual encoder running over a whole
  batched cohort in `on_transition`, not anything the correction does: every predictor
  call in that hook together accounts for 14 MB. Encoding the chunk's frames in one call
  instead of one at a time made it 1510 MB, which is what an earlier version shipped.
  The floor is one ViT forward at `B = 50` and is not reducible for a batched method.
- **κ stays low for most of an episode and rises sharply near its end.** On `pushobj`
  it runs ≈0 for the first six replans, 0.14–0.43 through the middle, and 0.90+ at
  replans 19–20. The likely reading is that the residual is non-stationary while the
  object is being pushed and becomes consistent once motion settles, so the veto refuses
  to generalize across the non-stationary phase. An earlier draft predicted that this
  would confine LEV's effect to the safety columns; that prediction was wrong. LEV gains
  +0.175 success on `pushobj` while spending most of each episode near frozen.

## Files

| file | what |
|---|---|
| `adapter.py` | `LEVAdapter`; the statistics, the fit, the veto, the predictor hook |
| `method.yaml` | manifest and the two tunable axes |
| `../../tests/test_lev_fit.py` | GPU-free checks of the fit and the veto against direct refits |

The fit and the held-out score are algebra on sufficient statistics — they never build a
design matrix and never re-evaluate a correction on data. That is what makes them cheap,
and it is also exactly the kind of derivation that returns a plausible number when it is
wrong, so every identity is checked in `tests/test_lev_fit.py` against a direct
computation that *does* hold the data and *does* refit.

## Numerical guards

Two, both in the fit, both because the correction is installed in *raw* coordinates
rather than the standardized ones it is solved in.

- **The statistics are accumulated in double.** The fit recovers the variance as
  `m2/n − mean²` and the slope numerator as `q1 − mean·q0`, each a difference of
  like-sized quantities. In float32 a channel whose across-patch spread is small next to
  its magnitude loses most of its significant digits there. The tensors involved are
  `(B, C)`, so this costs nothing.
- **An unidentifiable slope is dropped, not divided by.** Installed, the correction is
  `slope/σ · x + (intercept − slope·mean/σ)`: as `σ → 0` both terms blow up and cancel
  on application, so a slope fitted on a channel with no spread does not merely add
  noise — it destroys the offset that *is* identifiable. Channels with
  `σ ≤ 1e-3 · (|mean| + σ)` keep their offset and lose their slope. The floor is applied
  inside the fit, so the held-out score grades the correction that is actually installed.
