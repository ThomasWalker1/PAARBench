# AdaJEPA v2 — horizon-matched fit + fresh-evidence brake

**This is version 2 of [`methods/adajepa`](../adajepa/), not a new family.** It adapts
the same thing v1 adapts — a rank-2 LoRA plus a LayerNorm delta on the predictor's last
block, fitted online with AdamW on the world model's own prediction error — with the
same optimizer, the same clipping and the same selection cost. Three things about *how*
it is fitted change. It carries no offline training and no extra checkpoint; everything
it needs is the base model and the episode it is in.

Read [`methods/adajepa/README.md`](../adajepa/README.md) before reading the numbers
here. The v1 row is a port of a predecessor project's re-implementation, and it fits a
one-frame window, where the published AdaJEPA implementation fits `min(num_hist, T)`-frame
teacher-forced windows over the whole episode. "v2 against v1" is therefore a comparison
between two arms *on this board*, and the published method's own fitting rule is a third
arm neither of them measures yet.

## Why these three changes

**The objective is the rollout the planner solves through, not one step of it.**
The planner rolls the predictor forward `goal_H / frameskip` model steps open loop
(5 on the Push settings), feeding each prediction back in as context, and scores the
end of that rollout against the goal. v1 fits single-step prediction from a single
frame, so a correction that is accurate for one step and drifts over five looks perfect
to it and plans badly. v2 rolls the correction forward `horizon` steps under the actions that
were really executed, against the states the environment really reached, and
accumulates loss at every depth — mirroring `VWorldModel.rollout`, including its
growing `num_hist` context window, so the thing being fitted is the thing being used.

At `horizon: 1` this reduces to **v1's** objective. That value is in the selection grid
deliberately: if the multi-step objective does not earn its place, the protocol picks
the arm that does not use it. It is *not* the published implementation's objective,
which is one-step-ahead but teacher-forced over a `num_hist` window — so the gain
recorded below is a gain over v1, and how much of it is the multi-step rollout rather
than the restored context is not yet measured. See *Open ablations*.

**The update is braked by evidence it has not seen.** v1's correction is an
optimizer trajectory that only accumulates: on `pushobj` it ends more than 2× further
from the goal than frozen on 17.1% of episodes, and its paired distance gap grows with
replan index (compounding +0.24). Nothing inside the method can notice, because an
episode's outcome is not observable and its buffer only ever holds data it has already
fitted.

The transition that arrives at replan *n* is data the update at replan *n−1* was
fitted before. v2 scores the current correction against the frozen predictor on
exactly that transition, before folding it into the fit — out-of-sample by
construction, no held-out split, no outcome. When the correction is worse there by
more than `brake_ratio` (1.25), the delta is set back to zero and the optimizer
rebuilt: adaptation restarts from the frozen model rather than compounding a
correction that fresh prediction error says is hurting.

The check is *single-step*, even though the fit is multi-step, because that is the only
quantity available here whose input and target were both unseen: a deeper rollout
ending at the newest state would have to start inside the buffer and pass through
states the last update was fitted on, which biases the comparison in the adapted arm's
favour — exactly the direction that would make the brake fail to fire.
`adajepa_v2/braked` and
`adajepa_v2/fresh_loss_ratio` are recorded per replan, so how often this fires — and
whether adaptation is helping out-of-sample at all — is visible in the record rather
than inferred.

**Target latents are cached, so a gradient step is predictor-only.** Only predictor
LoRA parameters are trained, so every encoding in the buffer is a constant for the
rest of the episode. v1 re-encodes its whole buffer inside every gradient step
(`steps × buffer_size` encoder forwards per replan); v2 encodes each transition
once when it arrives — two forwards per replan, independent of `steps` — which is what
makes a multi-step objective over an 8-transition buffer cheaper than v1's
single-step one over a 5-transition buffer.

## What is held fixed on purpose

The parameterization (`predlast_all`, rank 2, `lora_scale` 1.0), AdamW, and
`grad_clip_norm: 1.0` are v1's, unchanged. The comparison between the two rows is meant
to be about the fitting rule and the brake, not about capacity or optimizer tuning.

None of these are the *published* AdaJEPA's either: that implementation adapts the
predictor's last layer densely plus the encoder head, with a fresh optimizer each
replan. Both rows on this board inherit the predecessor's LoRA parameterization, so
neither is evidence about the paper's parameter choice.

## Results

All seven settings. Selection ran independently on the three selectable ones and froze
`horizon: 5` on every one of them; the learning rate it chose differs by domain
(`2e-3` on both push settings, `1e-2` on `maze_medium`, where the axis expanded twice).
At the selected learning rate, `horizon: 5` beat `horizon: 1` on all three — 0.685 vs
0.670 on `pushobj`, 0.427 vs 0.407 on `pusht`, 0.900 vs 0.800 on `maze_medium` — so the
multi-step objective is what the protocol picked, not what it tolerated.

| setting | v2 success | v1 | median dist Δ | compounding | catastrophe | regret | adapt s/replan |
|---|---|---|---|---|---|---|---|
| `pushobj` | **0.697** | 0.697 | +8 vs +6 | +0.22 vs +0.24 | 14.4% vs 17.1% | 56%/+156 vs 57%/+256 | 0.239 vs 0.294 |
| `pushobj_shift` | **0.409** | 0.400 | +1 vs +16 | +0.27 vs +0.97 | 15.9% vs 16.9% | 51%/+204 vs 59%/+205 | 0.239 vs 0.290 |
| `pusht` | **0.478** | 0.440 | +16 vs +55 | +1.23 vs +3.87 | 22.2% vs 26.8% | 55%/+274 vs 66%/+549 | 0.238 vs 0.293 |
| `maze_medium` | 0.847 | **0.853** | -0 vs +0 | +0.01 vs +0.03 | 0.0% vs 0.0% | 56%/+1 vs 82%/+2 | 0.247 vs 0.296 |
| `maze_medium_low_density` | 0.773 | **0.827** | +0 vs +4 | +0.02 vs +0.04 | **6.2% vs 26.7%** | 50%/+4 vs 73%/+7 | 0.239 vs 0.298 |
| `maze_medium_high_damping` | 0.740 | **0.747** | -0 vs +0 | +0.00 vs +0.00 | 0.0% vs 0.0% | 48%/+2 vs 52%/+2 | 0.239 vs 0.298 |
| `maze_diverse` | **0.727** | 0.640 | -1 vs -1 | **-0.09 vs -0.06** | 7.7% vs 5.9% | **19%/+2 vs 29%/+2** | 0.239 vs 0.296 |

Pooled over all 2,100 held-out episodes: **0.6095 vs 0.5981**. That pooled number is not
the interesting part — the settings are not commensurable and the benchmark says so.

**Where v2 wins, and why it is the same reason each time.** The gain tracks how far v1's
unbraked correction drifts, not how hard the setting is. It is largest on `maze_diverse`
(+0.087 over v1, ~2.3 SE, and +0.133 over frozen — the largest success gain any method
records there) and on `pusht` (+0.038), the two settings where a correction fitted to
one distribution is used on another. It is an exact tie on `pushobj` and within one SE
on `maze_medium` and `maze_medium_high_damping` — the three settings where v1 records
0.0% catastrophe or a compounding interval touching zero, i.e. where there is nothing
for a brake to prevent. This was the prediction made before the maze runs, and it held.

**Where v2 loses, and what it buys.** On `maze_medium_low_density` v2 gives up 0.054
success (0.773 vs 0.827, ~1.2 SE, intervals overlapping) and cuts catastrophe from
26.7% to 6.2%, regret frequency from 73% to 50%, and the median paired distance change
from +4 to +0. This is the one setting in the suite where v1's willingness to keep a
drifting correction *pays* in success, and it is a real trade rather than a wash: a
deployment that would rather not quadruple its distance-to-goal on a quarter of
episodes should read this row as v2 winning, and one optimizing success alone should
not. It is exactly the row the benchmark's multi-objective framing exists to make
legible.

v2 does not match HyperJEPA's risk profile on `pusht` (compounding −0.07, catastrophe
7.7%, median distance change −0) despite scoring above it on success (0.478 vs 0.453).
A correction recomputed from frozen weights every replan compounds less than one that is
braked, and the brake does not close that gap.

**Diagnostics the records do not carry.** On `pushobj`, fitting lowers *out-of-sample*
single-step prediction error to a median 0.77× frozen at `horizon: 1`, rising to 0.92×
at `horizon: 5` — the multi-step objective spends single-step accuracy on rollout
consistency, as intended. The brake's behaviour is domain-dependent in a way worth
knowing: on push it fires on 7–12% of episodes (0.1–0.2 times per episode), while on the
maze settings the fresh-evidence ratio has a median near 1.0 and the brake fires 1.4–2.1
times per episode. Maze episodes are short (6–8 replans) and their single-transition
prediction error is noisy, so the brake is a much coarser instrument there — it is
firing often on `maze_diverse`, where v2 wins by the largest margin, and on
`maze_medium_low_density`, where it costs success. The brake's estimator is only as
stable as single-transition prediction error, and that stability varies by domain.

## Open ablations

Neither is required to read the table above, and both are cheap:

1. **The published implementation's objective** — 1-step-ahead, teacher-forced over a
   `min(num_hist, T)` window, on the merged episode. It sits between `horizon: 1` and
   this method, so without it the reported gain cannot be split into "restored the
   context v1 dropped" and "fitted the open-loop rollout".
2. **The brake alone**, at `horizon: 1`. The selection sweeps score that cell (0.670 on
   `pushobj`, 0.407 on `pusht`, 0.800 on `maze_medium`), but only on selection cohorts.

A third is now motivated by the maze results rather than by symmetry: **a brake that
needs more than one sample to fire** — a consecutive-strikes rule or a running ratio —
since the single-transition estimator is demonstrably noisy where episodes are short.

## Selection

Two axes, 9 initial cells — the same selection cost as v1:

| axis | grid | expansion pool |
|---|---|---|
| `pred_lr` | 5e-4, 2e-3, 1e-2 | log |
| `horizon` | 1, 3, 5 | 1, 2, 3, 5, 8 |

Cost was 12 columns on each push setting (one expansion) and 14 on `maze_medium` (two).
`method.yaml` ships the push choice, `pred_lr: 2e-3`; `maze_medium` froze `1e-2` and the
three maze shift settings inherit that, exactly as v1's do.

`steps` is **authored at 5, not tuned**: v1's own selection chose 5 on all three
settings it was tuned on, and spending one of two axes to rediscover that would leave
the mechanism this method is about untested. `buffer_size: 8` and `brake_ratio: 1.25`
are likewise authored — 8 transitions is the shortest buffer that supports a 5-step
window, and 1.25 is a deliberately loose brake, meant to catch a correction that is
clearly worse than frozen rather than to arbitrate small differences. Neither was
chosen by looking at benchmark episodes.

## Alignment note

`on_transition` locates the executed chunk's frame boundaries by counting **backwards**
from the end of `rollout_obs`. The protocol documents that argument as
`1 + T * frameskip` frames, but the evaluator replays the episode's whole action
sequence from its initial condition on every replan, so what arrives is the entire
trajectory so far and only its tail belongs to the chunk just executed. Counting from
the front pairs the current state with the endpoint of the episode's *first* action
from replan 2 onward. Counting from the end is correct under either reading.

## Cost and determinism

Episode-isolated (`requires_episode_isolation: true`): it owns predictor parameters
and an AdamW trajectory, so a batched cohort would average unrelated episodes into one
correction.

The only stochastic element is the LoRA `A` initialization, drawn from a CPU generator
seeded by the declared `lora_init_seed: 0`; `B` starts at zero, so the correction
starts at exactly zero and the first replan of every episode plans with the frozen
model.

## Reported per replan

| key | meaning |
|---|---|
| `adajepa_v2/loss` | rollout loss at the last gradient step of this replan |
| `adajepa_v2/rollout_depth` | depths actually rolled (`min(horizon, buffered transitions)`) |
| `adajepa_v2/fresh_loss`, `adajepa_v2/fresh_loss_frozen` | single-step loss on the newest transition, adapted vs frozen |
| `adajepa_v2/fresh_loss_ratio` | their ratio; > 1 means adaptation is hurting out-of-sample |
| `adajepa_v2/braked`, `adajepa_v2/brakes` | did the brake fire this replan, and how often this episode |
| `adajepa_v2/buffer_size` | buffered transitions |
