# HOVER — horizon-matched online TTA with a fresh-evidence brake

HOVER adapts the same thing AdaJEPA adapts — a rank-2 LoRA plus a LayerNorm delta on
the predictor's last block, fitted online with AdamW on the world model's own
prediction error — and changes three things about *how*. It carries no offline
training and no extra checkpoint; everything it needs is the base model and the
episode it is in.

## Why these three changes

**The objective is the rollout the planner solves through, not one step of it.**
The planner rolls the predictor forward `goal_H / frameskip` model steps open loop
(5 on the Push settings), feeding each prediction back in as context, and scores the
end of that rollout against the goal. AdaJEPA fits single-step prediction, so a
correction that is accurate for one step and drifts over five looks perfect to it and
plans badly. HOVER rolls the correction forward `horizon` steps under the actions that
were really executed, against the states the environment really reached, and
accumulates loss at every depth — mirroring `VWorldModel.rollout`, including its
growing `num_hist` context window, so the thing being fitted is the thing being used.

At `horizon: 1` this reduces to a single-step objective. That value is in the
selection grid deliberately: if the multi-step objective does not earn its place, the
protocol picks the arm that does not use it.

**The update is braked by evidence it has not seen.** AdaJEPA's correction is an
optimizer trajectory that only accumulates: on `pushobj` it ends more than 2× further
from the goal than frozen on 17.1% of episodes, and its paired distance gap grows with
replan index (compounding +0.24). Nothing inside the method can notice, because an
episode's outcome is not observable and its buffer only ever holds data it has already
fitted.

The transition that arrives at replan *n* is data the update at replan *n−1* was
fitted before. HOVER scores the current correction against the frozen predictor on
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
`hover/braked` and
`hover/fresh_loss_ratio` are recorded per replan, so how often this fires — and
whether adaptation is helping out-of-sample at all — is visible in the record rather
than inferred.

**Target latents are cached, so a gradient step is predictor-only.** Only predictor
LoRA parameters are trained, so every encoding in the buffer is a constant for the
rest of the episode. AdaJEPA re-encodes its whole buffer inside every gradient step
(`steps × buffer_size` encoder forwards per replan); HOVER encodes each transition
once when it arrives — two forwards per replan, independent of `steps` — which is what
makes a multi-step objective over an 8-transition buffer cheaper than AdaJEPA's
single-step one over a 5-transition buffer.

## What is held fixed on purpose

The parameterization (`predlast_all`, rank 2, `lora_scale` 1.0), AdamW, and
`grad_clip_norm: 1.0` are AdaJEPA's, unchanged. The comparison between the two rows is
meant to be about the objective and the brake, not about capacity or optimizer
tuning.

## Results

Selection froze `pred_lr: 2e-3, horizon: 5` on **both** selectable settings, chosen
independently, after 12 columns each (`horizon: 5` sat on the initial grid's boundary,
so that axis expanded once; `horizon: 8` then scored below it). `horizon: 5` beat
`horizon: 1` at the selected learning rate on both — 0.685 vs 0.670 on `pushobj`,
0.427 vs 0.407 on `pusht` — so the multi-step objective is what the protocol picked,
not what it tolerated.

| setting | HOVER success | AdaJEPA | median dist Δ | compounding | catastrophe | regret | adapt s/replan |
|---|---|---|---|---|---|---|---|
| `pushobj` | **0.697** | 0.697 | +8 vs +6 | +0.22 vs +0.24 | 14.4% vs 17.1% | 56%/+156 vs 57%/+256 | 0.239 vs 0.294 |
| `pushobj_shift` | **0.409** | 0.400 | +1 vs +16 | +0.27 vs +0.97 | 15.9% vs 16.9% | 51%/+204 vs 59%/+205 | 0.239 vs 0.290 |
| `pusht` | **0.478** | 0.440 | +16 vs +55 | +1.23 vs +3.87 | 22.2% vs 26.8% | 55%/+274 vs 66%/+549 | 0.238 vs 0.293 |

Read the success column carefully: +0.038 on `pusht` is ~1.6 SE and the other two are
inside one SE, so on success alone HOVER and AdaJEPA are not separable at these n. What
*is* separable is the risk side, and it moves in the same direction on all three
settings. On `pushobj_shift` and `pusht` AdaJEPA's median paired distance change and
compounding slope both have intervals excluding zero — it ends measurably further from
the goal than frozen on shared failures, and gets worse per replan — while HOVER's
compounding interval straddles zero on `pushobj_shift` and its median distance change
straddles zero on all three. The ordering across settings tracks how much room the
correction has to go wrong: a tie where AdaJEPA barely drifts (`pushobj`), the largest
gain where it drifts hardest (`pusht`).

HOVER does not match HyperJEPA's risk profile on `pusht` (compounding −0.45,
catastrophe 10.7%) despite scoring above it on success. A correction that is recomputed
from frozen weights every replan compounds less than one that is braked, and the brake
does not close that gap.

Diagnostics from the `pushobj` selection sweep, which the record does not carry:
fitting lowers *out-of-sample* single-step prediction error to a median 0.77× frozen at
`horizon: 1`, rising to 0.92× at `horizon: 5` — the multi-step objective spends
single-step accuracy on rollout consistency, as intended. The brake fires on 7–12% of
episodes, 0.1–0.2 times per episode: loose enough to leave adaptation alone, which is
why the median trajectory is unchanged while the tails are not.

## Selection

Two axes, 9 initial cells — the same selection cost as AdaJEPA:

| axis | grid | expansion pool |
|---|---|---|
| `pred_lr` | 5e-4, 2e-3, 1e-2 | log |
| `horizon` | 1, 3, 5 | 1, 2, 3, 5, 8 |

`steps` is **authored at 5, not tuned**: AdaJEPA's own selection chose 5 on all three
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
| `hover/loss` | rollout loss at the last gradient step of this replan |
| `hover/rollout_depth` | depths actually rolled (`min(horizon, buffered transitions)`) |
| `hover/fresh_loss`, `hover/fresh_loss_frozen` | single-step loss on the newest transition, adapted vs frozen |
| `hover/fresh_loss_ratio` | their ratio; > 1 means adaptation is hurting out-of-sample |
| `hover/braked`, `hover/brakes` | did the brake fire this replan, and how often this episode |
| `hover/buffer_size` | buffered transitions |
