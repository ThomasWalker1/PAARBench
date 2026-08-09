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
