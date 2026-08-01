# The adapter protocol — M1 design notes

The protocol itself is `paarbench/adapter.py`. This document records the call-site
analysis it was derived from, so that the port can be checked against something other
than intuition. Everything here is about the two methods that already exist; a protocol
that cannot express them exactly is wrong, because M1 is accepted only when their
published numbers reproduce.

Source: `~/HyperJEPA/` at commit `628dff7` — `planning/mpc.py`, `planning/adaptation.py`
(`AdaJEPAAdapter`, online gradient TTA), `planning/hyper_adapter.py` (`HyperJEPAAdapter`,
amortized hypernetwork). Configs for both are preserved in `docs/reference/`.

## What the planner does today

`MPCPlanner.plan` holds one hardcoded branch per method (PLAN §2.1). Per replan `i`:

| order | online (`self.adapter`) | amortized (`self.hyper_adapter`) |
|---|---|---|
| before the loop | — | `maybe_apply_episode_start(context_obs_0 or cur_obs_0)` |
| 1. solve | `sub_planner.plan(cur_obs_0, obs_g, memo_actions, i)` | same |
| 2. execute + score | `evaluator.eval_actions(...)` → `e_obses`, `e_states` | same |
| 3. observe | `append(cur_obs_0, taken_actions, e_final_obs)` | `append_executed_transitions(cur_obs_0, taken_actions, e_obses, frameskip)` |
| 4. update | `step()`, gated by `i % refresh_interval == 0` **in the planner** | — |
| 5. advance `cur_obs_0` | — | — |
| 6. refresh | — | `maybe_refresh_after_mpc(cur_obs_0)`, gated **inside the method** |

Five asymmetries, each of which the protocol has to resolve rather than paper over.

### 1. The online method sees only the chunk's final observation

Step 3 hands it `slice_trajdict_with_t(e_obses, start_idx=-1)` — one frame — and it
builds a single transition `(cur_obs_0, taken_actions[:, :1], final_obs)`. Note it
silently truncates to the *first* action (`action[:, :1]` in `_prepare_transition`)
while pairing it with the *last* observation. That is the predecessor's semantics and
the port must keep it, mismatch and all, or the online numbers move.

The amortized method gets the full rollout plus `frameskip` and builds one feature per
executed action at the correct frame boundaries.

**Resolution:** `on_transition` passes the full rollout and `frameskip`. The superset is
the only signature that expresses both. Methods wanting the endpoint slice it themselves.

### 2. Refresh gating lives in two different places

The planner gates the online method's `step()`; the amortized method gates itself in
`maybe_refresh_after_mpc`. Planner-side gating *is* the branch M1 removes.

**Resolution:** all scheduling moves inside the method, into `before_plan`.

### 3. `step()` runs at step 4, `before_plan` would run at step 1 of the next replan

Safe to move: nothing between step 4 and the next solve touches the world model
(`cur_obs_0` is reassigned, the evaluator's initial condition is set). Confirm during the
port by diffing the per-replan loss logs, not by argument.

Consequence: `before_plan` also fires before the *first* solve, where no transition has
been seen. The online method's `step()` already returns early on an empty buffer, so this
is a no-op — but any new method must tolerate it.

### 4. Episode start is a different event for each method

`maybe_apply_episode_start` applies a correction immediately **unless**
`context_mode == "transition_buffer"`, in which case it deliberately leaves the frozen
model alone until an action has produced feedback. Separately, `refresh` is either
`episode_start` or `every_mpc`, and `maybe_refresh_after_mpc` returns `{}` unless it is
`every_mpc`.

So the amortized method has two independent switches — *what conditions the correction*
and *how often it is recomputed* — and the planner sees the product of them. The port must
keep both as method-internal configuration, expressed as: `on_episode_start` may apply a
correction; `before_plan` decides each replan whether to refresh.

**This is the part of the port most likely to go wrong.** Reproduce HyperJEPA 0.610 before
touching anything else.

### 5. Only one method can restore the base model

`HyperJEPAAdapter.clear()` snapshots and restores base weights, precisely so a later
episode does not inherit a folded correction. `AdaJEPAAdapter` **has no reset method at
all** — it is constructed once per planning process, and each process evaluates exactly
one batch, so the question never arose.

That is a latent correctness bug the moment anything reuses an adapter across episodes,
and it is why PLAN §5 asks for `on_episode_start` to restore the base model and for a test
that asserts bit-identity. Building it for the online method is new work, not a port:
snapshot the selected parameter tensors at construction, restore them and rebuild the
optimizer (dropping its moment estimates) on reset.

## Why the metric depends on getting this right

The compounding-slope metric (§4) exists to separate *accumulated* from *recomputed*
updates: the predecessor measured degradation rising monotonically to +1156 over 20
replans for the accumulated update, and flat at ≈0 when the correction is recomputed from
frozen weights each replan. That contrast is exactly the distinction between step 4 and
step 6 above. A port that quietly makes one method behave like the other destroys the
benchmark's mechanism metric while still looking plausible.

## Timing and memory decomposition

Adaptation cost must be separable from planner cost (§4: latency; §5). Time
`on_transition` and `before_plan` separately from `sub_planner.plan`, and sample
`torch.cuda.max_memory_allocated` around the same hooks. The predecessor already logs
`timing/ada_update_s` and `timing/hyper_update_s` next to `timing/planner_s`; keep the
decomposition, drop the per-method names.

## Acceptance

1. Frozen **0.485**, HyperJEPA **0.610**, tuned online **0.678** on PushObj test seeds
   (n=600 — cohorts 100/200/400).
2. Base weights bit-identical across an episode boundary, for every adapter.
3. `planning/mpc.py` contains no method-specific branch.

Anything else means the port changed semantics.
