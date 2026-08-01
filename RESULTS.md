# Results

Reproduction log. One section per milestone acceptance, newest last. Numbers here are
produced by this repo; targets are the predecessor's, pinned at `628dff7` (`PROVENANCE.md`).

---

## M0 — scaffold

**Accept:** a frozen PushObj column runs end to end and reports success 0.490 ± noise on
seed 300.

**Status: accepted, exactly.** 2026-08-01.

```
.venv/bin/python scripts/eval_column.py \
    --setting pushobj --cohort selection --tag frozen --gpus 0,1,2,3 \
    --tmpfs-root /dev/shm/tw78/data
```

| shape | PAARBench | predecessor (`eval_outputs/indist_matrix/frozen/seed300`) |
|---|---|---|
| T | 0.500 | 0.50 |
| L | 0.440 | 0.44 |
| Z | 0.600 | 0.60 |
| + | 0.420 | 0.42 |
| **mean** | **0.4900** | **0.490** |

n=200 (4 shapes × 50), seed 300, 20 MPC replans. Agreement is per-shape and exact, not
just on the mean — which is the stronger check, since a mean can match while two shapes
move in opposite directions. Exactness is expected rather than lucky: the planner is
deterministic (§6), and that determinism is what licenses paired comparison later.

Environment differences from the predecessor that this run shows are immaterial: a fresh
`uv` venv (`torch 2.3.0+cu121`, same pins), `conf/eval.yaml` in place of
`conf/plan_gd_mpc_local.yaml` (byte-identical content), and the training dataset read from
tmpfs via `dataset_data_path` rather than the SMB mount (same bytes; verified by a
path-and-size manifest comparison over all 16,051 files).

### What M0 covers

- `uv` project, Python 3.9, cu121 torch pinned explicitly per §9. `torch.cuda.is_available()`
  verified `True` on 8× A6000.
- §7 "take" list extracted from `628dff7` via `git archive`; one path fixed
  (`env/__init__.py`'s `LOCAL_DEPS_DIR` pointed into the predecessor). `PROVENANCE.md`
  records every file's origin and every deliberate omission.
- `paarbench/settings.py` — the §6 suite as data, with the selection/test cohort split
  declared by the benchmark. `pointmaze` is registered but **disabled** pending §2.3.
- `paarbench/staging.py` — tmpfs staging baked into the harness (§9) rather than left to
  per-config overrides. Adopts an already-correct stage instead of re-copying 2 GB.
- `scripts/eval_column.py` — the column driver: shapes fan out one per GPU, resumable,
  tees to disk.

### What M0 does not cover

The frozen arm exercises no adapter code, so this says nothing about the two methods being
portable. That is M1, and it is the first point at which a number could disagree.

---

## M1 — the adapter interface

**Accept:** frozen 0.485, HyperJEPA 0.610, tuned online 0.678 reproduced on PushObj test
seeds (n=600); base weights bit-identical across episode boundaries.

**Status: reproduced.** 2026-08-01.

| arm | PAARBench | target | per-cohort (100 / 200 / 400) |
|---|---|---|---|
| Frozen | **0.4850** | 0.485 | 0.550 / 0.460 / 0.445 |
| HyperJEPA (amortized, r2 distill0 ep2) | **0.6100** | 0.610 | 0.625 / 0.615 / 0.590 |
| AdaJEPA (online, steps=10, lr=5e-4, r2 predlast) | **0.6783** | 0.677–0.678 | 0.695 / 0.700 / 0.640 |

n=600 each (3 test cohorts × 4 shapes × 50). The frozen arm also matches the
predecessor's *per-cohort* numbers exactly — 0.550 / 0.460 / 0.445 — not just the pooled
figure. HyperJEPA's delta over frozen is +0.125, matching the predecessor's
+0.125 [+0.088, +0.162].

`planning/mpc.py` now contains no method-specific branch. The frozen selection column
re-ran through the rewritten loop at **0.4900, per-shape identical** to before the
refactor, so the control flow change is verified inert.

### Two things this exposed, neither of which was a harness bug

**1. Episode isolation was missing, and its absence failed silently.**

AdaJEPA owns mutable shared state — world-model parameters plus an AdamW trajectory. Run
against a batched cohort, one gradient step averages 50 unrelated episodes into a single
correction. That is a different method, and it does not announce itself:

| | pooled |
|---|---|
| AdaJEPA, batched cohort (wrong) | 0.5533 |
| AdaJEPA, episode-isolated (correct) | **0.6783** |

No error, no warning, a plausible number 0.125 below the truth. The harness now supports
`requires_episode_isolation: true` in `method.yaml`, which fans out one process per
episode; AdaJEPA declares it and HyperJEPA does not, which is exactly the distinction the
flag exists to draw. Cost is `n_evals`× the processes — ~7 min per column at
`--per-gpu 4` on 8 GPUs, against ~2.5 min batched.

The predecessor guarded this with `require_single_episode: true`, which *raises* on a
batched call. That flag was in its config and not in the first port's — which is how the
silent version happened. It is now set, so the guard and the harness agree.

**2. The wrong adapter epoch produced a plausible number.**

The first HyperJEPA port used epoch 4 and scored 0.588 rather than 0.610. Epoch 4 came
from misreading a three-column epoch table in the predecessor's results, where 0.610
appears in row 4 of the *distill1* column, not distill0's. On the selection cohort
distill0 scores 0.560 / **0.605** / 0.595 / 0.570 / 0.570 across epochs 1–5, so epoch 2 is
the correct pick — and epoch 2 gives 0.610 on test.

Nothing about 0.588 looked wrong. What would have caught it is running the declared
selection rule (`EpochSelection`, 5 columns) instead of hand-transcribing a number, which
is the argument for the protocol in a sentence.

### Still open

- **The reset guarantee is unit-tested, not enforced per method.** `BaseWeightGuard` is
  covered directly, but the cross-episode bit-identity check on real adapters needs a
  world model, and constructing one outside `plan.py`'s Hydra machinery needs a loader the
  harness does not expose. `tests/test_adapter_reset.py` carries it as an explicit skip.
  Note the reproduction above does not exercise it: every process still evaluates one
  cohort, so no adapter is reused across an episode boundary yet.
- **Both reproductions used `--skip-selection`**, so selection cost is recorded as
  `unknown`, not 0. Running the real rules (16 columns for AdaJEPA, 5 for HyperJEPA) is
  what a genuine submission does.

---

## M2 — output schema and metrics

**Status: schema and metrics landed; `safety_span.tex` reproduction still outstanding.**
2026-08-01.

Before this, the harness recorded only `mpc/mean_state_dist` — a mean, which is the one
summary §4 says never to use on distance, and which makes every discriminating metric
uncomputable. `planning/evaluator.py` now emits per-episode values,
`planning/mpc.py` writes `episodes.jsonl` (one row per episode per replan),
`paarbench/schema.py` reads it identically for batched and episode-isolated runs, and
`paarbench/metrics.py` computes the metrics.

All three arms were re-run under the new schema and reproduce **exactly**: frozen 0.4850,
HyperJEPA 0.6100, AdaJEPA 0.6783, per-cohort identical. The schema change is inert.

### The leaderboard is a frontier, not a ranking

| method | success | median dist Δ | catastrophe | compounding | adapt s/replan |
|---|---|---|---|---|---|
| AdaJEPA (online) | **0.678** | **+13** | **14.0%** | +0.9/replan | 0.568 |
| HyperJEPA (amortized) | 0.610 | **−6** | **8.3%** | +0.0/replan | **0.147** |
| Frozen | 0.485 | — | — | — | 0.000 |

**The ordering reverses depending on the metric.** AdaJEPA wins on success by +0.068 and
loses on everything else: its median paired distance is *worse than not adapting at all*
(+13, Wilcoxon p<0.05), it leaves an episode >2× further from the goal 14.0% of the time
against HyperJEPA's 8.3%, and it costs 3.9× the adaptation time per replan.

This is exactly why §4 refuses a single collapsed score. Ranked on success rate alone the
table says "online gradient TTA wins"; the same 600 episodes also say it more often makes
things much worse and pays more to do it.

The compounding slope separates the mechanism cleanly, as designed: **+0.9/replan** for
the optimizer trajectory that accumulates, **+0.0/replan** for the correction regenerated
from frozen weights each replan.

### One thing this shook out

The leaderboard initially rendered AdaJEPA's continuous columns from a *partially
complete* re-run while taking its success rate from the previous complete record — giving
median dist Δ −18 and catastrophe 4.5%, both flattering and both wrong (the true values
are +13 and 14.0%). `scripts/leaderboard.py` now checks that the episode count on disk
matches the result record's `n` and suppresses the continuous columns when they disagree.
Mixing a complete summary with partial detail is a quiet way to publish a wrong number.

### Follow-up pass (same day)

Bootstrap CIs, the remaining two §4 metrics, and four correctness fixes:

| method | success | median dist Δ | catastrophe | compounding | regret | adapt s | peak MB |
|---|---|---|---|---|---|---|---|
| AdaJEPA | **0.678** | +13 [+7, +22] | 14.0% [9%, 19%] | **+0.91** [+0.40, +1.68] | 62% / +196 | 0.568 | **177** |
| HyperJEPA | 0.610 | **−6** [−14, −1] | **8.3%** [5%, 12%] | **+0.01** [−0.16, +0.09] | **43% / +93** | **0.147** | 977 |

Both distance intervals exclude zero, so the reversal is real and not a rounding
artifact: AdaJEPA is *significantly worse than not adapting* on final distance while
being significantly better on success. The compounding intervals separate the mechanism
cleanly — accumulating (+0.91, excludes 0) against recomputed (+0.01, includes 0).
Peak memory is the one axis where the amortized method loses: its generator is 10.7 M
parameters, 977 MB against 177 MB, in exchange for 3.9× lower latency.

CIs are 95% percentile bootstrap over **episodes** (2000 resamples, fixed seed).
Episodes are the unit of independence; replans within one are a trajectory, so the
compounding slope is refit per resample from an episode × replan matrix rather than
bootstrapped over rows.

Fixed in the same pass:

- **`episodes.jsonl` was opened in append mode.** Hydra does not clear a reused run
  directory, so a re-run would have interleaved two runs' rows and the metrics would
  have paired episodes against duplicates of themselves. Truncated at episode start.
- **Resume keyed on `logs.json` existing**, but that file is written on the *first*
  replan. An interrupted unit looked finished and would be skipped forever. Now keyed
  on `final_eval/success_rate`, which is written once, at the end.
- **AdaJEPA's selection grid was mis-sized.** It topped out at `lr=5e-3`, an order of
  magnitude below where this method's correction becomes comparable to the weights it
  corrects (`lr × steps × replans ≈ 4e-2`). It would have found a plausible best cell
  while hiding that a bad hyperparameter reverses the adaptation signal — the single
  most important thing to know about the method. Re-derived from that scaling rather
  than copied.
- **The reset guarantee is now enforced per method** against a real world model, not
  just unit-tested. See below.

The acceptance criterion of reproducing the predecessor's exact distance spans is
**dropped** by owner decision: exact replication is no longer a goal now that the port
is validated.

### The reset test found a contract error — in the test

`paarbench/world_model.py` factors model loading out of `plan.py` (which now imports it,
so there is one implementation), which finally let the cross-episode bit-identity check
run for real. Its first version perturbed *every* weight and demanded the adapter restore
them. AdaJEPA passed; HyperJEPA failed on 173 tensors.

That was the test being wrong. HyperJEPA declares **zero** trainable base parameters — it
freezes the model and keeps its correction in LoRA wrapper slots, which `clear()` does
reset. Demanding it repair damage it cannot see would have forced every method to carry a
400 MB full-model snapshot for nothing.

The contract is now stated precisely — *undo everything you can reach* — and the test
perturbs exactly that surface: parameters the adapter marked trainable, plus the
correction slots it installed. Both methods pass, along with an idempotence check. 5
integration tests, opt-in with `-m integration` so the default suite stays GPU-free.

### Note for M0

`AdaJEPAAdapter` has **no episode-reset method at all** — it is constructed once per
planning process and each process evaluates one batch, so the need never arose. The reset
that §5 requires is therefore new work for that method, not a port. See
`docs/ADAPTER_PROTOCOL.md`, which records the full call-site analysis behind the protocol.
