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

---

## Second setting, and the unconditioned control

2026-08-01. `pusht` runs for the first time, and `static_lora` joins as a fourth arm.

### pushobj (n=600)

| method | success | median dist Δ | catastrophe | compounding | regret | adapt s | peak MB |
|---|---|---|---|---|---|---|---|
| AdaJEPA | **0.678** | +13 [+7, +22] | 14.0% [9%, 19%] | +0.91 [+0.40, +1.68] | 62% / +196 | 0.568 | 177 |
| HyperJEPA | 0.610 | **−6** [−14, −1] | 8.3% [5%, 12%] | +0.01 [−0.16, +0.09] | 43% / +93 | 0.147 | 977 |
| **Static LoRA** | 0.567 | −5 [−9, +1] | **6.2%** [3%, 10%] | −0.08 [−0.24, +0.04] | 46% / **+73** | **0.000** | **157** |
| Frozen | 0.485 | — | — | — | — | 0.000 | 157 |

**The unconditioned control is the story here.** Static LoRA buys +0.082 success over
frozen for *zero* per-replan adaptation cost and no extra memory, and it has the **lowest
catastrophe rate on the board** — 6.2%, against HyperJEPA's 8.3% and AdaJEPA's 14.0%. Its
distance delta, −5 [−9, +1], is statistically indistinguishable from HyperJEPA's
−6 [−14, −1].

So on this setting HyperJEPA's *conditioning* buys +0.043 success over a constant
correction, and nothing measurable on distance, catastrophe or compounding — in exchange
for 0.147 s and 977 MB per replan against 0.000 s and 157 MB. That is a much harder
question for the amortized method than "does it beat frozen", and it only becomes
visible once the control is on the board. It reproduces the predecessor's 0.567 for this
arm exactly.

### pusht (n=300) — the ordering flips

> **Superseded 2026-08-02.** Every PushT arm in this section was evaluated with
> `--skip-selection`. Running the declared rules moved three of them: HyperJEPA
> 0.430 -> 0.440 (its checkpoint was undeclared), AdaJEPA 0.397 -> 0.420 (its step size
> is not what its own grid picks), and Static LoRA's per-cohort numbers. The qualitative
> reading below survives and in fact sharpens; the numbers are stale. See *Running the
> skipped selection rules found two wrong checkpoints* and *What the corrected board
> says*.

| method | success | median dist Δ | catastrophe | compounding | adapt s |
|---|---|---|---|---|---|
| HyperJEPA | **0.430** | −7 [−22, +6] | 10.0% [6%, 15%] | −0.05 [−0.44, +0.26] | 0.137 |
| AdaJEPA | 0.397 | +3 [−9, +16] | 13.9% [9%, 19%] | **+0.65** [+0.07, +1.16] | 0.558 |
| Frozen | 0.350 | — | — | — | 0.000 |

On PushObj the online learner wins on success; here the amortized one does. Neither
distance interval excludes zero at this n, so the two are not separated on the
discriminating metric either way — which is the honest reading, and exactly why the
leaderboard reports intervals rather than an ordering.

What *does* carry across both settings is the compounding signature: AdaJEPA accumulates
(+0.65 [+0.07, +1.16] here, +0.91 [+0.40, +1.68] on PushObj — both exclude zero) while
the two recomputed-correction methods sit flat. The mechanism metric is measuring a
property of the method, not of the domain.

### What running a second setting exposed

- **`pusht`'s base ships `data_path: <path>`** — the literal placeholder, never
  substituted. Staging now detects placeholder values and falls back to a `dataset_path`
  declared on the setting, so a caller never has to know which bases are affected.
- **A method carrying trained weights needs different ones per setting.** `method.yaml`
  had a single `params` block, so HyperJEPA on `pusht` would have loaded the PushObj
  hypernetwork. Added `params_by_setting`, validated against the declared settings list.
  Duplicating the method per domain would have split one method across two leaderboard
  rows.

### Adding a third method sharpened the reset tests

Three problems surfaced, in the tests rather than the methods:

1. **The world-model fixture was module-scoped.** Constructing an adapter *mutates the
   model's structure* — it installs LoRA wrappers on the predictor — so the second
   method under test received a predictor the first had already wrapped and emptied.
   That reported a bug in `static_lora` that did not exist. Now one model per test.
2. **A weight-based probe cannot see a wrapper-held correction.** HyperJEPA and
   static_lora keep their correction as plain tensors inside the LoRA modules, which
   never enter `state_dict` — so the bit-identity check passes *vacuously* for them.
   Added `test_installed_corrections_are_cleared_on_reset` for that path.
3. **That new test then flagged AdaJEPA wrongly.** Its LoRA factors are registered
   `nn.Parameter`s — its parameterization, not a transient correction — and clearing
   them would delete what its optimizer points at. The check now distinguishes plain
   tensors (must be cleared) from parameters (restored by value, already covered).

Current reset coverage, honestly: **adajepa** by bit-identity, **static_lora** by
correction-clearing, **hyperjepa** by idempotence only — its correction is generated
from a real observation, which these tests do not supply. Closing that needs a driven
episode with real observation tensors.

### Static LoRA on pusht — and whether conditioning earns its keep

An earlier note here claimed PushT's static checkpoints were stored only as baked
full-model weights that the adapter could not read. That was wrong:
`pvs_adapters/static_r2/` carries `hyper_lora_epoch_*.pth` in exactly the same factored
format as the PushObj ones, alongside the `baked_ep*` directories. Supporting it was a
config change, not a loader.

| setting | frozen | static LoRA | HyperJEPA | static's share of the gain |
|---|---|---|---|---|
| pushobj | 0.485 | 0.567 (**+0.082**) | 0.610 (+0.125) | **66%** |
| pusht | 0.350 | 0.363 (+0.013) | 0.430 (+0.080) | **16%** |

**Whether conditioning earns its keep is domain-dependent.** On PushObj a single constant
correction captures two-thirds of what the hypernetwork achieves, so most of that method's
benefit is not conditioning at all. On PushT the same control captures almost nothing —
+0.013 against an SE of 0.028, indistinguishable from doing nothing — and the
hypernetwork's +0.080 is genuinely episode-dependent adaptation.

A single-setting evaluation would have supported either conclusion. That is an argument
for the control arm and for the second setting together; neither shows this alone.

One thing is consistent across both: **static LoRA has the lowest catastrophe rate
wherever it runs** (6.2% on PushObj, 6.6% on PushT, against 8.3–14.0% for the adaptive
methods) at zero adaptation cost. A correction that cannot react also cannot overreact.

Both `EpochSelection` rules are now setting-aware, keyed on `harness.setting_id` — the
only thing a selection rule is told about where it is running, which is enough to pick
the right weights and not enough to reach a test cohort.

### Note for M0

`AdaJEPAAdapter` has **no episode-reset method at all** — it is constructed once per
planning process and each process evaluates one batch, so the need never arose. The reset
that §5 requires is therefore new work for that method, not a port. See
`docs/ADAPTER_PROTOCOL.md`, which records the full call-site analysis behind the protocol.

---

## Selection can optimize paired safety metrics

2026-08-01. `SelectionHarness.run()` now returns the candidate's conventional success
score *and* three metrics paired to a harness-owned frozen column on the same selection
cohort: median final-distance delta, catastrophe rate, and compounding slope. The rule
still has no setting object, runner, output path, or test-cohort API; only the harness
can launch/cache the frozen selection reference, and it is not charged as a candidate
selection column.

This changes Restore TTA's declared choice on PushObj. The original success-only rule
selected `restore_probability=0.001` (selection success 0.685). Selecting the smallest
paired compounding slope over the identical four already-declared columns instead picks
**`p=0.01`**:

| p | selection success | median dist Δ | catastrophe | compounding slope |
|---|---:|---:|---:|---:|
| 0.001 | 0.685 | +2.44 | 11.9% | +0.086 |
| **0.01** | 0.670 | −5.82 | 7.8% | **−0.316** |
| 0.05 | 0.560 | −1.70 | 2.3% | −0.077 |
| 0.1 | 0.535 | −0.46 | 1.1% | −0.051 |

So the selection target, not just the method, materially changes the chosen amount of
restoration. At the time this was written the committed held-out Restore TTA result was
still the prior success-selected submission and this was reported as a selection-cohort
finding only. It has since been carried through to held-out data — see *The
slope-selected submission on held-out data*, below.

## PAD — inverse-dynamics adaptation during deployment

2026-08-02. PAD is the first adapter in the harness that owns a trainable module in
addition to adapting part of the world model: an inverse-dynamics head predicts the MPC
action chunk from consecutive encoder latents, and its loss updates the encoder online.
The reset/weight-guard interface was extended so such owned modules are part of the
adapter's reachable, restored surface; this is an interface correction, not a PAD-only
exception.

To give the method a meaningful initialization, the head was pretrained offline on
PushObj transitions from `pushobj_multishape` before deployment (2,048 pairs, 12 epochs,
MSE 0.249628). It receives no selection or test data. The resulting checkpoint and exact
recipe are recorded in `docs/CHECKPOINTS.md`. This is not a bit-identical reproduction of
Hansen et al.: their auxiliary head is pretrained jointly with a policy, while PAARBench
uses its MPC planner and a locally pretrained head.

On PushObj (n=600), PAD reaches 0.553 success (+0.068 over frozen, ±0.020 SE) with 0.029
s/replan adaptation cost. Its paired distance result is +4 [-2, +9], catastrophe is 5.8%
[3%, 9%], and compounding is +0.21 [-0.08, +0.51]. Thus the success gain is real enough to
report, but there is no interval-separated evidence that this inverse-dynamics update
improves the benchmark's primary paired distance mechanism. The low point estimate for
catastrophe is promising, but its interval overlaps the static control. PAD is therefore
a fair, documented baseline here rather than evidence for a broad deployment-adaptation
claim; it is evaluated only on PushObj because that is the domain for which the offline
head was pretrained.

---

## The failed p=0.01 columns were a method fault, not an interrupted launcher

2026-08-02. The first attempt at the slope-selected held-out columns left
`eval_outputs/restore_tta/pushobj/test100` with all 200 units at rc=1 after 172.9 s
against a healthy column's ~432 s, and a sampled episode log that showed planning
running to completion. The working hypothesis was an interrupted launcher. It was not.

Sorting the 200 unit logs by content and by mtime separates two runs cleanly:

| | 19:12:12 → 19:15:04 | 19:16:05 → 19:23:09 |
|---|---:|---:|
| died with a traceback | 66 | 0 |
| planned to completion | 3 | 115 |
| truncated mid-run | 0 | 16 |

Every traceback is the same one:

```
File "methods/restore_tta/adapter.py", line 419, in _stochastic_restore
    reference = self._guard._snapshot[name].to(
AttributeError: 'BaseWeightGuard' object has no attribute '_snapshot'
```

`restore_tta` reached into `BaseWeightGuard._snapshot`, a private dict. Extending the
guard to cover PAD's owned trainable module turned it into `_snapshots`, a list of
dicts, one per guarded module. Nothing referenced the old name except this one line, in
a different method, so nothing failed until the next Restore TTA column ran — and then
every unit of it failed, because `_stochastic_restore` is called after every optimizer
step regardless of `p`. **The p=0.001 rows are unaffected only because they were run
before the rename**, not because the low probability kept the branch cold.

The rest of the wreckage follows from that. A crashing unit costs ~25 s rather than
~65 s, which is why the column finished in 172.9 s — a column failing everything is
*faster* than a healthy one, so wall clock read as good news. The second band is a
relaunch after a `_snapshot = self._snapshots[0]` alias was added, killed by hand
partway through; the 115 clean logs and the "looks fine" sampled episode belong to
that repaired run, which is why sampling a log did not show the fault.

Confirmed the current tree is sound by running one p=0.01 episode end to end: rc=0,
20 replans, `total_planning_main_s=49.9`.

### What was actually wrong, beyond the one line

The `_snapshot` alias made the symptom go away and left the coupling in place, so the
fix here is a public `BaseWeightGuard.reference(name)`. Stochastic restoration needs
the pretrained value of an individual weight, which is a legitimate thing for a method
to want and was simply missing from the interface; a method that has to reach into a
private attribute to do its job will break again the next time the harness changes.

Three harness problems the failure exposed, all fixed:

1. **Resume could splice two configurations into one column.** `run_column` skipped any
   unit with a completion marker, and that marker records that a unit *finished*, not
   *what it ran*. Pointing p=0.01 at the p=0.001 column — which is exactly what the
   default `--out-root` does — would have kept the finished p=0.001 units and reported
   the mean of two configurations as one. It survived only because the earlier data had
   been moved aside by hand first. `check_resumable` now refuses on a parameter
   mismatch and names the offending key.
2. **A failing column was not loud.** The driver printed `[done] N/M` and nothing else;
   the 197 failures were visible only inside a JSON file. Failure counts now ride the
   progress line, and a column with failures prints a `[FAIL]` summary with examples.
3. **`rc` could be unbound.** If the launch itself raised, the `if rc != 0` check raised
   `UnboundLocalError`, killing that worker thread and leaving the rest of the queue
   unrun while the column still wrote a summary.

`evaluate.py --tag` now exists so a variant can be evaluated without pointing at a
committed submission's evidence at all. Its selection sweep stays keyed to the method,
so a variant that differs only in which objective is read off the same columns reuses
them rather than paying twice.

Verified before re-running: all 600 pushobj and 300 pusht `.hydra` overrides under the
p=0.001 columns record `restore_probability=0.001`, and every `logs.json` is stamped
16:44–17:05, before the 19:12 incident. That data is intact and is retained as an
ablation.

---

## The slope-selected submission on held-out data

2026-08-02. The declared Restore TTA rule now optimizes the paired compounding slope, so
the submission is whatever that rule freezes. Both settings were re-run end to end. The
four selection columns were cached, so this is test compute; the p=0.001 evidence is
retained untouched as an explicitly labelled ablation (`restore_tta_success_selected`),
and the leaderboard renders it in a separate *Ablations (not submissions)* table rather
than as a second entry.

### PushObj (n=600, frozen 0.485)

| | success | median dist Δ | catastrophe | compounding | regret |
|---|---:|---:|---:|---:|---:|
| success-selected, p=0.001 *(ablation)* | 0.687 | +7.5 [−0.2, +15.1] | 13.6% [9%, 19%] | +0.58 [+0.11, +1.25] | 57% / +260 |
| **slope-selected, p=0.01** *(submission)* | 0.678 | +5.1 [−0.2, +8.8] | **7.0%** [4%, 11%] | +0.32 [+0.01, +0.67] | 57% / **+138** |

### PushT (n=300, frozen 0.350)

| | success | median dist Δ | catastrophe | compounding | regret |
|---|---:|---:|---:|---:|---:|
| success-selected, p=0.001 *(ablation)* | 0.377 | +10.9 [+0.7, +27.3] | 12.3% [8%, 18%] | +0.58 [−0.02, +1.12] | 57% / +186 |
| **slope-selected, p=0.1** *(submission)* | 0.347 | **−2.6** [−4.7, −0.4] | **3.2%** [1%, 6%] | −0.06 [−0.20, +0.05] | 41% / **+51** |

**The finding replicates on held-out data, and the two settings say different things
about what it costs.** On PushObj the trade is mild: 0.9 pp of success for roughly half
the catastrophe rate and half the tail regret. On PushT it is total. The slope-selected
arm is the only entry anywhere on this benchmark whose paired distance interval excludes
zero *on the good side* — −2.6 [−4.7, −0.4], against every other arm's overlapping or
positive interval — and it has the lowest catastrophe rate on the board at 3.2%. It pays
for that with the entire success gain: 0.347 against frozen's 0.350, which is to say the
method no longer beats doing nothing on the conventional metric at all.

So a single method, one declared rule apart, occupies two ends of the frontier. That is
the benchmark's thesis stated as a measurement rather than as an argument, and it is the
reason §4 refuses a collapsed score: no scalar ranking can hold both of these rows.

### The selection cohort over-promises, on a safety objective too

`RESULTS.md`'s earlier selection-cohort table is not what held-out data shows.

| setting | selected p | slope on selection cohort | slope on held-out cohorts |
|---|---|---:|---:|
| pushobj | 0.01 | **−0.316** | **+0.323** |
| pusht | 0.1 | +0.059 | −0.061 |

On PushObj the objective's own value changes sign between selection and test. PLAN.md §7
trap 4 — "grid maxima are optimistic; never quote a selection-cohort maximum as a score"
— was recorded for success rate; it applies just as forcefully to a paired safety metric,
and this is the first measurement of that in the repo. What *does* transfer is the
**ordering**: on both settings the slope-selected p beats the success-selected p on every
safety metric held out, even though the magnitude does not survive.

That is the right way to read a selection rule, and it is why the protocol re-runs the
selected configuration instead of reporting the cell.

### PushT's selection cohort is where the objectives disagree most

| p | selection success | median dist Δ | catastrophe | compounding slope |
|---|---:|---:|---:|---:|
| 0.001 | 0.427 | +20.58 | 24.1% | +1.276 |
| 0.01 | 0.420 | +0.49 | 12.2% | +0.229 |
| 0.05 | 0.380 | +8.38 | 11.5% | +0.490 |
| **0.1** | 0.413 | +0.75 | **7.0%** | **+0.059** |

Success spans 0.047 across these four columns — inside one SE — while the compounding
slope spans 22× and catastrophe spans 3.4×. A success-only rule choosing between p=0.001
and p=0.1 here is choosing on noise, and it picked the cell with 24% catastrophe.

---

## Running the skipped selection rules found two wrong checkpoints

2026-08-02. Three of six leaderboard rows carried `unknown` in the selection-cost
column because they had been evaluated with `--skip-selection`. That column exists to
make tuning burden comparable, and it could not, for half the board. All three rules
were run for real.

### The two `EpochSelection` rules

| method | setting | epochs on the selection cohort | declared in `method.yaml` | rule picks |
|---|---|---|---|---|
| hyperjepa | pushobj | 0.560 / **0.605** / 0.595 / 0.570 / 0.570 | epoch 2 | epoch 2 ✓ |
| hyperjepa | pusht | 0.413 / 0.440 / **0.460** / 0.447 | `hyper_lora_best.pth` | **epoch 3** |
| static_lora | pushobj | 0.520 / **0.530** / 0.525 / 0.520 / 0.480 | epoch 2 | epoch 2 ✓ |
| static_lora | pusht | **0.380** / 0.347 / 0.367 / 0.340 / 0.347 | epoch 2 | **epoch 1** |

**Both PushObj declarations were right and both PushT declarations were wrong.** The two
PushObj rows re-ran to *bit-identical* per-cohort numbers (hyperjepa 0.625/0.615/0.590 →
0.6100; static_lora 0.605/0.535/0.560 → 0.5667), which is the expected behaviour of a
deterministic planner and confirms the re-run itself is inert.

The two PushT rows moved:

| | before (undeclared choice) | after (declared rule) |
|---|---|---|
| hyperjepa pusht | 0.4300 — `hyper_lora_best.pth`, cost unknown | **0.4400** — epoch 3, cost 4 |
| static_lora pusht | 0.3633 — epoch 2, cost unknown | **0.3633** — epoch 1, cost 5 |

`hyper_lora_best.pth` is a checkpoint the declared rule never considers, whose "best" was
decided outside this benchmark; static_lora's PushT entry had epoch 2 carried across from
PushObj, where epoch 2 *is* correct, rather than selected on PushT. Both `method.yaml`
files now record what their own rule selects.

static_lora's pooled PushT score is unchanged at 0.3633 while its per-cohort numbers
moved (0.400/0.327 → 0.380/0.347). A pooled match is not evidence that nothing changed —
worth remembering when checking a re-run.

The precedent held: `EpochSelection` caught a transcription error once before
(`docs/CHECKPOINTS.md`, epoch 4 vs epoch 2), and running it caught two more. The pattern
in all three is the same — the second setting inherits the first setting's answer.

### AdaJEPA's 16-cell grid: worth the compute, and it caught the third

`StepSizeGrid` is 16 columns on an episode-isolated method: 3,200 planning processes on
PushObj and 2,400 on PushT, 2h05 and 1h55 wall on 8 idle A6000s. That was judged worth
paying, on three grounds: it is the one method the cost column exists to price, so
leaving it `unknown` empties the column of its purpose; the rule re-derives a declared
cell rather than trusting it, which had already caught two errors elsewhere the same
day; and the grid is itself the benchmark's headline measurement, previously inherited
from the predecessor and never made natively here.

**PushObj — success on the selection cohort, n=200 per cell**

| lr \ steps | 1 | 3 | 5 | 10 |
|---|---:|---:|---:|---:|
| 5e-4 | 0.515 | 0.630 | 0.670 | **0.680** |
| 2e-3 | 0.625 | 0.650 | 0.660 | 0.675 |
| 1e-2 | 0.670 | 0.620 | 0.640 | 0.645 |
| 5e-2 | 0.525 | 0.500 | 0.500 | 0.450 |

Selects `(steps=10, lr=5e-4)` — the declared cell, confirmed. Its test columns were
already on disk under identical parameters, so they resumed and the row is unchanged at
**0.6783**, now priced at **16 columns** instead of `unknown`.

**PushT** selects `(steps=10, lr=2e-3)`, **not** the declared `5e-4` — the third
mis-declared parameter of the day, and again on the second setting.

*The harness refused to report it.* `evaluate.py` stopped with

```
[refused] eval_outputs/adajepa/pusht/test200 already holds a column run with
    different parameters:  pred_lr: 0.0005 -> 0.002
```

which is the resume guard added this morning doing exactly its job: the committed PushT
columns were run at `5e-4`, resume would have kept them, and the reported mean would
have been two configurations averaged together. Those columns were archived and the row
re-run at the selected cell.

### What the grid measures natively

The predecessor's central claim is that the hyperparameter dominates the method. Over
this grid, on the PushObj selection cohort:

| quantity | range across the 16 cells |
|---|---|
| success | 0.450 → 0.680 |
| catastrophe rate | **2.1% → 23.0%** |
| compounding slope | −0.51 → +1.70 |
| median paired distance Δ vs frozen | −10 → +34 |
| median **absolute** final distance | 121 → 148 (**1.2×**) |

The first four reproduce the claim natively: one hyperparameter choice, inside a grid a
submitter would plausibly search, moves catastrophe rate by 11× and flips the compounding
slope's sign. Catastrophe rate is again the most discriminative single number, as §4
predicts.

The last row does **not** reproduce the predecessor's 12.7× median-distance span, and
should not be expected to. That figure comes from a *full-weight* sweep; this grid
confines the correction to a rank-2 LoRA subspace, and the predecessor's own measurement
for that case is a bounded +25 against the full-weight +1156. So the two agree: rank-2
bounds how far a bad hyperparameter can push the state, and the span collapses
accordingly. What it does not bound is the *outcome* — success still spans 0.230 and
catastrophe still spans 11× inside that subspace. A method can be well-behaved in weight
space and still be dominated by its step size.

### The cost column now says something

| method | before | after |
|---|---|---|
| adajepa | unknown | 16 |
| hyperjepa | unknown | 5 (pushobj) / 4 (pusht) |
| static_lora | unknown | 5 |
| restore_tta | 4 | 4 |
| pad | 0 | **0 (authored)** |
| frozen | 0 | 0 |

PAD's zero and frozen's zero were rendering identically while meaning different things:
frozen has no hyperparameters, while PAD's `encoder_lr`, `head_lr`, `steps` and
`buffer_size` were authored and never selected. Records now carry
`selection_cost_basis`, the leaderboard renders `0` / `0 (authored)` / `N` / `unknown` /
`N (inherited)`, and `methods/README.md` documents which zero a contributor is claiming.
PAD keeps its `FixedParams` rule rather than acquiring a fabricated one: authoring the
values is what actually happened, and the record should say so.

---

## The shift condition preserves PushObj's ordering; PushT's inversion is PushT's

2026-08-02. `pushobj_shift` — held-out shapes {I, small_tee, square} on the PushObj base,
one declared cohort at seed 100, n=150 — had been registered and enabled since M0 and
never run for any method. All five arms now have a row.

### It could not be run as configured, and the reason is a protocol violation

The setting declared `selection_seed=100` and `test_seeds=(100,)`. Its episodes are
held-out by construction, so there is no honest way to split them into a selection half
and a test half, and the registration recorded that by pointing both at the same cohort.
For the frozen arm that is harmless. For any of the four methods with a selection rule it
is not: the rule would have read the cohort the submission is then scored on, which is
precisely the leak `docs/PLAN.md` §3 says the harness must make structurally impossible.

Nothing caught it because nothing had ever run a method here.

`Setting.has_selection_cohort` is now false whenever the selection seed is also a test
seed, `SelectionHarness` refuses to be constructed for such a setting at all, and a
setting in that position must name where its parameters come from
(`inherits_selection_from`). `pushobj_shift` names `pushobj`, so each submission is
evaluated here **as it was selected in-distribution** — which is the more interesting
question about a shift condition anyway, and is priced as `N (inherited)` rather than as
a free zero.

### Results (n=150, frozen 0.293)

| method | success | ±1 SE | vs frozen | median dist Δ | catastrophe | compounding | regret |
|---|---:|---:|---:|---:|---:|---:|---:|
| AdaJEPA | **0.387** | 0.040 | +0.093 | +3.3 [−9, +20] | 14.6% [8%, 22%] | +0.65 [−0.10, +1.48] | 51% / +158 |
| Restore TTA | 0.340 | 0.039 | +0.047 | +0.1 [−12, +20] | 8.4% [3%, 14%] | +0.18 [−0.69, +0.95] | 51% / +106 |
| HyperJEPA | 0.320 | 0.038 | +0.027 | −2.0 [−11, +17] | 7.5% [3%, 13%] | +0.06 [−0.42, +0.56] | 46% / **+81** |
| Frozen | 0.293 | 0.037 | — | — | — | — | — |
| Static LoRA | 0.287 | 0.037 | −0.007 | **−3.6** [−14, +8] | 9.1% [4%, 15%] | −0.45 [−1.01, +0.24] | 48% / +93 |

The frozen arm lands on 0.2933 against the 0.293 registered in `paarbench/settings.py`.

**The ordering is PushObj's, not PushT's.** On PushObj the gradient methods lead; on
PushT they collapse and only HyperJEPA separates from frozen. Under held-out-shape shift
the PushObj ordering returns — AdaJEPA on top, then Restore TTA, then HyperJEPA, then
frozen. So whatever inverts the ranking on PushT is a property **of that setting** — a
different base checkpoint under visual shift — and not of distribution shift as such.
That is the question this run was cheap enough to answer, and it is the answer that makes
PointMaze worth more rather than less: a third *environment* is now the open question,
and a third *condition* has been shown not to substitute for one.

**What this does not support.** At n=150 the binomial SE is 0.038, so only AdaJEPA's
+0.093 clears two of them; Restore TTA and HyperJEPA are not separated from frozen or
from each other on success. **No continuous interval on this setting excludes zero** —
every distance, catastrophe and compounding CI above spans it. The ordering claim rests
on success rate at small n plus its agreement with the in-distribution result, and should
be treated as consistent-with rather than established. Raising n here is cheap (the
batched arms run in ~2.5 min a column) and is the obvious next thing.

One reading that does survive at this n: **Static LoRA is the only arm that goes below
frozen** (0.287 against 0.293). In distribution its constant correction captured 66% of
HyperJEPA's PushObj gain for zero adaptation cost. On shapes the correction was never
fitted to, it captures nothing and costs slightly more than nothing. A correction that
cannot react also cannot re-aim, which is a cleaner argument for conditioning than
anything the in-distribution comparison produced.

`pad` is absent by declaration: its inverse-dynamics head was pretrained on PushObj
transitions and `method.yaml` lists `settings: [pushobj]`. Extending it means pretraining
another head, which is out of scope here.

---

## What the corrected board says

2026-08-02, after every declared rule was run for real. Numbers below are the current
`LEADERBOARD.md`.

### PushObj (n=600, frozen 0.485)

| method | success | median dist Δ | catastrophe | compounding | adapt s | cost |
|---|---:|---:|---:|---:|---:|---:|
| AdaJEPA | 0.678 | +13 [+7, +22] | 14.0% [9%, 19%] | +0.91 [+0.40, +1.68] | 0.568 | 16 |
| Restore TTA | 0.678 | +5 [−0, +9] | **7.0%** [4%, 11%] | +0.32 [+0.01, +0.67] | 0.651 | 4 |
| HyperJEPA | 0.610 | **−6** [−14, −1] | 8.3% [5%, 12%] | **+0.01** [−0.16, +0.09] | 0.147 | 5 |
| Static LoRA | 0.567 | −5 [−9, +1] | 6.2% [3%, 10%] | −0.08 [−0.24, +0.04] | **0.000** | 5 |
| PAD | 0.553 | +4 [−2, +9] | 5.8% [3%, 9%] | +0.21 [−0.08, +0.51] | 0.029 | 0 (authored) |
| Frozen | 0.485 | — | — | — | 0.000 | 0 |

**Restore TTA and AdaJEPA now tie exactly on success at 0.678, and Restore TTA is better
on every other column** — half the catastrophe rate, a third the compounding slope, half
the tail regret, at a quarter of AdaJEPA's selection cost. Restoration is not buying
success here; it is buying everything else at no cost in success. That comparison was
invisible while Restore TTA sat at the success-selected p=0.001.

### PushT (n=300, frozen 0.350)

| method | success | median dist Δ | catastrophe | compounding | adapt s | cost |
|---|---:|---:|---:|---:|---:|---:|
| HyperJEPA | **0.440** | −13 [−30, +7] | 4.7% [2%, 9%] | **−0.55** [−1.16, −0.07] | 0.141 | 4 |
| AdaJEPA | 0.420 | **+23** [+5, +38] | **21.4%** [15%, 28%] | **+1.45** [+0.37, +2.33] | 0.570 | 16 |
| Static LoRA | 0.363 | +4 [−10, +13] | 11.2% [7%, 16%] | +0.08 [−0.38, +0.30] | 0.000 | 5 |
| Frozen | 0.350 | — | — | — | 0.000 | 0 |
| Restore TTA | 0.347 | **−3** [−5, −0] | **3.2%** [1%, 6%] | −0.06 [−0.20, +0.05] | 0.665 | 4 |

**PushT is where running the real rules changed the story, not just the digits.** Both
adaptive methods gained success, and in doing so they separated on everything else:

- AdaJEPA at its *own grid's* choice (`lr=2e-3`, up from the hand-carried `5e-4`) buys
  +0.070 success and pays +23 [+5, +38] distance, a 21.4% catastrophe rate and a
  +1.45 [+0.37, +2.33] compounding slope. All three intervals exclude zero. It is
  significantly worse than doing nothing on the metric §4 calls discriminating, while
  being better on the metric §4 calls weak.
- HyperJEPA is better than AdaJEPA on **every column at once** here — more success, less
  distance, a quarter the catastrophe rate, a compounding slope that excludes zero on the
  *negative* side, and 4× lower latency at 4 selection columns against 16.
- Restore TTA occupies the far end: the safest row on the board on every safety metric,
  and no success gain at all.

The earlier reading — "on PushObj the online learner wins on success; here the amortized
one does, but neither distance interval excludes zero" — was an artifact of both arms
being scored at parameters their own rules do not pick. With the rules run, the PushT
distance intervals do separate, and they separate in the direction that makes the
benchmark's argument: **the method that wins on success is significantly worse than
frozen on distance, and the method that wins on distance also wins on success here.**

At n=300 the success SE is 0.028, so HyperJEPA 0.440 against AdaJEPA 0.420 is *not*
separated on success. The separation is entirely in the continuous columns, which is the
point of reporting them.

### What moved, and why

| row | before | after | cause |
|---|---|---|---|
| restore_tta pushobj | 0.687 | 0.678 | declared objective changed to compounding slope |
| restore_tta pusht | 0.377 | 0.347 | same |
| adajepa pushobj | 0.678 | 0.678 | grid confirms the declared cell; cost unknown → 16 |
| adajepa pusht | 0.397 | 0.420 | grid picks `lr=2e-3`, not the declared `5e-4` |
| hyperjepa pushobj | 0.610 | 0.610 | rule confirms epoch 2; cost unknown → 5 |
| hyperjepa pusht | 0.430 | 0.440 | rule picks epoch 3, not `hyper_lora_best.pth` |
| static_lora pushobj | 0.567 | 0.567 | rule confirms epoch 2; cost unknown → 5 |
| static_lora pusht | 0.363 | 0.363 | rule picks epoch 1; pooled unchanged, per-cohort moved |
| pad pushobj | 0.553 | 0.553 | unchanged; cost 0 → `0 (authored)` |

**Every PushObj declaration was correct and three of four PushT declarations were not.**
The common cause is that PushT was added second and inherited PushObj's answers instead
of running its own selection. That is a specific, cheap failure mode worth naming for
anyone adding a third setting: *a new setting needs its rule re-run, not its parameters
copied* — and the harness cannot detect the copy, because a copied parameter is a
perfectly valid parameter. What detects it is running the rule.

No selection cost column reads `unknown` any more.

---

## Re-cohorting PointMaze, and the evaluation-mode artifact it exposed

2026-08-03. §2.3's PointMaze question is resolved — **re-cohort** — and running the
setting for the first time exposed a harness fault that affects every setting.

### The re-cohorting is measured, not argued

Frozen was run on **eight** candidate cohorts (400 episodes, CPU-only, 0 failures). It
reproduces both recorded numbers exactly: seed 300 → **0.880**, seeds 0/1/2 → **0.7333**.
That is also the first validation of the port on a third setting.

| seed | 300 | 1 | 3 | 6 | 4 | 0 | 2 | 5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| frozen | **0.880** | 0.840 | 0.840 | 0.840 | 0.740 | 0.700 | 0.660 | 0.660 |

**§2.3's framing was half wrong.** A cohort seed only selects *which* 50 environment
seeds you get (`seed * eval_episode_total + index + 1`), so "the harder distribution" is
not a different distribution — it is a different draw. Pooled frozen over all eight is
**0.770**. Between-cohort SD is 0.090 against the 0.060 that binomial draws alone predict
(variance ratio 2.29, χ²₇ = 16.0, p = 0.025), so cohorts do differ beyond noise, but only
about half the spread is real and **no seed choice buys much headroom**.

What *was* wrong is narrower and worse: **seed 300 is the easiest of the eight**, and a
selection cohort easier than test flatters every candidate scored on it.

**The split.** Selection **seed 4** (0.740); test **seeds 0, 1, 2, 3, 5, 6** (0.757,
n=300). Seed 300 retired. Selection now sits marginally *harder* than test instead of
0.147 easier. `n_evals` stays 50 — changing it silently redraws every cohort and would
void the sweep the split was chosen from — so the extra power comes from pooling six
already-measured cohorts, free.

Frozen through the real harness on that split: **0.7600 (n=300)**, per cohort
0.700 / 0.880 / 0.700 / 0.820 / 0.680 / 0.780, ~78 min per cohort on CPU with the GPUs
idle. Against the 0.757 declared from the probe.

Enabling it needed five things that had been hardcoded to the pushing settings, now
declared per setting: `model_epoch` (PointMaze selects epoch 3; `latest` is a different,
later model), `goal_source` (`dset`, so `targets_path` returns `None` instead of
fabricating a `data/pushobj_eval/` path), `cpu_only`, `needs_mujoco`, and a single named
variant `umaze` because `shapes=()` makes `run_column` refuse the setting. Dataset-path
precedence was also flipped so a setting's declaration beats the checkpoint's: PointMaze's
base records an SMB directory whose `obses/` subtree is ~3 TB, and staging would have
tried to copy all of it into tmpfs.

### Batched and episode-isolated evaluation are not interchangeable

The batched frozen column disagreed with the episode-isolated probe. They are not
different draws — `plan.py:133` falls back `eval_episode_total → n_evals` so both derive
seeds `[1..50]`, and `plan.py:414-419` deliberately aligns the goal draw. Same model,
same episodes, no adapter:

| setting | outcome flips | agreeing episodes bit-identical |
|---|---:|---:|
| pushobj / test100 / T | 2 of 50 (4%) | **0 of 48** |
| pointmaze, 6 cohorts | 27 of 300 (9%) | — |

`planning/gd.py:149`'s `if np.all(successes): break` would couple a batch, but it is dead
at `eval_every: -1`. What remains is numerical — `total_loss = loss.mean() * n_evals`,
and kernels that depend on batch shape — amplified through 100 GD steps × 20 replans of a
closed loop.

This matters because the leaderboard paired **episode-isolated** arms (adajepa,
restore_tta, pad) against the **batched** frozen column, and §6 makes planner determinism
the thing that licenses per-episode pairing. Note how it hid: pushobj/T's success came out
identically 0.580 either way, because the two flips cancelled.

### The noise floor, in leaderboard units

§6 says not to introduce planner stochasticity without a noise-floor control. This is that
control, and it had never been measured: frozen run episode-isolated, scored against
frozen batched *as if it were a method*.

| setting | n | success iso/batched | median dist Δ | catastrophe | compounding |
|---|---:|---:|---:|---:|---:|
| pushobj | 600 | 0.4883 / 0.4850 | −0.01 [−0.04, +0.00] | **0.66%** [0%, 1.6%] | −0.0003 |
| pusht | 300 | 0.3567 / 0.3500 | −0.14 [−1.44, +0.06] | **4.23%** [1.6%, 7.4%] | −0.0065 |
| pushobj_shift | 150 | 0.2933 / 0.2933 | −0.02 [−0.36, +0.02] | **0.95%** [0%, 2.9%] | −0.0000 |
| pointmaze | 300 | 0.7567 / 0.7600 | −0.47 [−1.51, +0.12] | **10.17%** [3.4%, 18.6%] | −0.0095 |

The per-episode divergences are real but symmetric — on pushobj the median *absolute*
difference is 0.099 while the median *signed* paired delta is −0.01 — so they cancel in
aggregate rather than shifting it. On the pushing settings every reported effect clears
the floor by an order of magnitude or more.

**On PointMaze it does not.** Its catastrophe floor is 10.17%, and real methods elsewhere
on this benchmark score 3.2%–24.5% on that metric. Its distance floor is 12.1% of a
typical final distance (median 3.85 units). So §4's most discriminative number is, on this
setting, indistinguishable from the harness's own evaluation mode. This is a *second*
limitation, independent of the compression §2.3 complained about, and it revises the
conclusion written there earlier the same day: the setting does not simply "earn its place
through the continuous metrics".

### The fix, and what it moved

An isolated arm is now paired against a frozen column run **episode-isolated too**, so the
mode difference cancels instead of being charged to the method. `paarbench/schema.py`
tags every row with its mode, `metrics.reference_for` picks the matching frozen column,
and `scripts/leaderboard.py` warns when no matching reference exists rather than pairing
across modes silently. Four tests pin it. Isolated frozen references now exist for all
four settings (600 / 300 / 150 / 300 episodes).

Corrected numbers, mode-matched. **Two claims made earlier today do not survive**:

| row | as committed | mode-matched | verdict |
|---|---|---|---|
| restore_tta pusht, distance | −2.6 [−4.7, −0.4] | **−0 [−3, +2]** | **withdrawn** — the interval now includes zero |
| restore_tta pushobj, distance | +5 [−0, +9] | **+6 [+3, +11]** | **strengthened, against the method** — now excludes zero |
| adajepa pushobj, distance | +13 [+7, +22] | +14 [+8, +23] | unchanged in kind |
| adajepa pusht, distance | +23 [+5, +38] | +27 [+13, +52] | unchanged in kind |
| adajepa pusht, catastrophe | 21.4% | 24.5% | unchanged in kind |
| pad pushobj, distance | +4 [−2, +9] | +5 [−0, +11] | unchanged in kind |

The correction is systematically against the isolated arms, because the isolated frozen
reference is slightly *better* than the batched one (0.4883 vs 0.4850 on pushobj).

**The withdrawn claim matters and should not be quietly dropped.** Earlier today this log
reported the slope-selected Restore TTA on PushT as "the only entry anywhere on this
benchmark whose paired distance interval excludes zero on the good side". Mode-matched, it
does not: −0 [−3, +2]. What survives is the rest of that row — catastrophe 5.4% [3%, 9%]
against the success-selected variant's 16.4%, compounding +0.01 against +0.96, tail regret
+68 against +212, at no success gain (0.347 against frozen's 0.350). The selection-
objective finding stands; the specific superlative does not.

No ordering on any leaderboard changed.

### Still open on PointMaze

Only the frozen arm has run. Before spending method compute here, note what the two
measurements together say: 24.3% headroom, success resolving only effects ≥ +0.050 at
n=300, a catastrophe floor of 10.2%, and a distance floor of 12% of a typical distance —
against a predecessor effect on this domain of +0.044. The re-cohorting was necessary and
is done; whether the setting can support a claim is a separate question, and the honest
answer right now is that it cannot support a catastrophe-rate one.
