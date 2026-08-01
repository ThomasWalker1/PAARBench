# PlanActAdaptRepeatBench — a benchmark for test-time adaptation of latent world models

**Status:** **M0 complete** (2026-08-01) — scaffold up, frozen PushObj column reproduces
0.490 per-shape-exactly. See `RESULTS.md` for the acceptance record, `PROVENANCE.md` for
what was copied from where, and `docs/ADAPTER_PROTOCOL.md` for the M1 design. **M1 is next.**
**Predecessor:** `~/HyperJEPA/` — the source of the findings below. **Treat it as read-only.**
It is actively maintained by a separate session, so pin your reads to commit **`628dff7`** rather
than to whatever `HEAD` happens to be; later commits may revise the paper's tables.

---

## 0. Read this first

This is a **fresh directory with no code**. If you are a new session picking this up, the
useful order is: §1 (why), §2 (the honest blockers), §3–§6 (design decisions already made),
§7 (what to copy), §8 (milestones). Do not start coding before reading §2 — two of the
blockers can sink the project and are worth confirming with the owner early.

Nothing in §3–§6 is sacred, but each decision there has a reason recorded. If you change one,
change the reason too.

---

## 1. Why this exists

`~/HyperJEPA/` set out to show that an amortized hypernetwork adapts a frozen JEPA world model
better than online gradient test-time adaptation (TTA). **It does not** — a tuned online learner
reaches 0.678 against the hypernetwork's 0.610 on the primary domain (paired, n=600, p=0.0003).

What that project found instead is a **measurement** result, and it is the reason to build a
benchmark rather than another method:

> The way the field scores test-time adaptation hides most of what TTA does. Success rate over a
> best-found hyperparameter conceals (a) that the hyperparameter dominates the method, (b) that a
> wrong hyperparameter does not merely waste the adaptation signal but reverses it, and (c) that
> the damage compounds over the control loop. Score it properly and the ranking changes.

Verified numbers behind that claim (all from `~/HyperJEPA/`, reproducible with no GPU via
`~/HyperJEPA/scripts/rescore_distance.py`):

| finding | number |
|---|---|
| Closed-loop success within a single 16-cell $(\eta, K)$ grid | spans 0.010 → 0.690 |
| Full-weight sweep **mean** vs frozen model | below frozen in **all 4 settings** (0.360/0.490, 0.326/0.360, 0.802/0.880, 0.208/0.293) |
| Median final distance-to-goal span within one sweep | **12.7×** PushObj, 8.8× PushT, **14.7×** held-out shapes, 1.4× PointMaze |
| Worst cell's median paired degradation vs frozen | **+1337** / +1024 / +1861 distance units, $p<10^{-4}$ |
| Episodes ending >2× further from goal than not adapting | **18–178 of 200** depending only on the hyperparameter |
| Degradation across replans (accumulated update) | monotone, 0 → **+1156** over 20 replans |
| Same, restricted to a rank-2 subspace | bounded at +25 |
| Same, correction recomputed from frozen weights each replan | flat at ≈0 |
| Cost of inheriting a step size across parameterizations | **nothing** (PushObj: 0.690 either way) to **catastrophic** (PushT: −0.427 success, +820 distance, 23→118 destroyed) |
| Per-episode oracle over the grid vs best *fixed* cell | 0.815 vs 0.690 — **+0.125 that no fixed hyperparameter reaches**; 92/120 cell pairs non-nested |

The last row is why a leaderboard is interesting rather than decorative: the required amount of
adaptation is genuinely episode-dependent, so there is real headroom that no current method
captures, and a benchmark that measures it gives people something to climb.

**The contribution is the protocol, not the environments.** Be honest about this in any writeup.
The environment suite is thin (§2.3). What no one else has is the scoring design in §3.

---

## 2. Blockers and risks — confirm these before investing

### 2.1 There is no standardized adaptation interface anywhere yet  *(work item, not a risk)*

The thing the benchmark would claim as its core — "a standardized mechanism to specify a TTA
method" — does not exist. In `~/HyperJEPA/` the two methods have unrelated APIs and
`planning/mpc.py` has a hardcoded branch for each:

```
AdaJEPAAdapter    .append(prev_obs, actions, obs) -> .step();  .refresh_interval, .buffer, .log_prefix
HyperJEPAAdapter  .maybe_apply_episode_start(obs), .append_executed_transitions(...),
                  .maybe_refresh_after_mpc(obs), .apply(obs)
```

Designing one protocol and porting both is Milestone 1. Roughly a day. Low risk, high value —
worth doing even if the benchmark is abandoned.

### 2.2 Three methods is not a leaderboard  *(resolved 2026-08-01: descoped)*

> **Owner decision.** The deliverable is **infrastructure that makes proposing,
> implementing and evaluating a method easy** — not a populated leaderboard on day one.
> The leaderboard fills in over time.
>
> *Reason this is the right call:* the original framing made M4 (4+ method ports, 1–3
> weeks) the gate on everything, and it was rated the project's largest risk. But the
> contribution was never the entries — §1 already says "the contribution is the protocol".
> A harness that makes the 7th method cheap to add is worth more than six methods bolted
> onto a harness nobody else can extend. It also inverts the risk: M1–M3 are each
> independently useful, whereas M4 was all-or-nothing.
>
> *What this changes:* M1–M3 become the deliverable and their quality bar goes **up** —
> specifically the ease of adding a method is now a thing to design for and test, not a
> side effect. M4 becomes ongoing. The "if you cannot land 4+ new methods, this is not a
> benchmark paper" gate below no longer applies.
>
> *What this does not change:* the honesty requirements. A leaderboard with three entries
> must still report per-metric CIs and must still not display unresolvable orderings as
> real (§2.5).

Existing arms: frozen, online gradient TTA, an unconditioned (static) correction, HyperJEPA.
Candidates to add over time, best first:

- **PAD** (Hansen et al., policy adaptation during deployment) — closest fit, already cited in
  the HyperJEPA bibliography.
- **T3A** — training-free, cheap, a good "free lunch" reference point.
- **CoTTA**-style continual TTA with restoration — directly probes the accumulation pathology.
- **MAML / learned initialization** — keeps the inner loop, learns where it starts.
- **Entropy-minimization family (TENT, SHOT, EATA)** — awkward here: they need a classifier head
  and there isn't one. *That awkwardness is itself a reportable finding* — the dominant TTA
  family does not transfer to latent world models. Say so rather than forcing it.

~~**If you cannot land 4+ new methods, this is not a benchmark paper.**~~ Superseded by the
decision above. Add methods opportunistically; each one is also a test of whether the
interface is actually easy to implement against, so treat friction encountered while
porting as a bug in §5, not as a cost of the port.

### 2.3 The setting suite is thin, and one setting is broken as configured

- PushObj and PushT are both pushing tasks on released DINO-WM-style checkpoints — one task
  family, two checkpoints.
- **PointMaze is unusable as configured**: frozen sits at 0.880 on its selection cohort, which
  compresses everything (distance span 1.4× against PushObj's 12.7×), and n=50 gives SE ≈ 0.045.
  Either re-cohort to the harder distribution (its *test* cohort is frozen 0.733) or drop it.
  > **Owner decision (2026-08-01): defer.** Build the infrastructure so PointMaze *can* be
  > added; decide re-cohort-vs-drop later. It is registered in `paarbench/settings.py` with
  > `enabled=False` and its blocking reason recorded there, so `settings.get("pointmaze")`
  > raises rather than silently returning a compressed setting. Its base checkpoints (5.9 GB,
  > three variants) are deliberately **not staged** — which variant to take depends on how the
  > re-cohorting resolves. *Reason to defer rather than drop:* the harness work is identical
  > either way, and given how thin the suite is, a second task family is worth keeping
  > reachable. This also keeps the `pointmaze` dependency extra (mujoco-py, d4rl) off the
  > default install path, which is worth something on its own.
- Held-out-shape PushObj is a distribution-shift *condition* on an existing base, not a new
  environment. It is valuable (the span *widens* to 14.7× there) but should not be counted as a
  fourth environment.

Realistically: **2 environments + 1 shift condition**. Adding a genuinely different domain
(deformable manipulation is already in `~/HyperJEPA/env/deformable_env/`, and the dataset is at
`/mnt/richb/tw78/data/datasets/deformable.zip`) would materially strengthen it.

### 2.4 Base checkpoints are not distributable as-is  *(deferred 2026-08-01, not resolved)*

`~/HyperJEPA/` uses released third-party base world models and its own Limitations section
concedes the phase-two pool is "a contact-rich reconstruction rather than the base's own
training data." A benchmark must ship pinned, downloadable base models or submissions are not
comparable.

> **Owner decision.** Keep the inherited checkpoints in a subdirectory and **develop the
> benchmark against them as they stand.** The training code that reproduces them is added
> later.
>
> *Reason:* this is a packaging problem, not a design problem. Nothing in §3–§5 depends on
> where the weights came from, so blocking the harness on retraining would serialize two
> independent pieces of work for no benefit. Staged inventory and provenance are in
> `docs/CHECKPOINTS.md`.
>
> *What is still true, and must not get lost:* until the bases are retrained on owned data,
> **a third party cannot reproduce a submission's numbers.** That caps external adoption
> however good the harness is, so this is deferred rather than solved. Two constraints
> follow, both recorded in `docs/CHECKPOINTS.md`:
> - Retraining **invalidates every reproduction target** (frozen 0.485, HyperJEPA 0.610,
>   tuned online 0.678, the §4 distance spans), because all of them are defined against
>   these specific weights. It therefore has to come *after* M1/M2 validate the port, and
>   it implies a second full evaluation sweep to re-baseline.
> - `train.py`, `train_hyper_lora.py` and `conf/train.yaml` were left off the §7 take list.
>   Re-extract them from `628dff7` when the training work starts; do not reconstruct them.

### 2.5 n is too small to order a success-rate leaderboard

At n=200 the binomial SE is 0.035, so adjacent leaderboard entries will not be separable on
success rate. Two mitigations, use both: lean on the continuous metrics (far more power), and
raise n. Report per-metric confidence intervals on the leaderboard — a benchmark that displays
unresolvable orderings as if they were real is worse than no benchmark.

### 2.6 Evaluation cost vs. the central finding

The headline result is that the hyperparameter dominates. Scoring a submission across a 16-cell
grid is 16× the cost of one evaluation and will kill adoption. §3 resolves this.

---

## 3. The scoring protocol — the actual contribution

Three options were considered:

1. **Score the best cell.** Reproduces exactly the pathology the benchmark exists to expose. No.
2. **Score the whole grid.** Honest but 16× cost. No.
3. **Score the procedure.** ✅

> **A submission is an adaptation method *together with its hyperparameter-selection rule*.**
> The rule may read only the **selection cohort**. It is then frozen and run once on **held-out
> cohorts**, which is what the leaderboard reports.

This directly encodes the finding that the selection rule dominates the method, and it makes
"we tuned on the test set" structurally impossible rather than merely discouraged. It also
prices honestly: a method that needs a 16-cell sweep to work pays for that sweep in its
reported selection cost, and a method with no hyperparameters pays nothing.

Rules to enforce in the harness:

- **Cohort separation.** Selection cohort seeds and test cohort seeds are disjoint and declared
  by the benchmark, not the submitter. The harness should refuse to run a selection rule against
  test seeds.
- **Declare selection cost.** Number of planning columns the rule consumed. Report it as a
  leaderboard column — this is where the tuning burden becomes visible.
- **No online outcome access.** A method may use anything computable *during* an episode from
  observations and executed transitions, but not the episode's success or final distance. This is
  the constraint that makes the setting what it is (an episode's outcome is known only once it is
  over) and it must be enforced by the interface, not by trust.
- **Report the frozen model on every setting**, always, as the do-nothing reference. Every metric
  below is relative to it.

---

## 4. Metrics

The first four are what the predecessor project showed actually discriminate; the last two are
resource accounting. Do not report means or standard deviations on distance — the metric is
heavy-tailed (distances of 10⁴ occur under the *frozen* model, whose own max is ~9,000), so a
variance is a statement about two or three episodes.

| metric | definition | why |
|---|---|---|
| **Success rate** | closed-loop goal achievement | conventional, low power at these n |
| **Median distance to goal** | median final state distance, paired against frozen via Wilcoxon signed-rank | the discriminating metric; far more power than binary success |
| **Catastrophe rate** | fraction of episodes ending >2× further from goal than the frozen model | the single most discriminative number found; ranged 9%–89% across one grid |
| **Compounding slope** | median paired degradation vs frozen as a function of replan index | separates accumulated from recomputed updates; the mechanism metric |
| **Regret vs frozen** | does adapting ever lose to not adapting, and by how much | a TTA method that can be worse than doing nothing must show it |
| **Latency** | per-replan seconds, decomposed into planner vs adaptation | already measured in the predecessor; adaptation must be separable |
| **Peak memory** | peak CUDA allocation during a replan | not yet measured anywhere; `torch.cuda.max_memory_allocated`, few lines |

Leaderboard should be **multi-objective and not collapsed to a single score**. The interesting
finding is a frontier — the predecessor's hypernetwork is worse on success and better on
catastrophe rate and latency. Collapsing that to one number destroys the result the benchmark
exists to show. Offer sorting, not a scalar ranking.

---

## 5. The adapter interface

One protocol, called uniformly by the planner. Sketch — refine on contact with the two ports:

```python
class TestTimeAdapter(Protocol):
    """Everything a TTA method may do inside the control loop.

    The planner calls these; the method never sees episode outcome. Implementations
    must be stateless across episodes after on_episode_start().
    """

    def on_episode_start(self, obs_0, goal) -> None:
        """Reset all per-episode state. Must fully undo any previous episode."""

    def on_transition(self, obs, action, next_obs) -> None:
        """One executed action chunk and the observation it produced."""

    def before_plan(self, obs) -> None:
        """Last hook before the planner solves. Apply/refresh any correction here."""

    def state_dict(self) -> dict: ...
    def metrics(self) -> dict:
        """Per-replan scalars for logging (loss, correction norm, ...)."""
```

Design notes carried over from the predecessor, each with a reason:

- **`on_episode_start` must fully restore the base model.** The predecessor's hypernetwork
  recomputes its correction from frozen weights every replan; the online method walks an
  optimizer trajectory. If the harness does not force a reset, cross-episode leakage silently
  changes results. Add a test that asserts base weights are bit-identical after two episodes.
- **Keep `before_plan` separate from `on_transition`.** The predecessor's two methods differ
  exactly here (one updates on transitions, one regenerates at plan time) and collapsing them
  loses that distinction.
- **Adaptation cost must be measurable in isolation**, so time `before_plan` + `on_transition`
  separately from the planner. The predecessor's latency table depends on this decomposition.
- **Peak memory** should be sampled around the same hooks.

Correctness test for the refactor: port `AdaJEPAAdapter` and `HyperJEPAAdapter` onto the
protocol and **reproduce the predecessor's published numbers exactly** — frozen 0.485,
HyperJEPA 0.610, tuned online 0.678 on PushObj test seeds (n=600). Anything else means the port
changed semantics.

---

## 6. Settings suite (initial)

| id | base | cohorts | frozen success | notes |
|---|---|---|---|---|
| `pushobj` | `pushobj_shape_shift` | selection seed 300; test 100/200/400 | 0.490 sel / 0.485 test | primary; 4 shapes, n=200 sel / 600 test |
| `pushobj_shift` | same | seed 100, held-out shapes {I, small tee, square} | 0.293 | distribution-shift condition, n=150 |
| `pusht` | `pusht_visual_shift` | selection 100; test 200/400 | 0.360 | 3 shapes, n=150 |
| `pointmaze` | `pointmaze/...` | selection 300; test 0/1/2 | 0.880 sel / 0.733 test | **re-cohort or drop** — see §2.3 |

Planner config is shared and must stay fixed across methods: goal horizon 25, 100 GD steps,
`sample_type: zero`, `action_noise: 0`. **The planner being deterministic is load-bearing** — it
is what licenses per-episode paired comparison between methods, since for a fixed episode and
model the closed loop is reproducible. Do not introduce planner stochasticity without also
adding a noise-floor control.

---

## 7. What to reuse from `~/HyperJEPA/`

Copy, do not symlink — the point is a clean slate that cannot break the predecessor.

**Take:**

```
plan.py                      eval entry point (hydra); adapt to the new interface
planning/mpc.py              the control loop — rewrite the two adapter branches into one
planning/{gd,cem}.py         planners, unchanged
planning/evaluator.py        rollout metrics
planning/adaptation.py       -> port to TestTimeAdapter as the "online gradient TTA" baseline
planning/hyper_adapter.py    -> port as the "amortized hypernetwork" baseline
env/                         PushT / PointMaze / deformable / wall envs
conf/                        hydra configs; prune the ad-hoc plan_*_local.yaml variants
models/, preprocessor.py, utils.py, datasets/, metrics/
scripts/episode_outcomes.py  per-episode loader — but see below
scripts/rescore_distance.py  the distance/catastrophe metric implementations
scripts/paired_indist_test.py  paired McNemar / bootstrap
```

**Do not take:**

- `eval_outputs/` — 14 directories in three incompatible layouts. `episode_outcomes.py` exists
  only to paper over that. **Define one output schema up front** (one row per episode per replan,
  parquet or jsonl) and write metrics against it. Port the *metric definitions* from
  `rescore_distance.py`, not its loader.
- `scripts/ablation_grid.py`, `run_ablation_*`, `objrelfull_*` — HyperJEPA-specific design-space
  sweeps and one-off replication drivers. Nothing here generalizes to a benchmark harness.
- `checkpoints/pushobj_ablations*`, `*_adapters`, `pvs_adapters` — 33-cell ablation artifacts.
- `paper/` — the predecessor's paper.

**Traps recorded in the predecessor; do not rediscover them:**

1. **Never decompose a metric by each method's own failures.** `d | success` ≈ 110 for every
   method while `d | failure` rises with success rate, because better methods convert the easy
   near-goal failures and keep a harder residual. It looks like a real effect and is pure
   selection. Condition on a *fixed* set (frozen's outcome, or both-fail).
2. **Success is absorbing.** `planning/mpc.py` zeroes actions for succeeded episodes, so a
   successful episode's distance is held at its value on the replan it succeeded. Any
   distance comparison across methods must account for this — restrict to episodes all arms fail,
   or state the coupling.
3. **Dispersion claims do not survive.** Log-variance ratios between methods came out 0.96–0.99,
   permutation p > 0.5. The real effects are shifts in *location*. Do not build a "stability"
   metric on variance.
4. **Grid maxima are optimistic.** On PushT the best cell fell 0.513 → 0.403 when re-run on
   held-out cohorts while frozen moved 0.010. Always re-run selected configurations; never quote
   a selection-cohort maximum as a score. §3's protocol enforces this.

---

## 8. Milestones

Each has an acceptance criterion. Do not advance without it.

**M0 — scaffold (0.5 d).** ✅ **Done 2026-08-01.** `git init`; uv project; copy §7 "take" list;
`.venv` per §9; smoke-run one frozen evaluation column.
*Accept:* a frozen PushObj column runs end to end and reports success 0.490 ± noise on seed 300.
*Result:* **0.4900** (T 0.500 / L 0.440 / Z 0.600 / + 0.420) — matches the predecessor
per-shape, not merely on the mean. `RESULTS.md` §M0.

Also landed, ahead of their milestones because M0 needed them: `paarbench/settings.py` (the §6
suite with the cohort split declared as data, `pointmaze` registered but disabled per §2.3),
`paarbench/staging.py` (§9 tmpfs staging in the harness rather than per-config overrides), and
`paarbench/adapter.py` + `docs/ADAPTER_PROTOCOL.md` (the §5 protocol specified, with the
call-site analysis M1 must satisfy; the port itself is not started).

One finding worth carrying into M1, recorded in `docs/ADAPTER_PROTOCOL.md`: `AdaJEPAAdapter`
has **no episode-reset method at all**, so §5's "must fully restore the base model" is new
work for the online method rather than a port. Two further asymmetries the §5 sketch does not
yet cover — `on_transition` must carry the *full* rollout plus `frameskip` (the amortized
method aligns one feature per executed action; the online method was handed only the chunk's
final observation), and refresh scheduling must move inside the method (the planner gates one
and not the other, which *is* the branch M1 removes).

**M1 — the interface (1–2 d).** Define `TestTimeAdapter`. Port both existing methods. Collapse
mpc.py's two branches. Add the reset test from §5.
*Accept:* frozen 0.485, HyperJEPA 0.610, tuned online 0.678 reproduced on PushObj test seeds
(n=600); base weights bit-identical across episode boundaries.

*Sequencing (added 2026-08-01).* Split the acceptance into **M1a: reproduce before
refactoring.** Run both methods through the *unmodified* ported code first, via the configs in
`docs/reference/`, and land those numbers in `RESULTS.md`. Only then write the protocol
implementations and re-run. *Reason:* the single-step version conflates a broken port with a
PAARBench environment that differs from the predecessor in some way the frozen column never
exercised — the frozen arm touches no adapter code at all. With a pre-refactor baseline
measured *in this repo*, any later disagreement is unambiguously the port. Costs one extra
evaluation round; buys a clean bisect on the one milestone whose whole premise is "anything
else means the port changed semantics".

*Artifacts pinned for this.* HyperJEPA 0.610 is `pushobj_adapters/hyper_r2_distill0`
**epoch 4**; tuned online 0.678 is the grid cell `(steps=10, lr=5e-4)`, rank-2 `predlast`.
Both staged — see `docs/CHECKPOINTS.md`.

**M2 — output schema + metrics (1–2 d).** One canonical per-episode/per-replan output. Implement
all seven §4 metrics against it, with CIs.
*Accept:* every distance number in `~/HyperJEPA/paper/floats/safety_span.tex` reproduced from the
new schema (12.7× / 8.8× / 14.7× / 1.4× spans; catastrophe 18–178).

*Why this came before more baselines (2026-08-01).* Until it landed, the harness recorded
**only means** — `mpc/mean_state_dist` and nothing per-episode. That is the one summary §4
says never to use on distance, since the metric is heavy-tailed, and it makes every
discriminating metric uncomputable: median paired distance, catastrophe rate and
compounding slope all need the distribution. A benchmark whose whole premise is "success
rate hides what TTA does" was reporting success rate and nothing else.

Landed: `planning/evaluator.py` emits per-episode values instead of collapsing them,
`planning/mpc.py` writes `episodes.jsonl` (one row per episode per replan),
`paarbench/schema.py` reads it identically for batched and episode-isolated runs, and
`paarbench/metrics.py` implements the metrics with the three traps from §7 designed in
rather than left to the caller. Still outstanding: the `safety_span.tex` reproduction
itself, and CIs on the continuous metrics.

**M3 — harness + protocol (2–3 d).** Selection/test cohort separation enforced in code; a
submission is a method + selection rule; selection cost recorded; leaderboard generation.
*Accept:* a deliberately cheating submission that peeks at test seeds is *rejected by the
harness*, not by review.

*Reshaped 2026-08-01 by §10 Q2/Q5.* This is now the **primary deliverable**, and its target is
contributor ergonomics rather than leaderboard polish. Concretely: a submission is a directory
containing one `TestTimeAdapter` implementation, one selection rule, and one config; it is
proposed as a pull request; CI runs the declared columns and rejects any rule that reads a test
cohort. Static generated leaderboard page, since there is nothing to serve.
*Additional accept:* a newcomer can add a trivial method (e.g. T3A, or a no-op with a
one-line selection rule) end to end **without editing anything outside their own submission
directory**. If that requires touching the planner, the harness, or a registry by hand, §5 is
not done — this is the bar §2.2's descoping raises.

**M4 — baselines (ongoing, no longer a gate).** Port methods from §2.2 opportunistically.
*Accept (per method):* one declared selection rule, one reported selection cost, one
leaderboard row. Each port is also a usability test of M1/M3 — friction is a bug in the
interface, not a cost of the port. Order: whatever is cheapest next.

**M5 — settings (deferred, see §2.3/§2.4).** PointMaze and base-model distribution are both
explicitly deferred by owner decision; deformable is a yes but not started. What remains
schedulable now is **adding the deformable domain**, which is the cheapest route to genuine
task diversity since `env/deformable_env/` is already ported and the dataset exists.
*Accept (deformable):* a frozen column runs end to end and the setting is registered in
`paarbench/settings.py` with a declared cohort split.
*Accept (distribution, when it happens):* every base model pinned by hash and downloadable by
a third party — and the whole suite re-baselined, since retraining invalidates every
reproduction target.

**Then, and only then:** decide venue. NeurIPS Datasets & Benchmarks is the natural track, and it
is a *different deadline and bar* from the predecessor's target. Note that §2.4's deferral bears
directly on this: a D&B submission whose base models a reviewer cannot download is a weak one,
so the training work has to be done *before* a venue, even though it is not needed before the
harness.

**Scope boundary: this project is PAARBench only.** `~/HyperJEPA/` is owned by a separate session
and needs nothing from you. Do not run its experiments, edit its paper, or wait on its results.
The one dependency runs the other way: §5 and §8/M1–M2 ask you to *reproduce* a handful of its
published numbers as correctness tests for the port. Those numbers are already on disk and
final — take them from `~/HyperJEPA/paper/floats/` at commit `628dff7`
(`git -C ~/HyperJEPA show 628dff7:paper/floats/safety_span.tex`), and treat that repo as
read-only. If a number you are trying to reproduce disagrees with that commit, the port is
wrong, not the number.

---

## 9. Environment and infrastructure notes

- **Python/venv:** the predecessor uses `.venv` with **torch 2.3.0+cu121**. Match it initially so
  ported numbers are comparable, then upgrade deliberately.
- **CUDA gotcha (bit the owner before):** default resolution picks a torch built for a newer CUDA
  than this machine's driver supports, and `torch.cuda.is_available()` then silently returns
  False. Pin the cu121 index explicitly in `pyproject.toml`; `uv`'s `--torch-backend`
  auto-detection does not apply to the `uv add` / `uv sync` project workflow.
- **Hardware:** 8× RTX A6000 (49 GB). Evaluation columns are CPU/GPU-light but numerous — the
  predecessor's `scripts/run_eval_queue.py` fan-out pattern (one column per GPU slot) is worth
  keeping.
- **Data:**
  - `/mnt/richb/tw78/data/hyperjepa_pushobj/` — `pushobj_multishape`, `heldout_{I,small_tee,square}`
  - `/mnt/richb/tw78/data/datasets/` — deformable (zip parts)
  - Stage to tmpfs before any fan-out: `/dev/shm/tw78/data/pushobj_multishape`. The SMB mount
    goes down under a large fan-out; several predecessor configs override `dataset_data_path`
    for exactly this reason. Bake staging into the harness rather than per-config overrides.
- **Reference eval invocation** (predecessor, for shape):

  ```
  .venv/bin/python plan.py --config-name <cfg> ckpt_base_path=<base> model_epoch=latest \
      goal_source=segments +eval_data_path=<targets.pkl> +wandb_logging=false \
      hydra.run.dir=<out> seed=<seed> n_evals=<n> goal_H=25 \
      planner.sub_planner.opt_steps=100 decode_for_viz=false
  ```

- **Background jobs:** tee to disk. Session interruptions can lose captured stdout, and long
  training/eval runs have been lost that way before.

---

## 10. Open questions for the owner

All five are answered. The three settled on 2026-08-01 reshaped the project enough that
§2.2, §2.3 and §2.4 were rewritten to match; this section is the short record.

1. **Is §2.4 solvable?** If base world models cannot be distributed, the benchmark cannot be
   adopted externally and should be scoped as an internal evaluation suite instead.
   - A: Train own models from scratch — but **later**. *(2026-08-01)* Keep the inherited
     checkpoints in a subdirectory and develop against them; add the training code that
     reproduces them afterwards. See §2.4 and `docs/CHECKPOINTS.md`. Note this leaves the
     external-adoption limitation standing in the meantime — deferred, not solved.
2. **Appetite for M4?** 4+ method ports is the difference between a benchmark and a protocol
   section.
   - A: Ensure HyperJEPA and AdaJEPA are implemented, then we can think about other protocols.
   - A *(2026-08-01, superseding the framing)*: **it does not have to be a full leaderboard
     yet.** The goal is infrastructure that makes proposing, implementing and evaluating a
     method easy; the leaderboard populates over time. See §2.2 — this raises the bar on
     M1–M3 and turns M4 into ongoing work rather than a gate.
3. **PointMaze: re-cohort or drop?** Re-cohorting costs eval time; dropping leaves 2 environments.
   - A *(2026-08-01)*: **neither yet — defer.** Set the infrastructure up so it can be added,
     decide later. Registered but disabled; base checkpoints not staged. See §2.3.
4. **Does the deformable domain get added?** It is the cheapest route to genuine task diversity
   since the env code and data already exist.
   - A: Yes. *(Not started; `env/deformable_env/` is ported and the dataset is at
     `/mnt/richb/tw78/data/datasets/deformable.zip`.)*
5. **Public leaderboard hosting** — static generated page, or something with submissions? Affects
   M3's scope considerably.
   - A: Maybe as a github repository where new methods are introduced as pull requests?
   - Consistent with Q2's answer: a PR-based repo is exactly the "easy to propose a method"
     shape, and it means M3's real deliverable is a **submission format plus a CI check**, not
     a web app. What a contributor writes is one `TestTimeAdapter` subclass, one selection
     rule, and one config; what CI does is run the declared columns and refuse anything that
     reads a test cohort.
