# PlanActAdaptRepeatBench — a benchmark for test-time adaptation of latent world models

**Status:** greenfield. Nothing implemented yet. This document is the whole project.
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

### 2.2 Three methods is not a leaderboard  *(largest risk)*

Existing arms: frozen, online gradient TTA, an unconditioned (static) correction, HyperJEPA.
A credible benchmark needs **6–8**. Candidates, best first:

- **PAD** (Hansen et al., policy adaptation during deployment) — closest fit, already cited in
  the HyperJEPA bibliography.
- **T3A** — training-free, cheap, a good "free lunch" reference point.
- **CoTTA**-style continual TTA with restoration — directly probes the accumulation pathology.
- **MAML / learned initialization** — keeps the inner loop, learns where it starts.
- **Entropy-minimization family (TENT, SHOT, EATA)** — awkward here: they need a classifier head
  and there isn't one. *That awkwardness is itself a reportable finding* — the dominant TTA
  family does not transfer to latent world models. Say so rather than forcing it.

**If you cannot land 4+ new methods, this is not a benchmark paper.** Revisit with the owner
before committing weeks.

### 2.3 The setting suite is thin, and one setting is broken as configured

- PushObj and PushT are both pushing tasks on released DINO-WM-style checkpoints — one task
  family, two checkpoints.
- **PointMaze is unusable as configured**: frozen sits at 0.880 on its selection cohort, which
  compresses everything (distance span 1.4× against PushObj's 12.7×), and n=50 gives SE ≈ 0.045.
  Either re-cohort to the harder distribution (its *test* cohort is frozen 0.733) or drop it.
- Held-out-shape PushObj is a distribution-shift *condition* on an existing base, not a new
  environment. It is valuable (the span *widens* to 14.7× there) but should not be counted as a
  fourth environment.

Realistically: **2 environments + 1 shift condition**. Adding a genuinely different domain
(deformable manipulation is already in `~/HyperJEPA/env/deformable_env/`, and the dataset is at
`/mnt/richb/tw78/data/datasets/deformable.zip`) would materially strengthen it.

### 2.4 Base checkpoints are not distributable as-is  *(hard blocker for adoption)*

`~/HyperJEPA/` uses released third-party base world models and its own Limitations section
concedes the phase-two pool is "a contact-rich reconstruction rather than the base's own
training data." A benchmark must ship pinned, downloadable base models or submissions are not
comparable. **Resolve this before any public release.** Options: retrain bases from scratch on
data you own and can distribute.

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

**M0 — scaffold (0.5 d).** `git init`; uv project; copy §7 "take" list; `.venv` per §9; smoke-run
one frozen evaluation column.
*Accept:* a frozen PushObj column runs end to end and reports success 0.490 ± noise on seed 300.

**M1 — the interface (1–2 d).** Define `TestTimeAdapter`. Port both existing methods. Collapse
mpc.py's two branches. Add the reset test from §5.
*Accept:* frozen 0.485, HyperJEPA 0.610, tuned online 0.678 reproduced on PushObj test seeds
(n=600); base weights bit-identical across episode boundaries.

**M2 — output schema + metrics (1–2 d).** One canonical per-episode/per-replan output. Implement
all seven §4 metrics against it, with CIs.
*Accept:* every distance number in `~/HyperJEPA/paper/floats/safety_span.tex` reproduced from the
new schema (12.7× / 8.8× / 14.7× / 1.4× spans; catastrophe 18–178).

**M3 — harness + protocol (2–3 d).** Selection/test cohort separation enforced in code; a
submission is a method + selection rule; selection cost recorded; leaderboard generation.
*Accept:* a deliberately cheating submission that peeks at test seeds is *rejected by the
harness*, not by review.

**M4 — baselines (the long pole, 1–3 weeks).** Port 4+ methods from §2.2.
*Accept:* ≥6 total entries, each with a declared selection rule and reported selection cost.

**M5 — settings (parallel with M4).** Resolve PointMaze; resolve base-model distribution (§2.4);
consider adding the deformable domain.
*Accept:* every base model pinned by hash and downloadable by a third party.

**Then, and only then:** decide venue. NeurIPS Datasets & Benchmarks is the natural track, and it
is a *different deadline and bar* from the predecessor's target.

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

1. **Is §2.4 solvable?** If base world models cannot be distributed, the benchmark cannot be
   adopted externally and should be scoped as an internal evaluation suite instead. This is the
   single most important question and it is not a technical one.
   - A: Train own models from scratch.
2. **Appetite for M4?** 4+ method ports is the difference between a benchmark and a protocol
   section. If the answer is no, the better move is to strengthen §3 inside the predecessor paper.
   - A: Ensure HyperJEPA and AdaJEPA are implemented, then we can think about other protocols.
3. **PointMaze: re-cohort or drop?** Re-cohorting costs eval time; dropping leaves 2 environments.
4. **Does the deformable domain get added?** It is the cheapest route to genuine task diversity
   since the env code and data already exist.
   - A: Yes.
5. **Public leaderboard hosting** — static generated page, or something with submissions? Affects
   M3's scope considerably.
   - A: Maybe as a github repository where new methods are introduced as pull requests?
