# Provenance

Every file in this repo is either **new benchmark code** or a **pinned copy** of code
from the predecessor project. This document records which, so that a number that fails
to reproduce can be traced to a change in semantics rather than guessed at.

## Source

| | |
|---|---|
| Predecessor repo | `~/HyperJEPA/` |
| Pinned commit | **`628dff7`** ("paper: coherence pass over the streamlining edits") |
| Extracted with | `git archive 628dff7 <paths> \| tar -x` |
| Date extracted | 2026-08-01 |

The predecessor is **read-only** and is actively maintained by a separate session, so
its `HEAD` moves. Everything here is pinned to `628dff7`; reproduction targets are taken
from that commit's `paper/floats/`, and per docs/PLAN.md §8, a disagreement with those
numbers means this port is wrong, not the number.

## Copied verbatim from `628dff7`

```
plan.py  utils.py  preprocessor.py  custom_resolvers.py  env.sh
planning/   base_planner.py cem.py gd.py mpc.py objectives.py evaluator.py
            adaptation.py hyper_adapter.py image_corruption.py
env/        pusht/ pointmaze/ deformable_env/ wall/ venv.py serial_vector_env.py
            minari_pointmaze.py
models/     visual_world_model.py vit.py dino.py lora.py hyper_lora.py
            hyper_context.py proprio.py vqvae.py dummy.py encoder/ decoder/
datasets/   traj_dset.py pusht_dset.py point_maze_dset.py wall_dset.py
            deformable_env_dset.py img_transforms.py
metrics/    image_metrics.py lpipsPyTorch/
distributed_fn/
conf/       action_encoder/ decoder/ encoder/ env/ planner/ predictor/ proprio_encoder/
```

Verify any of these against the source with:

```
diff <(git -C ~/HyperJEPA show 628dff7:planning/mpc.py) planning/mpc.py
```

## Modified after copying

| file | change | reason |
|---|---|---|
| `env/__init__.py` | `LOCAL_DEPS_DIR` default `/home/tw78/HyperJEPA/.local-deps` → this repo's `.local-deps` | the only hardcoded path into the predecessor; leaving it would make this repo depend on a directory it does not own |
| `conf/eval.yaml` | renamed copy of `628dff7:conf/plan_gd_mpc_local.yaml`, **byte-identical content** | canonical benchmark eval entry point; a method is selected with `+planner.adapter.method=<name>` |
| `planning/mpc.py` | the two hardcoded per-method branches replaced by one `TestTimeAdapter`; adaptation timing and peak memory decomposed from planner cost; writes `episodes.jsonl` | the whole point of the interface. Verified not to change the frozen path: 0.4900 per-shape-exactly, before and after |
| `planning/evaluator.py` | `_compute_rollout_metrics` returns per-episode arrays instead of only their means | the mean is the one summary §4 forbids on distance, and every discriminating metric needs the distribution |
| `plan.py` | `load_ckpt` / `load_model` moved to `paarbench/world_model.py` and imported back | so anything wanting just a model — a test, a CI check — does not have to run the whole Hydra pipeline or reimplement it |

### Moved out of the core

Method implementations do not belong in the benchmark. Both were relocated verbatim
apart from the protocol surface at the bottom of each file:

| from (`628dff7`) | to | change |
|---|---|---|
| `planning/adaptation.py` | `methods/adajepa/adapter.py` | internals unchanged; added the four protocol hooks and an episode reset (the original had none — see below) |
| `planning/hyper_adapter.py` | `methods/hyperjepa/adapter.py` | internals unchanged; added the four protocol hooks, keeping its two independent switches (what conditions the correction, how often it is recomputed) inside the method |

`models/hyper_lora.py` and `models/lora.py` stay in the core: LoRA installation,
generators and weight snapshot/restore are reusable model surgery that any method may
want, not one method's implementation.

**The reset is new work, not a port.** `AdaJEPAAdapter` had no episode-reset method at
all — it was built once per planning process and each process evaluated exactly one
batch, so the question never arose. Under the protocol an adapter may be reused, and an
optimizer trajectory surviving an episode boundary would contaminate the next episode
invisibly. `BaseWeightGuard` now snapshots and restores, and the optimizer is rebuilt so
Adam's moment estimates go with it.

## Deliberately not copied

Per docs/PLAN.md §7 "Do not take", plus the pruning it asks for:

- `eval_outputs/` — 14 directories in three incompatible layouts. `paarbench/schema.py`
  defines one output schema instead.
- `scripts/{episode_outcomes,rescore_distance,paired_indist_test}.py` — extracted at M0 as
  the reference for the metric definitions, then **removed** once `paarbench/metrics.py`
  implemented them against the new schema. They only ever read the predecessor's output
  layouts, which this repo does not produce, and leaving two metrics implementations in
  the tree invites using the wrong one. Recover from `628dff7` if the derivations are
  needed again.
- `scripts/ablation_grid.py`, `run_ablation_*`, `objrelfull_*`, `pvs_matrix_jobs.py`,
  and the other one-off drivers — HyperJEPA design-space sweeps; nothing generalizes.
- `checkpoints/pushobj_ablations*`, `*_adapters`, `pvs_adapters` — ablation artifacts.
- `paper/` — the predecessor's paper.
- `train.py`, `train_hyper_lora.py`, `conf/train.yaml` — not on the take list. **Will be
  needed for M5**, which resolves §2.4 by training distributable base models from scratch;
  re-extract from `628dff7` at that point rather than reconstructing.
- `conf/plan_cem.yaml`, `conf/plan_gd.yaml`, `conf/plan_gd_mpc.yaml`,
  `conf/plan_gd_mpc_ada.yaml`, `conf/plan_gd_mpc_minari_local.yaml` — Slurm-launcher and
  dead-planner variants.
- `conf/plan_gd_mpc_ada_local.yaml`, `conf/plan_gd_mpc_hyper_local.yaml` — kept out of the
  live config tree but preserved under `docs/reference/` as the specification the M1
  adapter port has to match.

## Binary artifacts (untracked; staged locally)

Inventory, hashes and per-artifact notes live in **`docs/CHECKPOINTS.md`**. In summary:
base world models for `pushobj` and `pusht`, adapter checkpoints for the amortized and
static arms on both, and the PushObj evaluation targets — all copied from
`~/HyperJEPA/checkpoints/` and `~/HyperJEPA/data/`, layout mirrored one-to-one. PointMaze
bases (5.9 GB) are deliberately not staged.

Training data is read from `/mnt/richb/tw78/data/hyperjepa_pushobj/pushobj_multishape`
and staged onto tmpfs by `paarbench/staging.py`.

**These are third-party released base models and are not redistributable.** By owner
decision (2026-08-01) the benchmark is developed against them as inherited artifacts and
the training code that reproduces them is added later; docs/PLAN.md §2.4 records the
deferral and the two constraints it carries.

## New code (no predecessor)

```
paarbench/adapter.py         the TestTimeAdapter protocol, NullAdapter, BaseWeightGuard
paarbench/methods.py         discovery, loading and validation of methods/<name>/
paarbench/selection.py       selection rules + a harness that cannot reach a test cohort
paarbench/runner.py          column execution, including episode-isolated fan-out
paarbench/settings.py        the §6 settings suite and its declared cohort split
paarbench/schema.py          the per-episode/per-replan record (§7's "one output schema")
paarbench/metrics.py         the §4 metrics, with bootstrap CIs
paarbench/staging.py         tmpfs dataset staging (§9)
paarbench/planner_hooks.py   the planner's single method-resolution point
paarbench/world_model.py     base-model loading, extracted from plan.py

scripts/evaluate.py          the submission path: select, freeze, test, record
scripts/eval_column.py       single-column debugging driver
scripts/validate_method.py   GPU-free method check; the CI gate
scripts/leaderboard.py       leaderboard generation from results/

methods/_template/           what a contributor copies
methods/static_lora/         the unconditioned control, written against the protocol
                             directly rather than ported

.github/workflows/ci.yml     validate + test on every pull request
pyproject.toml               uv project; cu121 torch pinned (§9)
```

`methods/adajepa/` and `methods/hyperjepa/` are ports; see "Moved out of the core" above.
