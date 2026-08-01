# Checkpoints

`checkpoints/` is untracked (too large for git). This file is its manifest: what is
staged, where it came from, and what each artifact is for.

**These are inherited artifacts, not reproducible ones.** The training code that produced
them is not in this repo yet — see "Reproducing these" below. That is a deliberate,
recorded deferral: the benchmark can be developed against them as they stand, and the
training scripts land later.

Source for everything here: `~/HyperJEPA/checkpoints/` (read-only, pinned `628dff7`).
Layout deliberately mirrors the predecessor's so the mapping stays one-to-one.

## Base world models

The frozen model a method adapts. Every metric in §4 is measured relative to running one
of these with no adaptation at all.

| path | setting | sha256 (first 16) | notes |
|---|---|---|---|
| `pushobj_shape_shift/checkpoints/model_latest.pth` | `pushobj`, `pushobj_shift` | `6a0a75a94eefa4ca` | primary. 4 train shapes {T,L,Z,+}; held-out {I, small_tee, square} |
| `pusht_visual_shift/checkpoints/model_latest.pth` | `pusht` | `9098082f1d40bfa7` | ships an **unresolved** `env.dataset.data_path`; every column must set `dataset_data_path` or the dataset call dies with a bare `FileNotFoundError` naming only `env.dataset` |

Each base directory also carries the `hydra.yaml` it was trained under. `plan.py` reads
the model, dataset and environment configuration from there, so it is load-bearing, not
documentation.

### Not staged

`pointmaze/` — 5.9 GB across three variants
(`scratch_resnet_global_cos1e-1_{iid,minari_medium,minari_medium_e10}_seed0`). Deferred:
the `pointmaze` setting is registered but disabled pending §2.3, and which variant to
take depends on how that is resolved. Also needs the optional `pointmaze` dependency
extra (mujoco-py, d4rl).

## Adapter checkpoints

Trained adaptation modules — the *amortized* arm's learned weights, and the unconditioned
static control. The online gradient arm has no checkpoint; it starts from the base every
episode and its "weights" are its hyperparameters.

| path | arm | notes |
|---|---|---|
| `pushobj_adapters/hyper_r2_distill0/hyper_lora_epoch_4.pth` | HyperJEPA on `pushobj` | **this is the 0.610 checkpoint** — rank-2, epoch 4 (`~/HyperJEPA/RESULTS.md` §15, lines 542 and 831). `sha256 bd9fdb02f238b904` |
| `pushobj_adapters/static_r2/` | unconditioned static correction on `pushobj` | epoch checkpoints; the control that asks whether conditioning buys anything |
| `pvs_adapters/hyper_r2_distill0/` | HyperJEPA on `pusht` | |
| `pvs_adapters/static_r2/baked_ep*/` | static control on `pusht` | **baked** full weights, evaluated through the ordinary planner config so the control pays no adapter-wrapper overhead (that overhead is real: +0.80 s/replan) |

### `metadata.json` is load-bearing

Each adapter directory carries one, and `planning/hyper_adapter.py` **validates the
planning-time config against it** — `rank`, `context_feature_kind`, `context_aggregator`,
`context_transitions`, `adapter_gate`, `freeze_lora_A`. A mismatch raises rather than
silently mis-loading, which is the correct behaviour and worth preserving through the M1
port. For `hyper_r2_distill0` the values are:

```
rank: 2                                 target_scope: predlast_all
context_mode: transition_buffer         context_feature_kind: residual_action
context_aggregator: transformer_query   context_transitions: 5
adapter_gate: false                     static_lora: false
```

## Selecting an epoch

Do **not** pick an adapter epoch on a test cohort. Epoch selection is a
hyperparameter-selection decision and belongs on the selection cohort, priced in the
submission's declared selection cost (§3). The 0.610 figure comes from epoch 4 selected
that way; `~/HyperJEPA/RESULTS.md` line 873 notes the selection cohort gives 0.605 for the
same checkpoint against 0.610 on test, so the split is behaving as intended.

## Reproducing these

Not yet possible from this repo. Deliberately deferred — the benchmark is developed
against these checkpoints as inherited artifacts, and the training code lands later.

When it does, it means re-extracting `train.py`, `train_hyper_lora.py` and
`conf/train.yaml` from `628dff7` (they were left off the §7 take list precisely because
this was out of scope at the time), plus the training data at
`/mnt/richb/tw78/data/hyperjepa_pushobj/`.

Two consequences worth holding onto:

1. **The bases are third-party released models.** Until they are retrained on data the
   project owns, a third party cannot reproduce a submission's numbers — which caps how
   far the benchmark travels, whatever the harness does. This was §2.4.
2. **Retraining invalidates every reproduction target.** The M1/M2 acceptance numbers
   (frozen 0.485, HyperJEPA 0.610, tuned online 0.678, and the §4 distance spans) are all
   defined against *these* weights. Retraining means re-baselining the whole suite, so it
   has to come after the port is validated, not alongside it.
