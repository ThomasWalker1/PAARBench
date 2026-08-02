# Checkpoints

`checkpoints/` is untracked (too large for git). This file is its manifest: what is
staged, where it came from, and what each artifact is for.

The inherited base and HyperJEPA artifacts are not reproducible here yet — see
"Reproducing these" below. PAD is the exception: its small inverse-dynamics head is
generated locally by the checked-in method script, documented below.

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
| `pushobj_adapters/hyper_r2_distill0/hyper_lora_epoch_2.pth` | HyperJEPA on `pushobj` | **this is the 0.610 checkpoint** — rank-2, **epoch 2**, confirmed by reproducing 0.6100 on the test cohorts |
| `pushobj_adapters/static_r2/` | unconditioned static correction on `pushobj` | epoch checkpoints; the control that asks whether conditioning buys anything |
| `pvs_adapters/hyper_r2_distill0/` | HyperJEPA on `pusht` | |
| `pvs_adapters/static_r2/baked_ep*/` | static control on `pusht` | **baked** full weights, evaluated through the ordinary planner config so the control pays no adapter-wrapper overhead (that overhead is real: +0.80 s/replan) |
| `pad/pushobj_inverse_dynamics.pth` | PAD on `pushobj` | 0.79 MB, SHA-256 `252fae5fb2727534`; pretrained for 12 epochs on 2,048 PushObj training transition pairs with frozen base latents. Produced by `methods/pad/train_inverse_head.py`, not inherited from `~/HyperJEPA/`. |

### PAD inverse-dynamics head

PAD's original auxiliary head is jointly pretrained with its policy. This benchmark
uses MPC rather than that policy, so `methods/pad/train_inverse_head.py` trains the
closest fair substitute: an MLP that predicts the normalized executed action from
two consecutive *frozen* PushObj world-model latents. The target is the planner's
five-frame, 10-dimensional normalized action chunk, matching the transition it
actually executes. It reads only
`/dev/shm/tw78/data/pushobj_multishape` (or an explicit `--data` path), never a
selection or test cohort. The shipped run uses seed 0, 2,048 sampled pairs, batch size
64 and 12 Adam epochs; its final training MSE is 0.249628.

The checkpoint stores its latent/action dimensions and the adapter rejects a mismatch.
During deployment the encoder and this head are updated together, then both are reset
at each episode boundary through `BaseWeightGuard`.

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
submission's declared selection cost (§3). On the selection cohort (seed 300),
`hyper_r2_distill0` scores 0.560 / **0.605** / 0.595 / 0.570 / 0.570 across epochs 1–5, so
epoch 2 is selected, and epoch 2 is what scores 0.610 on the test cohorts.

> **This is worth reading before you trust an epoch number.** The first port of this
> method used **epoch 4** and scored 0.588 on test instead of 0.610 — a plausible-looking
> number, no error, nothing to notice. Epoch 4 had been misread out of a three-column
> table in the predecessor's results where 0.610 appears in row 4 of the *distill1*
> column, not distill0's. The fix was not a better reading; it was running
> `methods/hyperjepa/selection.py:EpochSelection`, which evaluates all five epochs on the
> selection cohort and picks epoch 2 for five columns of cost. A declared selection rule
> is not bureaucracy — it is the thing that stops a transcription error from becoming a
> published number.

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
