# Checkpoint and data artifacts

Large artifacts are not tracked in git. Stage them under the paths below before
running an evaluation.

## Base world models

| path | setting | SHA-256 prefix |
|---|---|---|
| `checkpoints/pushobj_shape_shift/checkpoints/model_latest.pth` | `pushobj`, `pushobj_shift` | `6a0a75a94eefa4ca` |
| `checkpoints/pusht_visual_shift/checkpoints/model_latest.pth` | `pusht` | `9098082f1d40bfa7` |

Each base directory must also contain its training-time `hydra.yaml`. Evaluation reads
the model architecture, preprocessing configuration, and environment parameters from
it. Saved data paths are repository-relative; segment-goal evaluation substitutes
fixed normalization metadata and does not open the training dataset.

## Adapter checkpoints

| path pattern | method | retained epochs | deployed epoch |
|---|---|---|---|
| `checkpoints/pushobj_adapters/hyper_r2_distill0/hyper_lora_epoch_N.pth` | HyperJEPA, PushObj | 1–5 | 2 |
| `checkpoints/pushobj_adapters/static_r2/hyper_lora_epoch_N.pth` | Static LoRA, PushObj | 1–5 | 2 |
| `checkpoints/pvs_adapters/hyper_r2_distill0/hyper_lora_epoch_N.pth` | HyperJEPA, PushT | 1–4 | 3 |
| `checkpoints/pvs_adapters/static_r2/hyper_lora_epoch_N.pth` | Static LoRA, PushT | 1–5 | 1 |
| `checkpoints/pad/pushobj_inverse_dynamics.pth` | PAD, PushObj / PushObj Shift | — | single checkpoint |
| `checkpoints/pad/pusht_inverse_dynamics.pth` | PAD, PushT | — | single checkpoint |

All epoch candidates are required: the declared selection rules evaluate them on the
selection cohort before freezing the deployed epoch. The checkpoint payloads embed the
metadata that the adapters validate; separate training logs and metadata sidecars are
not runtime inputs.

The complete 24-file inventory is approximately 1.1 GB. Verify it with:

```bash
sha256sum --check CHECKPOINTS.sha256
```

## Distribution

The artifacts are hosted at `ThomasWalker1/paarbench-checkpoints` on Hugging Face Hub.
The remote directory structure is identical to the contents of local `checkpoints/`.
Use `scripts/download_checkpoints.py` to fetch base setting models, method-specific
weights, or both. The downloader filters the snapshot and verifies every requested file
against `CHECKPOINTS.sha256`.

The current artifact release is pinned to Hub commit
`329f4215b0902d0eaa765a17290491ebe281a0b7`.

Pin releases by immutable revision; never use an unpinned `main` revision for a
published benchmark result. The downloader's default revision is updated whenever a
new artifact release is published.

Hugging Face Hub is the recommended operational host because it provides versioned
large-file storage, cached concurrent downloads, and selective snapshot downloads.
Publish the same release on Zenodo when a citable archival DOI is needed. Confirm that
the upstream licenses permit redistribution before making inherited weights public.

## Evaluation targets and training data

PushObj and PushT goal files are expected at:

```text
data/pushobj_eval/val_<shape>/plan_targets.pkl
```

They define the benchmark's fixed evaluation episodes (1000 pre-sampled segments per
shape, `goal_H=25`) and are **not** tracked in git. They originate from the
[AdaJEPA release](https://github.com/agentic-learning-ai-lab/adajepa) and are
distributed on Hugging Face Hub alongside the optional training dataset:

```bash
python scripts/download_targets.py
```

The downloader fetches `pushobj_eval/` from
[`ThomasWalker1/paarbench-data`](https://huggingface.co/datasets/ThomasWalker1/paarbench-data)
and verifies every file against `TARGETS.sha256`. If Hub is unavailable, place
`data/pushobj_eval.zip` locally and rerun with `--source local`, or use
`--source drive` as a last resort (Google Drive id
`1LNoPl-3XTlBFGSPg6DFqNXF3rv01LbUf`).

Publish updates with `scripts/upload_targets.py` (requires a Hub write token).

The optional training trajectories are distributed separately from the checkpoint release;
see [`DATASET.md`](DATASET.md).

These artifacts originate from the model releases used to establish the checked-in
baseline records. Training new base models requires re-baselining all methods because
their paired metrics are defined against the exact frozen checkpoint.
