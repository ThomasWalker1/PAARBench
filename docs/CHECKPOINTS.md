# Checkpoint and data artifacts

Large artifacts are not tracked in git. Stage them under the paths below before
running an evaluation.

## Base world models

| path | setting | SHA-256 prefix |
|---|---|---|
| `checkpoints/pushobj_shape_shift/checkpoints/model_latest.pth` | `pushobj`, `pushobj_shift` | `6a0a75a94eefa4ca` |
| `checkpoints/pusht_visual_shift/checkpoints/model_latest.pth` | `pusht` | `9098082f1d40bfa7` |
| `checkpoints/mediummaze_dynamics_shift/checkpoints/model_latest.pth` | maze settings | `df2caeb62f6fc8d6` |

Each base directory must also contain its training-time `hydra.yaml`. Evaluation reads
the model architecture, preprocessing configuration, and environment parameters from
it. Saved data paths are repository-relative; segment-goal evaluation substitutes
fixed normalization metadata and does not open the training dataset.

## Maze artifacts

Maze evaluation needs:

1. The `mediummaze_dynamics_shift` base (listed in `CHECKPOINTS.sha256`).
2. MediumMaze / DiverseMaze staging data hashed in `MAZE_ARTIFACTS.sha256`
   (upstream AdaJEPA Google Drive assets; see [`MAZE.md`](MAZE.md)).
3. Method adapters / PAD head under `checkpoints/maze_medium_adapters/` and
   `checkpoints/pad/maze_medium_inverse_dynamics.pth` (also in `CHECKPOINTS.sha256`).
4. Immutable episode corpora at `data/maze_eval/{maze_medium,maze_diverse}/seed_*.pkl`
   (tracked in git; also listed in `TARGETS.sha256`).

```bash
.venv/bin/python scripts/verify_maze_artifacts.py --manifest MAZE_ARTIFACTS.sha256
sha256sum --check --ignore-missing TARGETS.sha256   # maze_eval rows
```

## Adapter checkpoints

| path pattern | method | retained epochs | deployed epoch |
|---|---|---|---|
| `checkpoints/pushobj_adapters/hyper_r2_distill0/hyper_lora_epoch_N.pth` | HyperLoRA, PushObj | 1–5 | 2 |
| `checkpoints/pushobj_adapters/static_r2/hyper_lora_epoch_N.pth` | Static LoRA, PushObj | 1–5 | 2 |
| `checkpoints/pvs_adapters/hyper_r2_distill0/hyper_lora_epoch_N.pth` | HyperLoRA, PushT | 1–4 | 3 |
| `checkpoints/pvs_adapters/static_r2/hyper_lora_epoch_N.pth` | Static LoRA, PushT | 1–5 | 1 |
| `checkpoints/maze_medium_adapters/hyper_r2/hyper_lora_epoch_N.pth` | HyperLoRA, MediumMaze (+ OOD) | 1–5 | 2 |
| `checkpoints/maze_medium_adapters/static_r2/hyper_lora_epoch_N.pth` | Static LoRA, MediumMaze (+ OOD) | 1–5 | 2 |
| `checkpoints/pad/pushobj_inverse_dynamics.pth` | PAD, PushObj / PushObj Shift | — | single checkpoint |
| `checkpoints/pad/pusht_inverse_dynamics.pth` | PAD, PushT | — | single checkpoint |
| `checkpoints/pad/maze_medium_inverse_dynamics.pth` | PAD, MediumMaze (+ OOD) | — | single checkpoint |

Train MediumMaze HyperLoRA / Static LoRA adapters with
`scripts/train_maze_adapters.sh` or the commands in [`MAZE.md`](MAZE.md). The
`mediummaze_dynamics_shift` hydra config redacts the dataset root as `<path>`;
pass `--data-path data/point_maze_medium` when training.

All epoch candidates are required: the declared selection rules evaluate them on the
selection cohort before freezing the deployed epoch. The checkpoint payloads embed the
metadata that the adapters validate; separate training logs and metadata sidecars are
not runtime inputs.

Verify the full inventory with:

```bash
sha256sum --check CHECKPOINTS.sha256
```

## Distribution

The artifacts are hosted at `ThomasWalker1/paarbench-checkpoints` on Hugging Face Hub.
The remote directory structure is identical to the contents of local `checkpoints/`.
Use `scripts/download_checkpoints.py` to fetch base setting models, method-specific
weights, or both. The downloader filters the snapshot and verifies every requested file
against `CHECKPOINTS.sha256`.

The current artifact release (PushObj/PushT + MediumMaze base/adapters/PAD head) is
pinned to Hub commit `2e1152cc8fa26217bf826f2799a7d542ed283226`. Offline maze
trajectory assets remain Drive-staged and verified with `MAZE_ARTIFACTS.sha256`
([`MAZE.md`](MAZE.md)).

Pin releases by immutable revision; never use an unpinned `main` revision for a
published benchmark result. The downloader's default revision is updated whenever a
new artifact release is published.

Hugging Face Hub is the recommended operational host because it provides versioned
large-file storage, cached concurrent downloads, and selective snapshot downloads.
Publish the same release on Zenodo when a citable archival DOI is needed. Confirm that
the upstream licenses permit redistribution before making inherited weights public.

## Held-out evaluation records

Per-episode records used to regenerate continuous leaderboard columns are tracked in
git under `eval_outputs/<method>/<setting>/test*/**/episodes.jsonl`. See
[`EVAL_RECORDS.md`](EVAL_RECORDS.md). Checkpoints and PushObj goal files below remain
external; maze episode corpora under `data/maze_eval/` are tracked.

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

Maze episode corpora live at `data/maze_eval/{maze_medium,maze_diverse}/seed_{0,100,200,300}.pkl`
(also listed in `TARGETS.sha256`). They are small enough to track in git; regenerate
with `scripts/generate_maze_targets.py` after staging maze data if needed.

Publish updates with `scripts/upload_targets.py` (requires a Hub write token).

The optional training trajectories are distributed separately from the checkpoint release;
see [`DATASET.md`](DATASET.md) for PushObj/PushT and [`MAZE.md`](MAZE.md) for maze.

These artifacts originate from the model releases used to establish the checked-in
baseline records. Training new base models requires re-baselining all methods because
their paired metrics are defined against the exact frozen checkpoint.
