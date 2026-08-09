---
pretty_name: PAARBench PushObj Training Data
tags:
  - robotics
  - world-models
  - test-time-adaptation
---

# Optional training dataset

The PAARBench evaluation protocol does **not** require training trajectories: evaluation
reads fixed normalization metadata without opening the training dataset. Push settings
require the goal segments under `data/pushobj_eval/` — fetch them with
[`scripts/download_targets.py`](../scripts/download_targets.py) from
[`ThomasWalker1/paarbench-data`](https://huggingface.co/datasets/ThomasWalker1/paarbench-data)
and verify against `TARGETS.sha256`. Maze settings use the tracked corpora under
`data/maze_eval/` and offline Drive assets documented in [`MAZE.md`](MAZE.md).

The training release is provided for researchers who want to reproduce base-model or
adapter training, or develop methods that explicitly use offline trajectories:

```bash
.venv/bin/python scripts/download_data.py
```

The downloader fetches the immutable benchmark release from the
[`ThomasWalker1/paarbench-data`](https://huggingface.co/datasets/ThomasWalker1/paarbench-data)
dataset repository into `data/pushobj_multishape/` and verifies it against
`DATASET_MANIFEST.json`.

The current data release is pinned to Hub commit
`503481e4c0c025b3504b0503d976673d844758ce`; published benchmark work should retain
an immutable revision rather than following `main`.

## Contents

The release contains only the nonredundant splits used by the training pipeline:

| split | episodes | files | bytes |
|---|---:|---:|---:|
| `train` | 7,200 | 7,206 | 864,106,305 |
| `val` | 800 | 806 | 97,366,292 |

Each split contains `states.pth`, `velocities.pth`, `rel_actions.pth`,
`abs_actions.pth`, `seq_lengths.pkl`, `shapes.pkl`, and one MP4 observation stream per
episode under `obses/`. Redundant root-level copies and intermediate shards are not
part of the release.

The trajectories were generated in the PushObj simulator across the T, L, Z, and +
shapes. No license is asserted here beyond the rights of the repository owner; users
should review the dataset repository card and applicable upstream asset licenses
before redistribution.
