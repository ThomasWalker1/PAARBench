# Maze settings

All maze settings evaluate the `mediummaze_dynamics_shift` world-model base.

| nominal setting | inherited shifts | base |
|---|---|---|
| `maze_medium` | `maze_medium_low_density`, `maze_medium_high_damping`, `maze_diverse` | `mediummaze_dynamics_shift` |

`maze_medium` selects on seed 0; all reports use seed 100, 200, and 300 with 50
episodes each. The dynamics shifts rebuild the identical maze with
`density_scale=0.2` or `damping_scale=20`. `maze_diverse` evaluates the MediumMaze
base on held-out DiverseMaze validation layouts and inherits MediumMaze's frozen
parameters.

## Install and stage

The point-maze environment needs the legacy Gym MuJoCo interface:

```bash
# User-local (no sudo): stages GLEW, patchelf, and OSMesa in .mujoco-build/.
scripts/setup_mujoco_runtime.sh
uv sync --extra dev --extra maze
```

`mujoco-py==2.1.2.14` requires a locally installed MuJoCo 2.1 runtime at
`~/.mujoco/mujoco210`. The column runner automatically detects the user-local
`.mujoco-build/` runtime and uses CPU OSMesa for environment rendering while model and
planner computation remains on the selected CUDA GPU.

### Base checkpoint and offline data

1. Stage `checkpoints/mediummaze_dynamics_shift/` (`hydra.yaml` +
   `checkpoints/model_latest.pth`) from
   [AdaJEPA](https://github.com/agentic-learning-ai-lab/adajepa) (or from the
   PAARBench Hub release once the maze revision is published — see
   [`CHECKPOINTS.md`](CHECKPOINTS.md)).
2. Stage MediumMaze trajectories at `data/point_maze_medium/` and the held-out
   DiverseMaze layout corpus at `data/diverse_maze/val/`.
3. Lock and verify the Drive-sourced assets (they have no immutable upstream revision):

```bash
.venv/bin/python scripts/verify_maze_artifacts.py \
  --manifest MAZE_ARTIFACTS.sha256
```

To recreate the manifest after restaging:

```bash
.venv/bin/python scripts/verify_maze_artifacts.py \
  --write-manifest MAZE_ARTIFACTS.sha256
```

Adapter checkpoints and the PAD inverse-dynamics head are listed in
`CHECKPOINTS.sha256`. Train them locally (below) or download them once the Hub
revision includes the `mediummaze_dynamics_shift/`, `maze_medium_adapters/`, and
`pad/maze_medium_inverse_dynamics.pth` paths.

## Freeze evaluation episodes

AdaJEPA samples maze goals at runtime. PAARBench materializes deterministic episode
corpora under `data/maze_eval/`. Those files are tracked in git; regenerate only if
you restage the underlying datasets:

```bash
.venv/bin/python scripts/generate_maze_targets.py --setting maze_medium
.venv/bin/python scripts/generate_maze_targets.py --setting maze_diverse
sha256sum --check --ignore-missing TARGETS.sha256
```

The medium dynamics shifts reuse `maze_medium` target files. A complete corpus is
required before any worker runs; `scripts/evaluate.py` names every missing file.

## Train HyperLoRA / Static LoRA / PAD

```bash
# Parallel single-GPU trainers (HyperLoRA on GPU 0, Static LoRA on GPU 1):
bash scripts/train_maze_adapters.sh

# Or explicitly:
.venv/bin/python methods/hyperlora/train.py \
  --ckpt-dir checkpoints/mediummaze_dynamics_shift \
  --data-path data/point_maze_medium \
  --output-dir checkpoints/maze_medium_adapters/hyper_r2 \
  --no-distill-adapter --rank 2 --target-scope predlast_all \
  --context-mode transition_buffer --context-aggregator transformer_query \
  --context-transitions 5 --epochs 5 --device cuda:0

.venv/bin/python methods/static_lora/train.py \
  --ckpt-dir checkpoints/mediummaze_dynamics_shift \
  --data-path data/point_maze_medium \
  --output-dir checkpoints/maze_medium_adapters/static_r2 \
  --no-distill-adapter --rank 2 --epochs 5 --device cuda:0

.venv/bin/python methods/pad/train_inverse_head.py \
  --dataset-kind point_maze --data data/point_maze_medium \
  --base checkpoints/mediummaze_dynamics_shift \
  --output checkpoints/pad/maze_medium_inverse_dynamics.pth
```

Expected HyperLoRA / Static LoRA outputs: `hyper_lora_epoch_{1..5}.pth` under each
adapter directory (verify with `sha256sum --check CHECKPOINTS.sha256`). Epoch
selection stays on the selection cohort (`tunable.training_epoch`); do not report
val-loss `best`. The staged `data/point_maze_medium` zip uses per-episode uint8
tensors; the loader falls back from the hydra config's per-frame layout
automatically.

Published MediumMaze selection chose **epoch 2** for both HyperLoRA and Static LoRA.

## Evaluate

Run nominal selections first. The shifts then read their frozen result record and
cannot start until it exists. Published tables pair every method against batched
Frozen (`--frozen`); optional `--frozen --isolated` runs are appendix diagnostics only.

```bash
.venv/bin/python scripts/evaluate.py --frozen --setting maze_medium --gpus 0,1,2,3,4,5,6,7

.venv/bin/python scripts/evaluate.py adajepa --setting maze_medium \
  --gpus 0,1,2,3,4,5,6,7 --per-gpu 1
.venv/bin/python scripts/evaluate.py hyperlora --setting maze_medium \
  --gpus 0,1,2,3,4,5,6,7 --per-gpu 1
.venv/bin/python scripts/evaluate.py static_lora --setting maze_medium \
  --gpus 0,1,2,3,4,5,6,7 --per-gpu 1
.venv/bin/python scripts/evaluate.py pad --setting maze_medium \
  --gpus 0,1,2,3,4,5,6,7 --per-gpu 1

# Inherited OOD (after the nominal result record exists):
for setting in maze_medium_low_density maze_medium_high_damping maze_diverse; do
  .venv/bin/python scripts/evaluate.py --frozen --setting "$setting" --gpus 0,1,2,3,4,5,6,7
  for method in adajepa hyperlora static_lora pad; do
    .venv/bin/python scripts/evaluate.py "$method" --setting "$setting" \
      --gpus 0,1,2,3,4,5,6,7 --per-gpu 1
  done
done
```

Compact records land in `results/<method>/<setting>.json`. Held-out
`eval_outputs/<method>/<setting>/test*/**/episodes.jsonl` regenerate continuous
leaderboard columns (see [`EVAL_RECORDS.md`](EVAL_RECORDS.md)).

```bash
.venv/bin/python scripts/leaderboard.py --out LEADERBOARD.md
.venv/bin/python scripts/build_website.py
```
