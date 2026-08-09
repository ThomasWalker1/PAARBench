# Static LoRA

Context-independent LoRA (+ LayerNorm delta) trained directly on the frozen
MediumMaze / PushObj / PushT base. Same offline trainer as HyperLoRA with
`--static-lora`.

## Offline training

```bash
.venv/bin/python methods/static_lora/train.py \
  --ckpt-dir checkpoints/mediummaze_dynamics_shift \
  --model-epoch latest \
  --data-path data/point_maze_medium \
  --output-dir checkpoints/maze_medium_adapters/static_r2 \
  --no-distill-adapter \
  --rank 2 \
  --epochs 5 \
  --device cuda:0
```

Produces `hyper_lora_epoch_{1..5}.pth` under the output directory (naming matches
`method.yaml` `maps_to` templates). Selection uses the benchmark's
`tunable.training_epoch` axis on the selection cohort.
