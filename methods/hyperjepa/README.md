# HyperJEPA

Amortized hypernetwork that emits a rank-2 LoRA correction for the predictor's
last block from the episode's recent transitions. Corrections are regenerated
from the frozen base before every replan.

## Offline training

```bash
.venv/bin/python methods/hyperjepa/train.py \
  --ckpt-dir checkpoints/mediummaze_dynamics_shift \
  --model-epoch latest \
  --data-path data/point_maze_medium \
  --output-dir checkpoints/maze_medium_adapters/hyper_r2 \
  --no-distill-adapter \
  --rank 2 \
  --target-scope predlast_all \
  --context-mode transition_buffer \
  --context-aggregator transformer_query \
  --context-transitions 5 \
  --epochs 5 \
  --device cuda:0
```

Epoch files must be named `hyper_lora_epoch_{1..5}.pth` (see `method.yaml`).
PAARBench selects the reported epoch on the selection cohort via
`tunable.training_epoch`; do not use val-loss `best` for reported numbers.

Or: `scripts/train_maze_adapters.sh` (HyperJEPA + Static LoRA in parallel).
