#!/usr/bin/env bash
# Train MediumMaze HyperJEPA + Static LoRA adapters (single-GPU each, parallel).
# Usage: HYPER_GPU=0 STATIC_GPU=1 bash scripts/train_maze_adapters.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="${PY:-$ROOT/.venv/bin/python}"
HYPER_GPU="${HYPER_GPU:-0}"
STATIC_GPU="${STATIC_GPU:-1}"
EPOCHS="${EPOCHS:-5}"
RANK="${RANK:-2}"
CKPT="${CKPT:-checkpoints/mediummaze_dynamics_shift}"
DATA="${DATA:-data/point_maze_medium}"
COMMON=(
  --ckpt-dir "$CKPT"
  --model-epoch latest
  --data-path "$DATA"
  --no-distill-adapter
  --rank "$RANK"
  --epochs "$EPOCHS"
  --device cuda:0
  --num-workers 8
)

HYPER_OUT="${HYPER_OUT:-checkpoints/maze_medium_adapters/hyper_r${RANK}}"
STATIC_OUT="${STATIC_OUT:-checkpoints/maze_medium_adapters/static_r${RANK}}"
mkdir -p "$HYPER_OUT" "$STATIC_OUT"

echo "=== HyperJEPA gpu=$HYPER_GPU -> $HYPER_OUT ==="
CUDA_VISIBLE_DEVICES="$HYPER_GPU" PYTHONUNBUFFERED=1 "$PY" methods/hyperjepa/train.py \
  "${COMMON[@]}" \
  --target-scope predlast_all \
  --context-mode transition_buffer \
  --context-aggregator transformer_query \
  --context-transitions 5 \
  --output-dir "$HYPER_OUT" \
  >"$HYPER_OUT/train.log" 2>&1 &
HYPER_PID=$!

echo "=== Static LoRA gpu=$STATIC_GPU -> $STATIC_OUT ==="
CUDA_VISIBLE_DEVICES="$STATIC_GPU" PYTHONUNBUFFERED=1 "$PY" methods/static_lora/train.py \
  "${COMMON[@]}" \
  --output-dir "$STATIC_OUT" \
  >"$STATIC_OUT/train.log" 2>&1 &
STATIC_PID=$!

echo "PIDs: hyper=$HYPER_PID static=$STATIC_PID"
wait "$HYPER_PID"
wait "$STATIC_PID"
echo "=== both trainers finished ==="
ls -la "$HYPER_OUT"/hyper_lora_epoch_*.pth "$STATIC_OUT"/hyper_lora_epoch_*.pth
