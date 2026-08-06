#!/usr/bin/env bash
# Re-run the full PAARBench protocol under the standardized selection rules.
# Requires checkpoints/, data/pushobj_eval/, and 8 GPUs by default.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PY="${PY:-$ROOT/.venv/bin/python}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
PER_GPU="${PER_GPU:-4}"
EVAL=( "$PY" scripts/evaluate.py --gpus "$GPUS" )

echo "[check] verifying staged artifacts ..."
"$PY" - <<'PY'
from paarbench import settings as s
for setting_id in ("pushobj", "pushobj_shift", "pusht"):
    s.get(setting_id).require_targets()
print("goal files ok")
PY

echo "[frozen] baseline columns ..."
"${EVAL[@]}" --frozen --setting pushobj
"${EVAL[@]}" --frozen --setting pushobj_shift
"${EVAL[@]}" --frozen --setting pusht

for setting in pushobj pusht; do
  echo "[method] hyperjepa $setting"
  "${EVAL[@]}" hyperjepa --setting "$setting"
  echo "[method] static_lora $setting"
  "${EVAL[@]}" static_lora --setting "$setting"
done

echo "[method] hyperjepa pushobj_shift (inherits selection)"
"${EVAL[@]}" hyperjepa --setting pushobj_shift
echo "[method] static_lora pushobj_shift (inherits selection)"
"${EVAL[@]}" static_lora --setting pushobj_shift

for setting in pushobj pusht; do
  echo "[method] restore_tta $setting"
  "${EVAL[@]}" --per-gpu "$PER_GPU" restore_tta --setting "$setting"
  echo "[method] adajepa $setting"
  "${EVAL[@]}" --per-gpu "$PER_GPU" adajepa --setting "$setting"
done

echo "[method] restore_tta pushobj_shift (inherits selection)"
"${EVAL[@]}" --per-gpu "$PER_GPU" restore_tta --setting pushobj_shift
echo "[method] adajepa pushobj_shift (inherits selection)"
"${EVAL[@]}" --per-gpu "$PER_GPU" adajepa --setting pushobj_shift

echo "[method] pad pushobj (authored params, no tunable axes)"
"${EVAL[@]}" --per-gpu "$PER_GPU" pad --setting pushobj

echo "[leaderboard] regenerating LEADERBOARD.md ..."
"$PY" scripts/leaderboard.py --out LEADERBOARD.md
echo "done."
