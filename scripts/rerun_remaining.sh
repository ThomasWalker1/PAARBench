#!/usr/bin/env bash
# Continue the benchmark rerun from adajepa onward (skips completed methods).
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

for setting in pushobj pusht; do
  echo "[method] adajepa $setting"
  "${EVAL[@]}" --per-gpu "$PER_GPU" adajepa --setting "$setting"
done

echo "[method] adajepa pushobj_shift (inherits selection)"
"${EVAL[@]}" --per-gpu "$PER_GPU" adajepa --setting pushobj_shift

for setting in pushobj pusht pushobj_shift; do
  echo "[method] pad $setting (authored params, no tunable axes)"
  "${EVAL[@]}" --per-gpu "$PER_GPU" pad --setting "$setting"
done

echo "[leaderboard] regenerating LEADERBOARD.md ..."
"$PY" scripts/leaderboard.py --out LEADERBOARD.md
echo "[website] regenerating static site ..."
"$PY" scripts/build_website.py
echo "done."
