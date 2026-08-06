#!/usr/bin/env bash
# Evaluate PAD on pushobj_shift and pusht, then refresh leaderboard + website.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PY="${PY:-$ROOT/.venv/bin/python}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
PER_GPU="${PER_GPU:-4}"
EVAL=( "$PY" scripts/evaluate.py --gpus "$GPUS" )

for setting in pushobj_shift pusht; do
  echo "[method] pad $setting"
  "${EVAL[@]}" --per-gpu "$PER_GPU" pad --setting "$setting"
done

echo "[leaderboard] regenerating LEADERBOARD.md ..."
"$PY" scripts/leaderboard.py --out LEADERBOARD.md
echo "[website] regenerating static site ..."
"$PY" scripts/build_website.py
echo "done."
