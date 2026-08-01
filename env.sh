#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_DEPS_DIR="${LOCAL_DEPS_DIR:-${ROOT_DIR}/.local-deps}"

export DATASET_DIR="${DATASET_DIR:-/mnt/richb/tw78/data/datasets}"
export MUJOCO_PY_MUJOCO_PATH="${MUJOCO_PY_MUJOCO_PATH:-/home/tw78/.mujoco/mujoco210}"
export WANDB_MODE="${WANDB_MODE:-offline}"

if [ -d "${LOCAL_DEPS_DIR}/usr/bin" ]; then
  export PATH="${LOCAL_DEPS_DIR}/usr/bin:${PATH}"
fi

if [ -d "${LOCAL_DEPS_DIR}/usr/include" ]; then
  export CPATH="${LOCAL_DEPS_DIR}/usr/include:${CPATH:-}"
fi

LOCAL_LIB_PATHS=()
for libdir in \
  "${LOCAL_DEPS_DIR}/usr/lib/x86_64-linux-gnu" \
  "${LOCAL_DEPS_DIR}/usr/lib64" \
  "${LOCAL_DEPS_DIR}/usr/lib"
do
  if [ -d "${libdir}" ]; then
    LOCAL_LIB_PATHS+=("${libdir}")
  fi
done

if [ ${#LOCAL_LIB_PATHS[@]} -gt 0 ]; then
  export LIBRARY_PATH="$(IFS=:; echo "${LOCAL_LIB_PATHS[*]}"):${LIBRARY_PATH:-}"
  export LD_LIBRARY_PATH="$(IFS=:; echo "${LOCAL_LIB_PATHS[*]}"):${MUJOCO_PY_MUJOCO_PATH}/bin:/usr/lib/nvidia:${LD_LIBRARY_PATH:-}"
else
  export LD_LIBRARY_PATH="${MUJOCO_PY_MUJOCO_PATH}/bin:/usr/lib/nvidia:${LD_LIBRARY_PATH:-}"
fi
