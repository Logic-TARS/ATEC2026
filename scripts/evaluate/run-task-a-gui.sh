#!/usr/bin/env bash
set -eo pipefail

source /home/1ctnltug/atec2026/scripts/env/activate-atec2026-sim.sh

headless_args=(--headless)
if [[ "${ATEC_GUI:-0}" == "1" ]]; then
  headless_args=()
fi
fabric_args=()
if [[ "${ATEC_DISABLE_FABRIC:-0}" == "1" ]]; then
  fabric_args=(--disable_fabric)
fi

python tools/atec/play_task.py \
  --task ATEC-TaskA-B2wPiper \
  "${headless_args[@]}" \
  --enable_cameras \
  "${fabric_args[@]}" \
  --num_envs 1 \
  --debug
