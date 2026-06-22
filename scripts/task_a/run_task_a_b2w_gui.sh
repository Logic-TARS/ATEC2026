#!/usr/bin/env bash
set -eo pipefail

cd /home/1ctnltug/atec2026/ATEC2026
source /home/1ctnltug/atec2026/scripts/env/activate_atec2026_sim.sh

headless_args=(--headless)
if [[ "${ATEC_GUI:-0}" == "1" ]]; then
  headless_args=()
fi
fabric_args=()
if [[ "${ATEC_DISABLE_FABRIC:-0}" == "1" ]]; then
  fabric_args=(--disable_fabric)
fi

python scripts/play_atec_task.py \
  --task ATEC-TaskA-B2wPiper \
  "${headless_args[@]}" \
  --enable_cameras \
  "${fabric_args[@]}" \
  --num_envs 1 \
  --debug
