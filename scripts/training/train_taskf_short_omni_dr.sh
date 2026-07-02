#!/usr/bin/env bash
set -eo pipefail

cd /home/1ctnltug/atec2026/ATEC2026
source /home/1ctnltug/atec2026/scripts/env/activate_atec2026_sim.sh

experiment_root="logs/rsl_rl/unitree_b2w_taskf_short_walk"
max_iterations="${ATEC_TASKF_DR_ITERS:-100}"
num_envs="${ATEC_TRAIN_NUM_ENVS:-256}"

ckpt="$(
  find "${experiment_root}" -maxdepth 2 -type f -path '*shortomni_v4_robust*/model_*.pt' \
    | sort -V \
    | tail -1
)"

if [[ -n "${ckpt}" ]]; then
  echo "Warm-starting ShortOmniDR from: ${ckpt}"
else
  echo "Could not find a robust checkpoint under ${experiment_root}"
  echo "Expected a path matching: ${experiment_root}/*shortomni_v4_robust*/model_*.pt"
  echo "Train the ShortOmniRobust stage first or rename/copy the intended checkpoint into a robust run directory."
  exit 1
fi

echo
echo "  task           = ATEC-Isaac-Velocity-ShortOmniDR-TaskF-Unitree-B2W-Piper-v0"
echo "  max_iterations = ${max_iterations}"
echo "  num_envs       = ${num_envs}"
echo "  log dir        = ${experiment_root}/*_from_robust/"
echo

python scripts/rsl_rl/train.py \
  --task ATEC-Isaac-Velocity-ShortOmniDR-TaskF-Unitree-B2W-Piper-v0 \
  --headless \
  --enable_cameras \
  --disable_fabric \
  --num_envs "${num_envs}" \
  --max_iterations "${max_iterations}" \
  --actor_checkpoint "$(realpath "${ckpt}")" \
  --run_name "from_robust"
