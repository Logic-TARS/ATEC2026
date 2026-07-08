#!/usr/bin/env bash
set -eo pipefail

source /home/1ctnltug/atec2026/scripts/env/activate-atec2026-sim.sh

flat_root="outputs/rsl_rl/unitree_b2_flat"
straight_root="outputs/rsl_rl/unitree_b2_rough_straight"
max_iterations="${ATEC_ROUGH_STRAIGHT_ITERS:-8000}"
num_envs="${ATEC_TRAIN_NUM_ENVS:-4096}"

flat_ckpt="$(
  find "${flat_root}" -maxdepth 2 -type f -name 'model_*.pt' \
    | sort -V \
    | tail -1
)"

if [[ -z "${flat_ckpt}" ]]; then
  echo "Could not find a flat checkpoint under ${flat_root}"
  exit 1
fi

echo "Bootstrapping rough-straight curriculum from:"
echo "  ${flat_ckpt}"
echo "  mode=actor-only, rough critic is trained from scratch"
echo

python tools/atec/rsl_rl/train.py \
  --task ATEC-Isaac-Velocity-Rough-Straight-Unitree-B2-v0 \
  --headless \
  --num_envs "${num_envs}" \
  --actor_checkpoint "$(realpath "${flat_ckpt}")" \
  --run_name from_flat \
  --max_iterations "${max_iterations}"
