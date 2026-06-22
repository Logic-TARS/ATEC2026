#!/usr/bin/env bash
set -eo pipefail

cd /home/1ctnltug/atec2026/ATEC2026
source /home/1ctnltug/atec2026/scripts/env/activate_atec2026_sim.sh

straight_root="logs/rsl_rl/unitree_b2_rough_straight"
omni_root="logs/rsl_rl/unitree_b2w_rough_omni"
max_iterations="${ATEC_B2W_OMNI_ITERS:-12000}"
num_envs="${ATEC_TRAIN_NUM_ENVS:-4096}"

# Find the latest rough-straight checkpoint to warm-start from.
straight_ckpt="$(
  find "${straight_root}" -maxdepth 2 -type f -name 'model_*.pt' \
    | sort -V \
    | tail -1
)"

if [[ -z "${straight_ckpt}" ]]; then
  echo "Could not find a rough-straight checkpoint under ${straight_root}"
  echo "Train a rough-straight policy first: ./scripts/training/train_b2_rough_straight_from_flat.sh"
  exit 1
fi

echo "Bootstrapping B2W+Piper rough-omni curriculum from:"
echo "  ${straight_ckpt}"
echo
echo "  max_iterations = ${max_iterations}"
echo "  num_envs       = ${num_envs}"
echo "  log dir        = ${omni_root}"
echo

python scripts/rsl_rl/train.py \
  --task ATEC-Isaac-Velocity-Rough-Omni-B2W-Piper-v0 \
  --headless \
  --num_envs "${num_envs}" \
  --actor_checkpoint "$(realpath "${straight_ckpt}")" \
  --run_name from_straight \
  --max_iterations "${max_iterations}"
