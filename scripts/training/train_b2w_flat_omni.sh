#!/usr/bin/env bash
set -eo pipefail

cd /home/1ctnltug/atec2026/ATEC2026
source /home/1ctnltug/atec2026/scripts/env/activate_atec2026_sim.sh

experiment_root="logs/rsl_rl/unitree_b2w_rough_omni"
max_iterations="${ATEC_B2W_FLAT_OMNI_ITERS:-10000}"
num_envs="${ATEC_TRAIN_NUM_ENVS:-256}"

# --- Warm-start checkpoint resolution ---
# 1. Latest model checkpoint from rough-omni training
ckpt="$(
  find "${experiment_root}" -maxdepth 2 -type f -name 'model_*.pt' \
    | sort -V \
    | tail -1
)"

# 2. Fall back to demo/policy_taskd_omni.pt
if [[ -z "${ckpt}" ]] && [[ -f "demo/policy_taskd_omni.pt" ]]; then
  ckpt="demo/policy_taskd_omni.pt"
fi

actor_checkpoint_arg=()
if [[ -n "${ckpt}" ]]; then
  echo "Warm-starting B2W+Piper flat-omni curriculum from:"
  echo "  ${ckpt}"
  actor_checkpoint_arg=(--actor_checkpoint "$(realpath "${ckpt}")")
else
  echo "WARNING: No warm-start checkpoint found under ${experiment_root}"
  echo "         and demo/policy_taskd_omni.pt is not available."
  echo "         Training from scratch."
fi

echo
echo "  max_iterations = ${max_iterations}"
echo "  num_envs       = ${num_envs}"
echo "  log dir        = logs/rsl_rl/unitree_b2w_flat_omni"
echo "  stability note = default num_envs is 256 for this 24GB GPU"
echo "                   scale manually only after stable: ATEC_TRAIN_NUM_ENVS=512, then 768, then 1024"
echo

python scripts/rsl_rl/train.py \
  --task ATEC-Isaac-Velocity-Flat-Omni-B2W-Piper-v0 \
  --headless \
  --enable_cameras \
  --disable_fabric \
  --num_envs "${num_envs}" \
  --max_iterations "${max_iterations}" \
  "${actor_checkpoint_arg[@]}" \
  --run_name "from_rough_omni"
