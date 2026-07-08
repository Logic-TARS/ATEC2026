#!/usr/bin/env bash
set -eo pipefail

source /home/1ctnltug/atec2026/scripts/env/activate-atec2026-sim.sh

MAX_ITERS="${ATEC_TASKD_ITERS:-7000}"
NUM_ENVS="${ATEC_TRAIN_NUM_ENVS:-1024}"

FLAT_EXP_ROOT="outputs/rsl_rl/unitree_b2w_taskd_flat_pretrain"
SOURCE_CKPT="$(
  find "${FLAT_EXP_ROOT}" -maxdepth 2 -type f -name 'model_*.pt' \
    | sort -V \
    | tail -1
)"
if [[ -z "$SOURCE_CKPT" || ! -f "$SOURCE_CKPT" ]]; then
  echo "ERROR: Flat pre-training checkpoint not found in ${FLAT_EXP_ROOT}/"
  echo "  Train the flat pre-train first:"
  echo "    ./scripts/train/train-env.sh --task ATEC-Isaac-Velocity-Flat-TaskF-Unitree-B2W-Piper-v0 --headless --enable_cameras --disable_fabric --num_envs 64 --max_iterations 1000"
  exit 1
fi

echo "Flat pre-train checkpoint: $SOURCE_CKPT"
echo "Task D official fine-tuning: iters=$MAX_ITERS envs=$NUM_ENVS"

python tools/atec/rsl_rl/train.py \
  --task "ATEC-Isaac-TaskD-FixedArm-B2W-Official-v0" \
  --headless \
  --enable_cameras \
  --disable_fabric \
  --num_envs "$NUM_ENVS" \
  --max_iterations "$MAX_ITERS" \
  --actor_checkpoint "$SOURCE_CKPT" \
  --run_name "from_flat_taskf"
