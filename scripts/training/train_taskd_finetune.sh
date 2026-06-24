#!/usr/bin/env bash
set -eo pipefail

cd /home/1ctnltug/atec2026/ATEC2026
source /home/1ctnltug/atec2026/scripts/env/activate_atec2026_sim.sh

STAGE="${1:-easy}"  # easy | medium | official
MAX_ITERS="${ATEC_TASKD_ITERS:-}"
NUM_ENVS="${ATEC_TRAIN_NUM_ENVS:-1024}"

case "$STAGE" in
  easy)
    TASK="ATEC-Isaac-TaskD-FixedArm-B2W-Easy-v0"
    EXP="unitree_b2w_taskd_easy"
    PREV_EXP=""
    DEFAULT_ITERS=2500
    ;;
  medium)
    TASK="ATEC-Isaac-TaskD-FixedArm-B2W-Medium-v0"
    EXP="unitree_b2w_taskd_medium"
    PREV_EXP="unitree_b2w_taskd_easy"
    DEFAULT_ITERS=4000
    ;;
  official)
    TASK="ATEC-Isaac-TaskD-FixedArm-B2W-Official-v0"
    EXP="unitree_b2w_taskd_official"
    PREV_EXP="unitree_b2w_taskd_medium"
    DEFAULT_ITERS=7000
    ;;
  *)
    echo "Usage: $0 [easy|medium|official]"
    exit 1
    ;;
esac

MAX_ITERS="${MAX_ITERS:-$DEFAULT_ITERS}"

# Find warm-start checkpoint
OMNI_EXP_ROOT="logs/rsl_rl/unitree_b2w_rough_omni"
if [[ "$STAGE" == "easy" ]]; then
  SOURCE_CKPT="$(
    find "${OMNI_EXP_ROOT}" -maxdepth 2 -type f -name 'model_*.pt' \
      | sort -V \
      | tail -1
  )"
else
  SOURCE_CKPT="$(
    find "logs/rsl_rl/$PREV_EXP" -maxdepth 2 -type f -name 'model_*.pt' \
      | sort -V \
      | tail -1
  )"
fi

if [[ -z "$SOURCE_CKPT" || ! -f "$SOURCE_CKPT" ]]; then
  echo "ERROR: Warm-start checkpoint not found for stage '$STAGE'"
  if [[ -n "$PREV_EXP" ]]; then
    echo "  Expected in: logs/rsl_rl/$PREV_EXP/"
    echo "  Train the previous stage first."
  else
    echo "  Expected: demo/policy_taskd_omni.pt"
  fi
  exit 1
fi

echo "Stage: $STAGE | Iters: $MAX_ITERS | Envs: $NUM_ENVS"
echo "Warm-start from: $SOURCE_CKPT"

python scripts/rsl_rl/train.py \
  --task "$TASK" \
  --headless \
  --num_envs "$NUM_ENVS" \
  --max_iterations "$MAX_ITERS" \
  --actor_checkpoint "$SOURCE_CKPT" \
  --run_name "from_${STAGE}"
