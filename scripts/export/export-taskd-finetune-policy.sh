#!/usr/bin/env bash
set -eo pipefail

source /home/1ctnltug/atec2026/scripts/env/activate-atec2026-sim.sh

STAGE="${1:-official}"  # easy | medium | official
EXP="unitree_b2w_taskd_${STAGE}"

LATEST_CKPT="$(
  find "outputs/rsl_rl/$EXP" -maxdepth 2 -type f -name 'model_*.pt' \
    | sort -V \
    | tail -1
)"

if [[ -z "$LATEST_CKPT" ]]; then
  echo "ERROR: Could not find a trained checkpoint under outputs/rsl_rl/$EXP/"
  echo "Train the $STAGE stage first:"
  echo "  ./scripts/train/train-taskd-finetune.sh $STAGE"
  exit 1
fi

run_name="$(basename "$(dirname "$LATEST_CKPT")")"
checkpoint_name="$(basename "$LATEST_CKPT")"

echo "Exporting Task D finetuned policy:"
echo "  stage      = ${STAGE}"
echo "  run        = ${run_name}"
echo "  checkpoint = ${checkpoint_name}"
echo

python tools/atec/rsl_rl/play.py \
  --task "ATEC-Isaac-TaskD-FixedArm-B2W-${STAGE^}-v0" \
  --headless \
  --num_envs 1 \
  --resume \
  --load_run "${run_name}" \
  --checkpoint "$(realpath "$LATEST_CKPT")" \
  --video \
  --video_length 2

exported_policy="$(dirname "$LATEST_CKPT")/exported/policy.pt"
target_policy="submission/policy_taskd_finetuned.pt"

if [[ ! -f "$exported_policy" ]]; then
  echo "Exported policy not found at ${exported_policy}"
  echo "Check that play.py completed successfully."
  exit 1
fi

# Back up old finetuned policy if one exists
if [[ -f "$target_policy" ]]; then
  backup_policy="submission/policy_taskd_finetuned_before_$(date +%Y%m%d_%H%M%S).pt"
  cp "$target_policy" "$backup_policy"
  echo "Backed up previous finetuned policy → ${backup_policy}"
fi

cp "$exported_policy" "$target_policy"

echo
echo "Updated ${target_policy}"
echo "Size: $(du -h "$target_policy" | cut -f1)"
