#!/usr/bin/env bash
set -eo pipefail

cd /home/1ctnltug/atec2026/ATEC2026
source /home/1ctnltug/atec2026/scripts/env/activate_atec2026_sim.sh

experiment_root="logs/rsl_rl/unitree_b2w_rough_omni"
task="ATEC-Isaac-Velocity-Rough-Omni-B2W-Piper-v0"

# Find the latest checkpoint, excluding any flat_bootstrap or other bootstrap dirs.
latest_ckpt="$(
  find "${experiment_root}" -mindepth 2 -maxdepth 2 -type f -name 'model_*.pt' \
    | sort -V \
    | tail -1
)"

if [[ -z "${latest_ckpt}" ]]; then
  echo "Could not find a trained B2W omni checkpoint under ${experiment_root}"
  echo "Train the omni policy first: ./scripts/training/train_b2w_rough_omni_from_straight.sh"
  exit 1
fi

run_name="$(basename "$(dirname "${latest_ckpt}")")"
checkpoint_name="$(basename "${latest_ckpt}")"

echo "Exporting B2W+Piper omni policy:"
echo "  run        = ${run_name}"
echo "  checkpoint = ${checkpoint_name}"
echo

python scripts/rsl_rl/play.py \
  --task "${task}" \
  --headless \
  --num_envs 1 \
  --resume \
  --load_run "${run_name}" \
  --checkpoint "$(realpath "${latest_ckpt}")" \
  --video \
  --video_length 2

exported_policy="$(dirname "${latest_ckpt}")/exported/policy.pt"
target_policy="demo/policy_taskd_omni.pt"

if [[ ! -f "${exported_policy}" ]]; then
  echo "Exported policy not found at ${exported_policy}"
  echo "Check that play.py completed successfully."
  exit 1
fi

# Back up old Task D omni policy if one exists.
if [[ -f "${target_policy}" ]]; then
  backup_policy="demo/policy_taskd_omni_before_$(date +%Y%m%d_%H%M%S).pt"
  cp "${target_policy}" "${backup_policy}"
  echo "Backed up previous omni policy → ${backup_policy}"
fi

cp "${exported_policy}" "${target_policy}"

echo
echo "Updated ${target_policy}"
echo "demo/policy.pt was NOT touched."
