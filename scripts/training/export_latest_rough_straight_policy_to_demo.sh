#!/usr/bin/env bash
set -eo pipefail

cd /home/1ctnltug/atec2026/ATEC2026_Simulation_Challenge
source /home/1ctnltug/atec2026/scripts/env/activate_atec2026_sim.sh

experiment_root="logs/rsl_rl/unitree_b2_rough_straight"
task="ATEC-Isaac-Velocity-Rough-Straight-Unitree-B2-v0"

latest_ckpt="$(
  find "${experiment_root}" -mindepth 2 -maxdepth 2 -type f -name 'model_*.pt' \
    ! -path "*/flat_bootstrap/*" \
    | sort -V \
    | tail -1
)"

if [[ -z "${latest_ckpt}" ]]; then
  echo "Could not find a trained rough-straight checkpoint under ${experiment_root}"
  exit 1
fi

run_name="$(basename "$(dirname "${latest_ckpt}")")"
checkpoint_name="$(basename "${latest_ckpt}")"

echo "Exporting:"
echo "  run=${run_name}"
echo "  checkpoint=${checkpoint_name}"
echo

python scripts/rsl_rl/play.py \
  --task "${task}" \
  --headless \
  --num_envs 1 \
  --resume \
  --load_run "${run_name}" \
  --checkpoint "${checkpoint_name}" \
  --video \
  --video_length 2

exported_policy="$(dirname "${latest_ckpt}")/exported/policy.pt"
target_policy="demo/policy.pt"
backup_policy="demo/policy_before_rough_straight_$(date +%Y%m%d_%H%M%S).pt"

if [[ ! -f "${exported_policy}" ]]; then
  echo "Exported policy not found at ${exported_policy}"
  exit 1
fi

cp "${target_policy}" "${backup_policy}"
cp "${exported_policy}" "${target_policy}"

echo
echo "Updated ${target_policy}"
echo "Backup saved to ${backup_policy}"
