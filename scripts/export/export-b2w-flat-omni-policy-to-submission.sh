#!/usr/bin/env bash
set -eo pipefail

source /home/1ctnltug/atec2026/scripts/env/activate-atec2026-sim.sh

EXP="unitree_b2w_flat_omni"
TARGET="submission/policy_b2w_flat_omni.pt"

# Find the latest checkpoint.
# 2>/dev/null suppresses find's "No such file or directory" when the experiment
# directory does not exist yet.  || true keeps set -e from aborting on that case.
LATEST_CKPT="$(
  find "outputs/rsl_rl/${EXP}" -maxdepth 2 -type f -name 'model_*.pt' 2>/dev/null \
    | sort -V \
    | tail -1
)" || true

if [[ -z "${LATEST_CKPT}" ]]; then
  echo "Error: No checkpoint found under outputs/rsl_rl/${EXP}"
  echo "Train the B2W flat omni policy first."
  exit 1
fi

run_name="$(basename "$(dirname "${LATEST_CKPT}")")"
checkpoint_name="$(basename "${LATEST_CKPT}")"

echo "Exporting B2W flat omni policy:"
echo "  run        = ${run_name}"
echo "  checkpoint = ${checkpoint_name}"
echo "  target     = ${TARGET}"
echo

cp -v "${LATEST_CKPT}" "${TARGET}"

echo
echo "Updated ${TARGET}"
