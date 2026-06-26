#!/usr/bin/env bash
set -eo pipefail

# ------------------------------------------------------------------
# Generic training wrapper for any registered Isaac Lab environment.
# Usage: ./scripts/train_env.sh --task <env_id> [extra_args...]
#
# Examples:
#   ./scripts/train_env.sh --task ATEC-Isaac-Velocity-Flat-TaskF-Unitree-B2W-Piper-v0 \
#       --headless --enable_cameras --disable_fabric --num_envs 64 --max_iterations 1000
#
#   ./scripts/train_env.sh --task ATEC-Isaac-Velocity-Flat-Omni-B2W-Piper-v0 \
#       --headless --enable_cameras --disable_fabric --num_envs 256 --max_iterations 10000
# ------------------------------------------------------------------

show_usage() {
    cat <<EOF
Usage: $(basename "$0") --task <env_id> [extra_args...]

Required:
  --task <env_id>   ID of the Isaac Lab environment to train on
                    (e.g. ATEC-Isaac-Velocity-Flat-TaskF-Unitree-B2W-Piper-v0)

Common extra args (forwarded to scripts/rsl_rl/train.py):
  --headless                   Run without GUI
  --enable_cameras             Enable camera sensors
  --disable_fabric             Disable Fabric (needed for training)
  --num_envs <N>               Number of parallel environments
  --max_iterations <N>         Maximum training iterations
  --seed <N>                   Random seed
  --video                      Record training video
  --actor_checkpoint <path>    Warm-start from a checkpoint
  --run_name <name>            Experiment run name

Example:
  $(basename "$0") --task ATEC-Isaac-Velocity-Flat-TaskF-Unitree-B2W-Piper-v0 \\
      --headless --enable_cameras --disable_fabric --num_envs 64
EOF
    exit 1
}

# Require at least --task
if [[ $# -eq 0 ]]; then
    show_usage
fi

has_task=false
for arg in "$@"; do
    if [[ "$arg" == "--task" ]] || [[ "$arg" == --task=* ]]; then
        has_task=true
        break
    fi
done
if ! $has_task; then
    echo "ERROR: --task is required."
    echo
    show_usage
fi

# Source environment and change to challenge root
source /home/1ctnltug/atec2026/scripts/env/activate_atec2026_sim.sh

# Forward all args to the train script
python scripts/rsl_rl/train.py "$@"
