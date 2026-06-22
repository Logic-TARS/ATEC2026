#!/usr/bin/env bash

set -e

source /home/1ctnltug/miniconda3/etc/profile.d/conda.sh
conda activate atec2026-sim

source /opt/IsaacSim/setup_conda_env.sh

export ATEC2026_ROOT=/home/1ctnltug/atec2026
export ISAACLAB_ROOT=/home/1ctnltug/atec2026/IsaacLab
export ATEC_CHALLENGE_ROOT=/home/1ctnltug/atec2026/ATEC2026

cd "$ATEC_CHALLENGE_ROOT"

echo "Activated atec2026-sim"
echo "Workspace: $ATEC2026_ROOT"
echo "Challenge: $ATEC_CHALLENGE_ROOT"
echo "Isaac Lab: $ISAACLAB_ROOT"
