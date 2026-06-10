#!/usr/bin/env bash

set -e

source /home/1ctnltug/miniconda3/etc/profile.d/conda.sh
conda activate atec2026

export ATEC2026_ROOT=/home/1ctnltug/atec2026
export ISAACLAB_ROOT=/home/1ctnltug/atec2026/IsaacLab

cd "$ATEC2026_ROOT"

echo "Activated atec2026"
echo "Workspace: $ATEC2026_ROOT"
echo "Isaac Lab: $ISAACLAB_ROOT"
