# AGENTS.md

## Quick Start

This is a workspace, not a Python package. Key paths:
- Challenge project: `ATEC2026_Simulation_Challenge/`
- Isaac Lab (workspace copy, v2.3.2): `IsaacLab/`
- Local scripts: `scripts/`

**Prerequisite**: The `atec_rl_lab` package must be installed editable in the simulation environment. The activation script does this automatically, but if you see import errors, run:
```bash
cd ATEC2026_Simulation_Challenge/source/atec_rl_lab && pip install -e .
```

## Environment

Two conda environments exist. **Use `atec2026-sim` for simulation** (Python 3.11 matches Isaac Sim binaries):

```bash
source scripts/env/activate_atec2026_sim.sh
```

The `atec2026` environment (Python 3.12) is for package checks only—cannot launch Isaac Sim.

## Critical Flags

When running `play_atec_task.py`:

- `--enable_cameras` **required** (ATEC environments spawn cameras)
- `--disable_fabric` **required** on this machine (Isaac Sim bundles Warp 1.7.1; Isaac Lab v2.3.2 expects `wp.transform_compose`)

## Core Commands

```bash
# Activate environment
source scripts/env/activate_atec2026_sim.sh

# Verify environment (list registered ATEC tasks)
cd ATEC2026_Simulation_Challenge && python scripts/list_envs.py

# Run Task A B2W (headless) – must be run from challenge directory
python scripts/play_atec_task.py --task ATEC-TaskA-B2wPiper --headless --enable_cameras --disable_fabric --num_envs 1 --debug

# Run Task A B2W (GUI)
ATEC_GUI=1 ./scripts/task_a/run_task_a_b2w_gui.sh

# Record Task A video
./scripts/task_a/record_task_a_b2w_video.sh

# Train rough-straight from flat checkpoint
./scripts/training/train_b2_rough_straight_from_flat.sh

# Export latest trained policy to demo/
./scripts/training/export_latest_rough_straight_policy_to_demo.sh
```

## Development

Edit only: `ATEC2026_Simulation_Challenge/demo/solution.py`

Submission shape:
```
submission.zip/
├── solution.py
├── requirements.txt
└── models/
```

## Training Parameters

- `ATEC_ROUGH_STRAIGHT_ITERS` (default 8000)
- `ATEC_TRAIN_NUM_ENVS` (default 4096)

## Gotchas

- Use workspace IsaacLab (`/home/1ctnltug/atec2026/IsaacLab`), not system `/opt/IsaacLab` (different versions).
- Default policy returns `total_episode_reward: 0.00` until `solution.py` is improved.
- Video config: `scripts/task_a/task_a_video_config.sh`
- Video output: `artifacts/task_a_videos/` (symlink: `artifacts/latest_task_a_video.mp4`)
