# AGENTS.md

## Quick Start

This is a workspace, not a Python package. Key paths:
- Challenge project: `ATEC2026/`
- Isaac Lab (workspace copy, v2.3.2): `IsaacLab/`
- Local scripts: `scripts/`

**Prerequisite**: The `atec_rl_lab` package must be installed editable in the simulation environment. The activation script does this automatically, but if you see import errors, run:
```bash
cd ATEC2026/source/atec_rl_lab && pip install -e .
```

## Environment

Two conda environments exist. **Use `atec2026-sim` for simulation** (Python 3.11 matches Isaac Sim binaries):

```bash
source scripts/env/activate_atec2026_sim.sh
```

The `atec2026` environment (Python 3.12) is for package checks only—cannot launch Isaac Sim.

The activate script exports `ATEC2026_ROOT`, `ISAACLAB_ROOT`, `ATEC_CHALLENGE_ROOT` and **`cd`s into the challenge directory** — so `python scripts/...` commands below run from there without a manual `cd`.

## Critical Flags

When running `play_atec_task.py`:

- `--enable_cameras` **always required** (ATEC environments spawn cameras)
- `--disable_fabric` **required for training and non-video runs** (Isaac Sim bundles Warp 1.7.1; Isaac Lab v2.3.2 expects `wp.transform_compose`). **Omit for video recording** — Fabric must be ON for video frames to refresh; with `--disable_fabric` the robot appears frozen in video.

## Core Commands

```bash
# Activate environment (also cd's into ATEC2026/)
source scripts/env/activate_atec2026_sim.sh

# Verify environment (list registered ATEC tasks)
python scripts/list_envs.py

# Run Task A B2W (headless)
python scripts/play_atec_task.py --task ATEC-TaskA-B2wPiper --headless --enable_cameras --disable_fabric --num_envs 1 --debug

# Run Task A B2W (GUI)
ATEC_GUI=1 ./scripts/task_a/run_task_a_b2w_gui.sh

# Record Task A video
./scripts/task_a/record_task_a_b2w_video.sh

# Train rough-straight from flat checkpoint (short test: ATEC_ROUGH_STRAIGHT_ITERS=200 ATEC_TRAIN_NUM_ENVS=1024)
./scripts/training/train_b2_rough_straight_from_flat.sh

# Train 16D omni B2W+Piper for Task D (smoke test: ATEC_B2W_OMNI_ITERS=10 ATEC_TRAIN_NUM_ENVS=64)
./scripts/training/train_b2w_rough_omni_from_straight.sh

# Train B2W flat omni from scratch (smoke test: ATEC_B2W_FLAT_OMNI_ITERS=10 ATEC_TRAIN_NUM_ENVS=64)
./scripts/training/train_b2w_flat_omni.sh

# TaskF v6 DR hardening — warm-starts from latest TaskF short checkpoint (smoke test: ATEC_TASKF_DR_ITERS=10 ATEC_TRAIN_NUM_ENVS=64)
./scripts/training/train_taskf_short_omni_dr.sh

# Export latest trained policy to demo/
./scripts/training/export_latest_rough_straight_policy_to_demo.sh

# Export omni policy → demo/policy_taskd_omni.pt
./scripts/training/export_latest_b2w_omni_policy_to_demo.sh

# Export flat omni policy → demo/policy_b2w_flat_omni.pt
./scripts/training/export_b2w_flat_omni_policy_to_demo.sh

# Task D fine-tune from official checkpoint (difficulty: easy|medium|official; ATEC_TASKD_ITERS=200 for test)
./scripts/training/train_taskd_finetune.sh official

# Export fine-tuned Task D policy → demo/policy_taskd_finetuned.pt
./scripts/training/export_taskd_finetune_policy.sh official

# Record Task D video
./scripts/task_d/record_task_d_b2w_video.sh
```

## Development

Edit only: `ATEC2026/demo/solution.py` and training configs under `source/atec_rl_lab/atec_rl_lab/train/`.

Read but don't modify: `source/atec_rl_lab/atec_rl_lab/tasks/` (official ATEC extension), `assets/robots/`, and `IsaacLab/`.

`solution.py` implements `AlgSolution.predicts(obs, current_score)` → `{"action": [...], "giveup": bool}`. B2W action vector (24 DoF): indices 0-11 legs (env scale ×0.5), 12-15 wheels (×5.0 → rad/s), 16-23 arms (×0.5).

Key env vars for solution.py:
- `ATEC_POLICY_PATH` — policy file (default `demo/policy.pt`)
- `ATEC_POLICY_MODE` — `""` (12D legacy), `b2w_omni16` (16D omni), or `b2w_taskd61`
- `ATEC_TASKD_COMMAND_MODE` — fixed command or `auto` (state machine)
- `ATEC_ROBOT_TYPE` — `""` (B2W) or `tron2awheel`

Submission shape:
```
submission.zip/
├── solution.py
├── requirements.txt
└── models/
```
Limits: ≤10 submissions/day, ≤3 successful/day, 300s service startup, 30 min max episode.

## Training Parameters

- `ATEC_ROUGH_STRAIGHT_ITERS` (default 8000)
- `ATEC_B2W_OMNI_ITERS` (default 12000)
- `ATEC_B2W_FLAT_OMNI_ITERS` (default 10000)
- `ATEC_TASKD_ITERS` (default 2000)
- `ATEC_TASKF_DR_ITERS` (default 100)
- `ATEC_TRAIN_NUM_ENVS` (default 4096)

Checkpoints accumulate under `ATEC2026/logs/rsl_rl/` (`unitree_b2_flat/`, `unitree_b2_rough_straight/`, `unitree_b2w_rough_omni/`, `unitree_b2w_flat_omni/`, `unitree_b2w_taskd_*`, `unitree_b2w_taskf_short_walk/`).

## Gotchas

- Use workspace IsaacLab (`/home/1ctnltug/atec2026/IsaacLab`), not system `/opt/IsaacLab` (different versions).
- Tron2AWheel (`ATEC_ROBOT_TYPE=tron2awheel`): env setup succeeds but play loop never starts — Isaac Sim env-level hang, not a solution.py bug.
- Video config: `scripts/task_a/task_a_video_config.sh`. Video output: `artifacts/task_a_videos/` (symlink: `artifacts/latest_task_a_video.mp4`).
