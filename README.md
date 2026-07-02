# ATEC 2026 Embodied RL Workspace

面向 ATEC 2026 仿真挑战的轮足机器人强化学习与部署工程：基于 Isaac Lab 和 RSL-RL，为 Unitree B2W + AgileX Piper 构建越野移动、全向运动、推箱越障和提交部署管线。

An embodied-RL engineering workspace for the ATEC 2026 Simulation Challenge: Isaac Lab + RSL-RL training, Unitree B2W + AgileX Piper locomotion, box-pushing obstacle traversal, and submission-ready policy deployment.

<p align="center">
  <img src="ATEC2026/doc/b2w.png" width="220" alt="Unitree B2W with Piper">
  <img src="ATEC2026/doc/task_a.gif" width="220" alt="Task A off-road navigation">
  <img src="ATEC2026/doc/task_d.gif" width="220" alt="Task D obstacle traversal">
</p>

## Highlights

- **Curriculum learning for legged locomotion**: flat locomotion -> rough straight walking -> rough omni B2W policy -> task-specific fine-tuning.
- **Task D loco-manipulation pipeline**: 61D task observation, 16D locomotion action, dense rewards for box pushing and obstacle traversal, easy/medium/official staged fine-tuning.
- **Deployment adapter**: maps trained 16D B2W locomotion output to the official 24D action interface: 12 leg position actions, 4 wheel velocity actions, and 8 fixed arm actions.
- **Online control logic**: `solution.py` combines policy inference with high-level state machines, score-aware phase switching, LiDAR/height-scan correction, heading lock, speed correction, and recovery behavior.
- **Reproducible engineering workflow**: scripted environment activation, smoke-test training, policy export, local evaluation, video recording, and submission packaging.

## Tech Stack

| Area | Tools / Components |
| --- | --- |
| Simulator | Isaac Sim, Isaac Lab v2.3.2 |
| RL | RSL-RL PPO, TorchScript policy export |
| Robots | Unitree B2 / B2W, AgileX Piper |
| Tasks | Task A off-road navigation, Task D box-pushing obstacle traversal, Task F flat pre-training |
| Engineering | Gym environment registration, shell automation, local video capture, submission adapter |

## System Overview

```text
ATEC2026/source/atec_rl_lab/
  train/locomotion/velocity/
    config/quadruped/unitree_b2/      custom training envs and PPO configs
    mdp/                              rewards, commands, observations, events

scripts/
  env/                                conda + Isaac Lab workspace activation
  training/                           curriculum training and export scripts
  task_a/, task_d/                    local play and video recording helpers

ATEC2026/demo/
  solution.py                         official submission entrypoint
  policy*.pt                          local policy artifacts, not intended for GitHub release
```

Training environments are registered in `ATEC2026/source/atec_rl_lab/atec_rl_lab/train/locomotion/velocity/config/quadruped/unitree_b2/__init__.py`. The deployment path is centralized in `ATEC2026/demo/solution.py`.

## Core Workflows

### 1. Environment Check

Requires the `atec2026-sim` conda environment and the workspace Isaac Lab copy. The activation script exports `ATEC2026_ROOT`, `ISAACLAB_ROOT`, `ATEC_CHALLENGE_ROOT`, and enters `ATEC2026/`.

```bash
source scripts/env/activate_atec2026_sim.sh
python scripts/list_envs.py
```

### 2. Task A: Off-Road B2W Playback

```bash
python scripts/play_atec_task.py \
  --task ATEC-TaskA-B2wPiper \
  --headless --enable_cameras --disable_fabric \
  --num_envs 1 --debug
```

Video recording uses Fabric-enabled rendering so viewport frames refresh correctly:

```bash
./scripts/task_a/record_task_a_b2w_video.sh
```

### 3. Locomotion Curriculum

```bash
# Rough straight walking from a flat checkpoint
./scripts/training/train_b2_rough_straight_from_flat.sh

# B2W + Piper rough omni policy from rough-straight locomotion
./scripts/training/train_b2w_rough_omni_from_straight.sh

# Export omni policy to demo/policy_taskd_omni.pt
./scripts/training/export_latest_b2w_omni_policy_to_demo.sh
```

### 4. Task D: Box-Pushing Obstacle Traversal

Task D fine-tuning adds task-specific observations and rewards while keeping the deployable 16D locomotion interface.

```bash
# Smoke test
ATEC_TASKD_ITERS=10 ATEC_TRAIN_NUM_ENVS=64 \
  ./scripts/training/train_taskd_finetune.sh official

# Full fine-tune entrypoint
./scripts/training/train_taskd_finetune.sh official

# Export to demo/policy_taskd_finetuned.pt
./scripts/training/export_taskd_finetune_policy.sh official
```

Task F is also used as a flat-terrain Task D pre-training stage with matching 61D observation and 16D action dimensions:

```bash
ATEC_TASKD_ITERS=7000 ATEC_TRAIN_NUM_ENVS=1024 \
  ./scripts/training/train_taskd_from_flat_pretrain.sh
```

## Submission Interface

The official evaluator calls:

```python
class AlgSolution:
    def predicts(self, obs, current_score):
        return {"action": action, "giveup": False}
```

For B2W + Piper, the official action is 24D:

| Slice | Meaning | Deployment behavior |
| --- | --- | --- |
| `0:12` | Leg joint position commands | From locomotion policy |
| `12:16` | Wheel velocity commands | From locomotion policy |
| `16:24` | Piper arm position commands | Fixed / zeroed for locomotion tasks |

This project keeps the learned locomotion policy compact at 16D and performs the official action expansion in `ATEC2026/demo/solution.py`.

## Key Environment IDs

| Environment | Purpose |
| --- | --- |
| `ATEC-Isaac-Velocity-Rough-Straight-Unitree-B2-v0` | B2 rough straight curriculum |
| `ATEC-Isaac-Velocity-Rough-Omni-B2W-Piper-v0` | B2W + Piper rough omni locomotion |
| `ATEC-Isaac-Velocity-Flat-Omni-B2W-Piper-v0` | Flat omni baseline / smoke tests |
| `ATEC-Isaac-TaskD-FixedArm-B2W-Easy-v0` | Task D easy fine-tuning |
| `ATEC-Isaac-TaskD-FixedArm-B2W-Medium-v0` | Task D medium fine-tuning |
| `ATEC-Isaac-TaskD-FixedArm-B2W-Official-v0` | Task D official fine-tuning |
| `ATEC-Isaac-Velocity-Flat-TaskF-Unitree-B2W-Piper-v0` | Flat Task D pre-training |
| `ATEC-Isaac-Velocity-ShortOmniDR-TaskF-Unitree-B2W-Piper-v0` | Task F domain-randomized hardening |

## Repository Notes

- `ATEC2026/readme.md` is the original challenge README and is kept as upstream-facing reference.
- `scripts/README.md` is the command quick reference for training, playback, and environment viewing.
- `docs/workspace-runbook.md` keeps local workspace operations that are useful for reproducing runs but too detailed for a portfolio homepage.
- Large assets are intentionally excluded from GitHub: `IsaacLab/`, `ATEC2026/logs/`, `artifacts/`, `archives/`, robot model downloads, submission zips, and local checkpoints.

## English Summary

This repository demonstrates an end-to-end embodied-RL workflow: custom Isaac Lab training environments, RSL-RL PPO curricula, policy export, and an official ATEC submission adapter. The most important engineering work is the bridge between train-time locomotion policies and the online competition interface: compact 16D learned actions, task-specific observations and rewards for Task D, and robust deployment logic in `AlgSolution.predicts`.

## License

The challenge project includes its own MIT license in `ATEC2026/LICENSE`. Third-party components such as Isaac Lab and robot assets follow their respective upstream licenses.
