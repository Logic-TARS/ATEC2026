# Workspace Runbook

This document keeps local workspace commands and operational notes that are useful for reproducing experiments but too detailed for the GitHub portfolio README.

## Workspace Layout

This repository root is a workspace, not a Python package.

| Path | Purpose |
| --- | --- |
| `ATEC2026/` | Challenge project, training configs, official task scripts, deployment entrypoint |
| `IsaacLab/` | Workspace Isaac Lab copy, ignored for public GitHub release |
| `scripts/` | Local automation for environment activation, training, playback, and video capture |
| `artifacts/` | Generated videos, inference logs, submission zips, symlinks |
| `archives/` | Downloaded packages, large backups, old checkpoints |
| `reports/` | Local issue reports and analysis output |
| `notes/` | Longer local notes and deployment references |

Most active development happens in:

```text
ATEC2026/demo/solution.py
ATEC2026/source/atec_rl_lab/atec_rl_lab/train/
scripts/
```

## Environment

Use the simulation environment for Isaac Sim / Isaac Lab work:

```bash
source scripts/env/activate_atec2026_sim.sh
```

The activation script:

- activates the `atec2026-sim` conda environment;
- exports `ATEC2026_ROOT`, `ISAACLAB_ROOT`, and `ATEC_CHALLENGE_ROOT`;
- enters `ATEC2026/`, so commands like `python scripts/list_envs.py` run from the challenge directory.

If `atec_rl_lab` import errors appear, reinstall the extension:

```bash
cd ATEC2026/source/atec_rl_lab
pip install -e .
```

## Critical Runtime Flags

For `play_atec_task.py`:

- `--enable_cameras` is required because ATEC environments spawn cameras.
- `--disable_fabric` is required for training and non-video runs in this workspace because Isaac Sim bundles Warp 1.7.1 while Isaac Lab v2.3.2 expects `wp.transform_compose`.
- Omit `--disable_fabric` when recording video. Fabric must stay enabled for rendered frames to refresh; otherwise physics may move while the viewport looks frozen.

## Common Commands

### List Registered Environments

```bash
source scripts/env/activate_atec2026_sim.sh
python scripts/list_envs.py
```

### Run Task A B2W

Headless:

```bash
python scripts/play_atec_task.py \
  --task ATEC-TaskA-B2wPiper \
  --headless --enable_cameras --disable_fabric \
  --num_envs 1 --debug
```

GUI helper:

```bash
ATEC_GUI=1 ./scripts/task_a/run_task_a_b2w_gui.sh
```

Record video:

```bash
./scripts/task_a/record_task_a_b2w_video.sh
```

Video configuration lives in:

```text
scripts/task_a/task_a_video_config.sh
```

Task A videos are written under:

```text
artifacts/task_a_videos/
artifacts/latest_task_a_video.mp4
```

### Train Rough Straight Locomotion

```bash
./scripts/training/train_b2_rough_straight_from_flat.sh
```

Short smoke test:

```bash
ATEC_ROUGH_STRAIGHT_ITERS=200 ATEC_TRAIN_NUM_ENVS=1024 \
  ./scripts/training/train_b2_rough_straight_from_flat.sh
```

Export the latest policy:

```bash
./scripts/training/export_latest_rough_straight_policy_to_demo.sh
```

### Train B2W Omni Policies

Rough omni from straight:

```bash
./scripts/training/train_b2w_rough_omni_from_straight.sh
```

Flat omni:

```bash
ATEC_B2W_FLAT_OMNI_ITERS=10 ATEC_TRAIN_NUM_ENVS=64 \
  ./scripts/training/train_b2w_flat_omni.sh
```

Export policies:

```bash
./scripts/training/export_latest_b2w_omni_policy_to_demo.sh
./scripts/training/export_b2w_flat_omni_policy_to_demo.sh
```

### Task D Fine-Tuning

```bash
./scripts/training/train_taskd_finetune.sh official
./scripts/training/export_taskd_finetune_policy.sh official
```

Smoke test:

```bash
ATEC_TASKD_ITERS=10 ATEC_TRAIN_NUM_ENVS=64 \
  ./scripts/training/train_taskd_finetune.sh official
```

Record Task D video:

```bash
./scripts/task_d/record_task_d_b2w_video.sh
```

### Task F / Flat Task D Pre-Training

```bash
./scripts/train_env.sh \
  --task ATEC-Isaac-Velocity-Flat-TaskF-Unitree-B2W-Piper-v0 \
  --headless --enable_cameras --disable_fabric \
  --num_envs 64 --max_iterations 1000
```

Transfer from latest flat pre-training checkpoint to official Task D:

```bash
ATEC_TASKD_ITERS=7000 ATEC_TRAIN_NUM_ENVS=1024 \
  ./scripts/training/train_taskd_from_flat_pretrain.sh
```

Task F v6 domain-randomized hardening:

```bash
ATEC_TASKF_DR_ITERS=10 ATEC_TRAIN_NUM_ENVS=64 \
  ./scripts/training/train_taskf_short_omni_dr.sh
```

## Training Outputs

Checkpoints accumulate under:

```text
ATEC2026/logs/rsl_rl/
```

Common experiment directories:

```text
unitree_b2_flat/
unitree_b2_rough_straight/
unitree_b2w_rough_omni/
unitree_b2w_flat_omni/
unitree_b2w_taskd_easy/
unitree_b2w_taskd_medium/
unitree_b2w_taskd_official/
unitree_b2w_taskf_short_walk/
```

These directories are generated artifacts and should stay out of GitHub.

## Submission Shape

```text
submission.zip/
  solution.py
  requirements.txt
  models/
```

ATEC limits noted for local planning:

- up to 10 submissions per day;
- up to 3 successful submissions per day;
- 300 seconds service startup limit;
- 30 minutes maximum episode length.

## Useful Environment Variables

| Variable | Purpose |
| --- | --- |
| `ATEC_POLICY_PATH` | Policy file path, defaulting to `demo/policy.pt` |
| `ATEC_POLICY_MODE` | Policy adapter mode such as `b2w_omni16`, `b2w_taskd61`, or `b2w_locomotion_56d` |
| `ATEC_TASKD_COMMAND_MODE` | Fixed Task D command mode or `auto` |
| `ATEC_HIGH_LEVEL_TASK` | High-level deployment behavior, such as `task_d_auto` |
| `ATEC_ROBOT_TYPE` | Empty for B2W, `tron2awheel` for Tron2AWheel experiments |
| `ATEC_TRAIN_NUM_ENVS` | Number of vectorized training environments |
| `ATEC_TASKD_ITERS` | Task D fine-tuning iteration count |
| `ATEC_TASKF_DR_ITERS` | Task F DR hardening iteration count |

## Gotchas

- Use the workspace Isaac Lab at `/home/1ctnltug/atec2026/IsaacLab`, not `/opt/IsaacLab`.
- Tron2AWheel environment setup can succeed while the play loop hangs; this has been observed as an Isaac Sim environment-level issue rather than a `solution.py` bug.
- Checkpoint dimensions must match the environment observation/action dimensions. For example, a 53D flat omni policy cannot be directly played in a 56D Task F config.
- For video capture, use the task-specific recording scripts and avoid `--disable_fabric`.
