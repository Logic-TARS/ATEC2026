# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Workspace Layout

This is a **workspace directory**, not a single Python package. It co-locates the competition project, the Isaac Lab framework, local convenience scripts, generated artifacts, and backups.

```
atec2026/
├── ATEC2026_Simulation_Challenge/   # Competition code (the thing you actually edit)
│   ├── demo/solution.py             # ← THE file to edit for policy logic
│   ├── demo/policy.pt               # ← TorchScript policy loaded by solution.py
│   ├── source/atec_rl_lab/          # Official ATEC extension (DO NOT EDIT)
│   │   └── atec_rl_lab/
│   │       ├── tasks/               #   Gym env registrations: task_a..task_e, task_base
│   │       ├── assets/robots/       #   Robot configs: b2.py, b2w.py, g1/, piper.py, tron*.py
│   │       └── train/locomotion/    #   Training env configs (velocity tracking)
│   ├── scripts/
│   │   ├── play_atec_task.py        # Evaluation runner (what the submission server calls)
│   │   ├── rsl_rl/train.py          # RSL-RL training entry point
│   │   └── rsl_rl/play.py           # Training policy playback + TorchScript export
│   └── logs/rsl_rl/                 # Training checkpoints live here
├── IsaacLab/                        # Isaac Lab v2.3.2 (workspace copy — use this, not /opt)
├── scripts/                         # Local convenience wrappers
│   ├── env/                         #   Conda activation scripts
│   ├── task_a/                      #   Task A run + record + video config
│   └── training/                    #   Curriculum training + model export
├── artifacts/task_a_videos/         # Recorded video output
├── notes/                           # Deployment guide and long-form docs
├── archives/                        # Original zip files and installers
├── reports/                         # HTML reports
└── .omo/                            # Boulder project planner state (plans, drafts, run journal)
```

## Environment

Two conda environments exist. **Always use `atec2026-sim` for simulation and training** — it has Python 3.11 matching Isaac Sim's binary modules. The `atec2026` env (Python 3.12) can only do package checks and cannot launch Isaac Sim.

```bash
source scripts/env/activate_atec2026_sim.sh
```

This script sets three env vars used by other scripts:
- `ATEC2026_ROOT` → `/home/1ctnltug/atec2026`
- `ISAACLAB_ROOT` → `/home/1ctnltug/atec2026/IsaacLab`
- `ATEC_CHALLENGE_ROOT` → `/home/1ctnltug/atec2026/ATEC2026_Simulation_Challenge`

It also `cd`s into the challenge directory — all `python scripts/...` commands expect to run from there.

If you see import errors for `atec_rl_lab`, reinstall it editable:
```bash
cd ATEC2026_Simulation_Challenge/source/atec_rl_lab && pip install -e .
```

## Two IsaacLab copies — use the workspace one

| Path | Version | Purpose |
|------|---------|---------|
| `/home/1ctnltug/atec2026/IsaacLab` | **v2.3.2** | **Use this** — ATEC workspace copy |
| `/opt/IsaacLab` | v2.2.0 | System image copy, incompatible |

The workspace IsaacLab symlinks to the system Isaac Sim: `IsaacLab/_isaac_sim -> /opt/IsaacSim`.

## Critical flags for `play_atec_task.py`

Two flags are **always required** on this machine:
- `--enable_cameras` — ATEC environments spawn cameras; omitting this breaks the env
- `--disable_fabric` — Isaac Sim bundles Warp 1.7.1 but Isaac Lab v2.3.2 expects `wp.transform_compose`; without this flag the simulation crashes

## Core Commands

```bash
# Activate the simulation environment
source scripts/env/activate_atec2026_sim.sh

# List all registered ATEC environments (safe — no Isaac Sim import)
python scripts/list_envs.py

# Run Task A B2W headless
python scripts/play_atec_task.py --task ATEC-TaskA-B2wPiper --headless --enable_cameras --disable_fabric --num_envs 1 --debug

# Run Task A B2W with GUI
ATEC_GUI=1 ./scripts/task_a/run_task_a_b2w_gui.sh

# Record a Task A video
./scripts/task_a/record_task_a_b2w_video.sh

# Short test run (200 iters, 1024 envs)
ATEC_ROUGH_STRAIGHT_ITERS=200 ATEC_TRAIN_NUM_ENVS=1024 ./scripts/training/train_b2_rough_straight_from_flat.sh

# Full training run (8000 iters, 4096 envs)
./scripts/training/train_b2_rough_straight_from_flat.sh

# Export latest trained checkpoint to demo/policy.pt (backs up old one automatically)
./scripts/training/export_latest_rough_straight_policy_to_demo.sh
```

## Development: what you edit and what you don't

**Edit only:**
- `ATEC2026_Simulation_Challenge/demo/solution.py` — implements `AlgSolution` with a `predicts(obs, current_score)` method

**Read but don't modify:**
- `ATEC2026_Simulation_Challenge/source/atec_rl_lab/` — official ATEC extension (task definitions, robot configs, terrain, MDP rewards/terminations)
- `IsaacLab/` — framework, treat as read-only unless debugging framework internals

**Key files to understand when debugging:**
- Task A env config: `source/atec_rl_lab/atec_rl_lab/tasks/task_a/env_cfg.py` — terrain layout, scoring thresholds, episode config
- Task A MDP: `source/atec_rl_lab/atec_rl_lab/tasks/task_a/mdp/rewards.py` and `terminations.py`
- B2W robot config: `source/atec_rl_lab/atec_rl_lab/assets/robots/b2w.py`
- Training env config: `source/atec_rl_lab/atec_rl_lab/train/locomotion/velocity/velocity_env_cfg.py`

## Training pipeline

The training uses an RSL-RL PPO runner with a curriculum learning approach:

1. **Flat policy** (baseline): pre-trained on flat terrain only → `atec_robot_model/baseline/unitree_b2_flat/policy.pt`
2. **Rough-straight curriculum**: bootstraps from the flat checkpoint, trains on progressively rougher terrain. This is what `train_b2_rough_straight_from_flat.sh` does — it finds the latest flat checkpoint, then runs `rsl_rl/train.py` with `--actor_checkpoint` pointing at it (actor weights transferred, critic trained from scratch)
3. **Export**: `rsl_rl/play.py` with `--video --video_length 2` triggers TorchScript export; the export script copies the resulting `exported/policy.pt` into `demo/policy.pt`

Training checkpoints accumulate under `ATEC2026_Simulation_Challenge/logs/rsl_rl/`:
- `unitree_b2_flat/` — flat terrain training runs
- `unitree_b2_rough_straight/` — rough-straight curriculum runs

Training env vars:
- `ATEC_ROUGH_STRAIGHT_ITERS` — max iterations (default 8000)
- `ATEC_TRAIN_NUM_ENVS` — parallel env count (default 4096)

## Task A scoring

Task A (Off-road Navigation) terrain segments and their point values:

| Segment | X range | Points |
|---------|---------|--------|
| Flat start | -140 to -120 | 2 |
| Random Rough | -120 to -40 | 4 |
| Slopes | -40 to 40 | 8 |
| Stairs | 40 to 120 | 8 |
| Flat end | 120 to 140 | 4 |
| **Total** | | **26** |

The `AlgSolution.predicts()` method receives proprioceptive observations and must return `{"action": [...], "giveup": False}`. The action vector layout for B2W (24 DoF): indices 0-11 are leg positions (scale 0.5), 12-15 are wheel velocities (scale 5.0), 16-23 are arm positions (scale 0.5).

## `solution.py` architecture

The current `AlgSolution`:
- Loads a TorchScript policy (`policy.pt`) trained via RSL-RL velocity tracking
- Extracts a policy observation vector from the combined proprio input: `[base_ang_vel*0.25, projected_gravity, velocity_commands, leg_joint_pos, leg_joint_vel*0.05, previous_leg_actions_scaled]`
- The policy only outputs leg actions (12D); wheel and arm actions are hardcoded (wheels forward at `$ATEC_WHEEL_ACTION_VALUE` m/s default 3.0, arms at zero)
- Applies train↔env action scale conversion for leg joints via per-joint multipliers

## Submission format

```
submission.zip/
├── solution.py
├── requirements.txt      # extra pip deps (optional)
└── models/               # model weights (optional)
```

Daily limit: ≤10 submissions, ≤3 successful. Service startup timeout: 300s. Episode max runtime: 30 min.

## Video recording

Config in `scripts/task_a/task_a_video_config.sh`:
- `ATEC_VIDEO_LENGTH` — frames (500 ≈ 10s, 1500 ≈ 30s, 3000 ≈ 60s)
- `ATEC_CAMERA_MODE` — `follow` (behind robot) or `fixed` (overhead)
- `ATEC_DISABLE_FABRIC` — set to `1` if the robot appears frozen in video despite moving

Output: `artifacts/task_a_videos/task_a_b2w_<timestamp>.mp4` with symlink `artifacts/latest_task_a_video.mp4`.
