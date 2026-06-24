# taskd-finetune - Draft

## Status: awaiting-approval (revised after Momus review)
## Pending action: user approval to start execution
## Approach: 15 todos across 5 waves, 3 new training envs, 61D obs, Task D rewards, warm-start 53→61
## Momus review: REJECT → fixed 3 blockers:
##   1. T3: Added explicit terrain override with _build_terrain_cfg() pattern
##   2. T9: Fixed warm-start to look in PREV_EXP not EXP
##   3. T1/T3/T8: Added pseudo-score computation from robot/box positions (no external current_score needed)

## Exploration Findings

### Codebase Structure
- Training envs registered in `train/locomotion/velocity/config/quadruped/unitree_b2/__init__.py` (4 existing)
- Each env needs `env_cfg_entry_point` + `rsl_rl_cfg_entry_point` kwargs
- Import chain: `train/__init__.py → locomotion → velocity → config → quadruped → unitree_b2/__init__.py`
- Training entry point: `scripts/rsl_rl/train.py` with `@hydra_task_config` decorator

### Current Omni16 Policy
- Observation: 53D (ang_vel×0.25, gravity, cmds, leg_pos, leg_vel×0.05, wheel_vel×0.05, prev_leg_actions, prev_wheel_actions)
- Action: 16D (12 leg positions + 4 wheel velocities)
- Actor architecture: [512, 256, 128] + ELU
- Checkpoint: `logs/rsl_rl/unitree_b2w_rough_omni/2026-06-21_19-02-24_from_straight/model_2800.pt`

### Warm-Start Mechanism
- `_load_actor_weights_only` in train.py handles obs dim mismatch
- Existing `_segmented_col_transfer_45_53` maps B2→B2W columns
- For 53→61: need new `_segmented_col_transfer_53_61` (53D prefix identical, 8D suffix random init)
- Critic and optimizer start fresh

### Task D Environment
- Box: 0.8×1.0×0.6m, 8kg, at (-3, 1.6, 0.5)
- Terrain: pit width 1.3-1.4m, platform height 1.0-1.2m, right side
- Rewards: +2 at x=-1.4, +14 box in [-1.4, 0.7], +20 at x=2.0
- Terminations: x>3.5 (success), fall (height<0.25), timeout (1200s)
- LiDAR: 16×360=5760 rays, detects box via mid-elevation right-front sector

### Key Design Decisions
1. **File location**: Add to existing `unitree_b2/` directory (consistent with pattern)
2. **Action space**: 16D (12 legs + 4 wheels), arms fixed at PD default
3. **Observation**: 61D = 53D prefix + 8D Task D features
4. **Curriculum**: 3 stages (Easy/Medium/Official) with progressive terrain/box difficulty
5. **Reward**: Keep sparse official rewards + add dense shaping for box progress, stage advancement, alignment

## Components
| ID | Component | Status | Evidence |
|----|-----------|--------|----------|
| C1 | Training env configs (3 envs) | planned | New file: taskd_omni_env_cfg.py |
| C2 | Observation terms (8D Task D features) | planned | New file: taskd_observations.py |
| C3 | Reward terms (dense shaping) | planned | New file: taskd_rewards.py |
| C4 | Warm-start 53→61 expansion | planned | Modify: train.py |
| C5 | solution.py b2w_taskd61 mode | planned | Modify: solution.py |
| C6 | Training/export scripts | planned | New files in scripts/training/ |
| C7 | Gym registration (3 new envs) | planned | Modify: unitree_b2/__init__.py |
| C8 | Testing & evaluation | planned | Smoke train + headless eval |
