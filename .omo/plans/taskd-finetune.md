# taskd-finetune - Work Plan

## TL;DR (For humans)

**What you'll get:** A Task D fine-tuning pipeline with 3 curriculum stages (Easy/Medium/Official), 61D observation space (53D locomotion + 8D Task D features), Task D aligned rewards, and matching solution.py inference mode — all warm-started from the existing omni locomotion policy.

**Why this approach:** Reuse the proven omni locomotion policy as foundation, add Task D specific observations (box detection, score, stage) and rewards (box progress, stage advancement) without modifying official task code. Three-stage curriculum lets the policy learn pushing before tackling the full difficulty pit.

**What it will NOT do:** Touch official `tasks/task_d/` code, change the 16D action space, add arm control, or modify the existing omni locomotion training pipeline.

**Effort:** Large — 15 implementation todos across 5 waves, plus training runs
**Risk:** Medium — warm-start 53→61 obs expansion needs careful column mapping; reward shaping weights need tuning
**Decisions to sanity-check:** 61D observation ordering (suffix after 53D prefix), Easy/Medium/Official terrain parameters, dense reward weights

Your next move: approve this plan, then I'll delegate execution.

---

> TL;DR (machine): Effort=Large Risk=Medium | 15 todos across 5 waves | deliverables=3 training envs + 61D obs + Task D rewards + solution.py mode + training scripts

## Scope
### Must have
1. Three new training env configs (Easy/Medium/Official) with box in scene, LiDAR box detection, Task D terrain
2. Eight new observation terms: score_norm, stage_one_hot(4), box_detected, box_bearing, box_distance_norm
3. Dense reward shaping: box_x_progress, stage_progress, alignment, fall_penalty, action_smoothness
4. Warm-start 53→61 column transfer in train.py
5. solution.py `b2w_taskd61` inference mode
6. Training and export scripts
7. Three gym.register calls for new envs
8. Smoke test + evaluation pipeline

### Must NOT have (guardrails, anti-slop, scope boundaries)
- NO modifications to `tasks/task_d/` (official competition code)
- NO changes to existing omni locomotion env or its training pipeline
- NO arm control in training (arms stay at PD default)
- NO changes to the 16D action space
- NO new dependencies beyond existing IsaacLab/RSL-RL
- NO modifications to `IsaacLab/` framework code

## Verification strategy
> Zero human intervention - all verification is agent-executed.
- Test decision: tests-after (shape checks + smoke train + headless eval)
- Evidence: .omo/evidence/task-<N>-taskd-finetune.<ext>

## Execution strategy
### Parallel execution waves

**Wave 1: Foundation (4 parallel todos — all independent new files)**
- T1. Create `taskd_observations.py` — 8 custom ObsTerm functions
- T2. Create `taskd_rewards.py` — 5 custom reward terms
- T3. Create `taskd_omni_env_cfg.py` — 3 env config classes
- T4. Create `agents/rsl_rl_ppo_taskd_cfg.py` — PPO runner config

**Wave 2: Integration (3 todos — depend on Wave 1)**
- T5. Register 3 new envs in `unitree_b2/__init__.py`
- T6. Add `_segmented_col_transfer_53_61` to `train.py`
- T7. Update `mdp/__init__.py` imports

**Wave 3: Inference (1 todo — depends on T1 for obs layout)**
- T8. Add `b2w_taskd61` mode to `solution.py`

**Wave 4: Scripts (2 todos — depend on Wave 2)**
- T9. Create `train_taskd_finetune.sh`
- T10. Create `export_taskd_finetune_policy.sh`

**Wave 5: Testing (5 todos — depend on Wave 4)**
- T11. Smoke test (64 envs, 10 iters) — verify shapes
- T12. Easy stage training (1024 envs, 1500-2500 iters)
- T13. Medium stage training (1024 envs, 2500-4000 iters)
- T14. Official stage training (1024-2048 envs, 4000-7000 iters)
- T15. Headless evaluation on official Task D env

### Dependency matrix
| Todo | Depends on | Blocks | Can parallelize with |
| --- | --- | --- | --- |
| T1. taskd_observations.py | — | T3, T7 | T2, T4 |
| T2. taskd_rewards.py | — | T3, T7 | T1, T4 |
| T3. taskd_omni_env_cfg.py | T1, T2 | T5, T7 | T4 |
| T4. rsl_rl_ppo_taskd_cfg.py | — | T5 | T1, T2, T3 |
| T5. Register 3 envs | T3, T4 | T9, T11 | T6, T7, T8 |
| T6. _segmented_col_transfer_53_61 | — | T9 | T5, T7, T8 |
| T7. mdp/__init__.py imports | T1, T2 | T3, T5 | T6, T8 |
| T8. solution.py b2w_taskd61 | T1 | T15 | T5, T6, T7 |
| T9. train_taskd_finetune.sh | T5, T6 | T11 | T10 |
| T10. export_taskd_finetune_policy.sh | T5 | T12-T14 | T9 |
| T11. Smoke test | T9 | T12 | T10 |
| T12. Easy training | T11 | T13 | T10 |
| T13. Medium training | T12 | T14 | — |
| T14. Official training | T13 | T15 | — |
| T15. Headless evaluation | T8, T14 | — | — |

## Todos

<!-- APPEND TASK BATCHES BELOW THIS LINE WITH edit/apply_patch - never rewrite the headers above. -->

### Wave 1: Foundation (parallel)

- [x] 1. Create `taskd_observations.py` — 5 custom Task D observation terms
  What to do / Must NOT do: Create file at `ATEC2026/source/atec_rl_lab/atec_rl_lab/train/locomotion/velocity/mdp/taskd_observations.py`. Implement 5 observation functions:

  **Pseudo-score mechanism** (computes score from env state, no external `current_score` needed):
  ```python
  def _compute_pseudo_score(env, robot_x, box_x, reward_given_buf):
      """Compute Task D score from robot and box positions. Returns (score, updated_buf)."""
      score = torch.zeros(robot_x.shape[0], device=robot_x.device)
      # +2 at x=-1.4
      mask1 = (robot_x > -1.4) & (reward_given_buf[:, 0] == 0)
      score[mask1] += 2.0
      reward_given_buf[mask1, 0] = 1
      # +14 box in [-1.4, 0.7]
      box_in_range = (box_x >= -1.4) & (box_x <= 0.7)
      mask2 = box_in_range & (reward_given_buf[:, 1] == 0)
      score[mask2] += 14.0
      reward_given_buf[mask2, 1] = 1
      # +20 at x=2.0
      mask3 = (robot_x > 2.0) & (reward_given_buf[:, 2] == 0)
      score[mask3] += 20.0
      reward_given_buf[mask3, 2] = 1
      return score
  ```
  The `reward_given_buf` is a persistent buffer (shape [num_envs, 3]) initialized in env reset, stored on `env._taskd_reward_given`. This mirrors the `RewardCrossX._reward_given` pattern from `task_d/mdp/rewards.py:126-133`.

  Implement 5 functions:
  - `taskd_score_norm(env, asset_cfg, box_cfg, max_score=36.0)` → 1D: pseudo_score / max_score, clamped [0,1]. Accesses `env.scene["robot"]` and `env.scene["box"]` for positions. Calls `_compute_pseudo_score`.
  - `taskd_stage_onehot(env, asset_cfg, box_cfg, thresholds=[1.9, 15.0, 21.0])` → 4D: one-hot based on pseudo-score. Stage 0 if score < thresholds[0], stage 1 if < thresholds[1], etc.
  - `taskd_box_detected(env, sensor_cfg, box_cfg)` → 1D: 1.0 if box detected in LiDAR, else 0.0. Reuse the same detection logic as solution.py `_parse_lidar` (mid-elevation channels 4-11, right-front sector rays 90-170, 15+ valid points, p10<2.5m, spread<0.6m).
  - `taskd_box_bearing(env, sensor_cfg, box_cfg)` → 1D: normalized bearing to box in [-1, 1] (radians / π).
  - `taskd_box_distance_norm(env, sensor_cfg, box_cfg, max_dist=5.0)` → 1D: box distance / max_dist, clamped [0,1].

  Each function must be a standalone callable matching `ObsTerm.func` signature `(env, **kwargs) -> torch.Tensor`. Must NOT import from tasks/ or demo/. Use the same lidar ray casting approach as existing `height_scan` in `mdp/observations.py`.
  Parallelization: Wave 1 | Blocked by: — | Blocks: T3, T7 | Can parallelize with: T2, T4
  References:
  - `ATEC2026/source/atec_rl_lab/atec_rl_lab/train/locomotion/velocity/mdp/observations.py` — existing ObsTerm pattern
  - `ATEC2026/demo/solution.py:338-364` — `_parse_lidar` detection logic to replicate
  - `ATEC2026/source/atec_rl_lab/atec_rl_lab/tasks/task_base/envs_base_cfg.py:47-59` — LiDAR pattern config (16 channels, 360 horizontal)
  - `ATEC2026/source/atec_rl_lab/atec_rl_lab/tasks/task_d/mdp/rewards.py:195-273` — RewardCrossX pattern for accessing robot state
  Acceptance criteria:
  - `python -c "from atec_rl_lab.train.locomotion.velocity.mdp.taskd_observations import *; print('OK')"` succeeds
  - Each function returns tensor of correct shape [num_envs, N]
  QA scenarios:
  - Happy: import succeeds, functions are callable
  - Failure: missing torch import → ImportError
  - Evidence: .omo/evidence/task-1-taskd-finetune-import.txt
  Commit: N (will be part of final commit)

- [x] 2. Create `taskd_rewards.py` — 5 custom Task D reward terms
  What to do / Must NOT do: Create file at `ATEC2026/source/atec_rl_lab/atec_rl_lab/train/locomotion/velocity/mdp/taskd_rewards.py`. Implement 5 reward functions:
  - `box_x_progress(env, box_cfg, asset_cfg)` → 1D: reward proportional to box's x-velocity (positive = pushing forward). Use `(box_pos_x - prev_box_pos_x) / dt`.
  - `stage_progress(env, asset_cfg, thresholds=[(-1.4, 2.0), (2.0, 3.5)])` → 1D: +1 when robot crosses each threshold (similar to RewardCrossX but dense per-step)
  - `alignment_with_box(env, asset_cfg, box_cfg)` → 1D: reward for robot facing toward the box. Compute angle between robot forward vector and vector to box. Reward = cos(angle), range [-1, 1].
  - `taskd_fall_penalty(env, asset_cfg, min_height=0.25)` → 1D: -1 if robot root height < min_height, else 0. (Already exists as termination but useful as dense penalty)
  - `taskd_action_smoothness(env)` → 1D: -||action_t - action_{t-1}||². Light penalty for jerky movements.
  Each function must match `RewardTerm.func` signature `(env, **kwargs) -> torch.Tensor`. Must NOT modify existing reward files.
  Parallelization: Wave 1 | Blocked by: — | Blocks: T3, T7 | Can parallelize with: T1, T4
  References:
  - `ATEC2026/source/atec_rl_lab/atec_rl_lab/train/locomotion/velocity/mdp/rewards.py` — existing reward pattern
  - `ATEC2026/source/atec_rl_lab/atec_rl_lab/tasks/task_d/mdp/rewards.py` — RewardCrossX, RewardBoxXInRange for box state access
  - `ATEC2026/source/atec_rl_lab/atec_rl_lab/train/locomotion/velocity/config/quadruped/unitree_b2/rough_env_cfg.py:77-141` — reward weight examples
  Acceptance criteria:
  - `python -c "from atec_rl_lab.train.locomotion.velocity.mdp.taskd_rewards import *; print('OK')"` succeeds
  - Each function returns tensor of shape [num_envs]
  QA scenarios:
  - Happy: import succeeds, functions are callable
  - Evidence: .omo/evidence/task-2-taskd-finetune-import.txt
  Commit: N

- [x] 3. Create `taskd_omni_env_cfg.py` — 3 Task D training env configs
  What to do / Must NOT do: Create file at `ATEC2026/source/atec_rl_lab/atec_rl_lab/train/locomotion/velocity/config/quadruped/unitree_b2/taskd_omni_env_cfg.py`. Implement 3 config classes:

  **Base class `TaskDOmniEnvCfg(UnitreeB2WPiperRoughOmniEnvCfg)`**:

  ```python
  from copy import deepcopy
  from atec_rl_lab.tasks.task_d.terrain import TASK_D_TERRAIN_CFG, PitAndPlatformTerrainCfg
  from isaaclab.assets import RigidObjectCfg
  import isaaclab.sim as sim_utils
  from isaaclab.sensors import MultiMeshRayCasterCfg
  from isaaclab.managers import ObservationTermCfg as ObsTerm
  from isaaclab.managers import SceneEntityCfg
  import atec_rl_lab.train.locomotion.velocity.mdp as mdp

  class TaskDOmniEnvCfg(UnitreeB2WPiperRoughOmniEnvCfg):
      pit_width_range: tuple = (1.3, 1.4)
      platform_height_range: tuple = (1.0, 1.2)

      def _build_terrain_cfg(self):
          """Override parent rough terrain with Task D pit-and-platform terrain."""
          terrain_cfg = deepcopy(TASK_D_TERRAIN_CFG)
          pit_cfg = terrain_cfg.terrain_generator.sub_terrains.get("pit_and_platform")
          if isinstance(pit_cfg, PitAndPlatformTerrainCfg):
              pit_cfg.pit_width_range = self.pit_width_range
              pit_cfg.platform_height_range = self.platform_height_range
          return terrain_cfg

      def __post_init__(self):
          super().__post_init__()  # parent calls disable_zero_weight_rewards()

          # Override terrain: rough → pit-and-platform
          self.scene.terrain = self._build_terrain_cfg()

          # Add box to scene
          self.scene.box = RigidObjectCfg(
              prim_path="{ENV_REGEX_NS}/Box",
              spawn=sim_utils.CuboidCfg(
                  size=(0.8, 1.0, 0.6),
                  rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
                  collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                  mass_props=sim_utils.MassPropertiesCfg(mass=8.0),
                  physics_material=sim_utils.RigidBodyMaterialCfg(
                      static_friction=0.9, dynamic_friction=0.8, restitution=0.0),
              ),
              init_state=RigidObjectCfg.InitialStateCfg(pos=(-3, 1.6, 0.5)),
          )

          # Modify LiDAR to detect box
          if self.scene.height_scanner is not None:
              self.scene.height_scanner = MultiMeshRayCasterCfg(
                  prim_path=self.scene.height_scanner.prim_path,
                  update_period=self.scene.height_scanner.update_period,
                  pattern_cfg=self.scene.height_scanner.pattern_cfg,
                  max_distance=self.scene.height_scanner.max_distance,
                  debug_vis=False,
                  offset=self.scene.height_scanner.offset,
                  mesh_prim_paths=[
                      "/World/ground",
                      MultiMeshRayCasterCfg.RaycastTargetCfg(
                          prim_expr="{ENV_REGEX_NS}/Box",
                          is_shared=True,
                          track_mesh_transforms=True,
                      ),
                  ],
              )

          # Add 8 new observation terms (total 53 + 8 = 61D)
          self.observations.policy.taskd_score_norm = ObsTerm(
              func=mdp.taskd_score_norm,
              params={"asset_cfg": SceneEntityCfg("robot"), "box_cfg": SceneEntityCfg("box"), "max_score": 36.0},
          )
          self.observations.policy.taskd_stage_onehot = ObsTerm(
              func=mdp.taskd_stage_onehot,
              params={"asset_cfg": SceneEntityCfg("robot"), "box_cfg": SceneEntityCfg("box"),
                      "thresholds": [1.9, 15.0, 21.0]},
          )
          self.observations.policy.taskd_box_detected = ObsTerm(
              func=mdp.taskd_box_detected,
              params={"sensor_cfg": SceneEntityCfg("height_scanner"), "box_cfg": SceneEntityCfg("box")},
          )
          self.observations.policy.taskd_box_bearing = ObsTerm(
              func=mdp.taskd_box_bearing,
              params={"sensor_cfg": SceneEntityCfg("height_scanner"), "box_cfg": SceneEntityCfg("box")},
          )
          self.observations.policy.taskd_box_distance_norm = ObsTerm(
              func=mdp.taskd_box_distance_norm,
              params={"sensor_cfg": SceneEntityCfg("height_scanner"), "box_cfg": SceneEntityCfg("box"), "max_dist": 5.0},
          )

          # Replace velocity-tracking rewards with Task D dense rewards
          # (these survived disable_zero_weight_rewards because weight > 0)
          self.rewards.track_lin_vel_xy_exp.weight = 0
          self.rewards.track_ang_vel_z_exp.weight = 0
          self.rewards.box_x_progress.weight = 2.0
          self.rewards.stage_progress.weight = 1.5
          self.rewards.alignment_with_box.weight = 0.5
          self.rewards.taskd_fall_penalty.weight = -5.0
          self.rewards.taskd_action_smoothness.weight = -0.01
          # Keep existing penalties (lin_vel_z_l2, ang_vel_xy_l2, joint_torques, etc.)

          # Disable curriculum
          self.curriculum.command_levels_lin_vel = None
          self.curriculum.command_levels_ang_vel = None
  ```

  **Key design: Pseudo-score computation** (CRITICAL — fixes Momus blocker #3):
  The `taskd_score_norm` and `taskd_stage_onehot` observation terms compute a **pseudo-score** directly from env state (robot x-position + box x-position), NOT from a `current_score` variable. The score is computed as:
  - +2.0 if robot root x > -1.4 (crossed first threshold)
  - +14.0 if box x ∈ [-1.4, 0.7] (box in target zone)
  - +20.0 if robot root x > 2.0 (crossed second threshold)
  - Max = 36.0

  The stage thresholds are: score < 1.9 → stage 0 (APPROACH), score < 15.0 → stage 1 (PUSH_BOX), score < 21.0 → stage 2 (NAV_PLATFORM), else → stage 3 (CLIMB_FINISH). This matches solution.py's state machine exactly.

  For the `_reward_given` tracking (to avoid re-counting the same threshold), the observation term uses a persistent buffer in the env (similar to RewardCrossX pattern in `task_d/mdp/rewards.py:126-133`).

  **`TaskDOmniEnvEasyCfg(TaskDOmniEnvCfg)`**:
  - `pit_width_range = (0.8, 0.9)` (narrow pit)
  - `platform_height_range = (0.5, 0.6)` (low platform)
  - Box mass: 5.0 kg (lighter)

  **`TaskDOmniEnvMediumCfg(TaskDOmniEnvCfg)`**:
  - `pit_width_range = (1.0, 1.1)` (medium pit)
  - `platform_height_range = (0.7, 0.8)` (medium platform)
  - Box mass: 6.5 kg

  **`TaskDOmniEnvOfficialCfg(TaskDOmniEnvCfg)`**:
  - `pit_width_range = (1.3, 1.4)` (official pit)
  - `platform_height_range = (1.0, 1.2)` (official platform)
  - Box mass: 8.0 kg (official)

  Must NOT modify `tasks/task_d/env_cfg.py`. Must NOT change action space (keep 16D: 12 legs + 4 wheels).
  Parallelization: Wave 1 | Blocked by: T1, T2 | Blocks: T5, T7 | Can parallelize with: T4
  References:
  - `ATEC2026/source/atec_rl_lab/atec_rl_lab/train/locomotion/velocity/config/quadruped/unitree_b2/b2w_omni_env_cfg.py` — parent class to extend
  - `ATEC2026/source/atec_rl_lab/atec_rl_lab/tasks/task_d/env_cfg.py:59-135` — TaskDEnvCfg for box/terrain/reward reference
  - `ATEC2026/source/atec_rl_lab/atec_rl_lab/tasks/task_d/terrain.py` — PitAndPlatformTerrainCfg parameters
  Acceptance criteria:
  - `python -c "from atec_rl_lab.train.locomotion.velocity.config.quadruped.unitree_b2.taskd_omni_env_cfg import TaskDOmniEnvEasyCfg, TaskDOmniEnvMediumCfg, TaskDOmniEnvOfficialCfg; print('OK')"` succeeds
  - Each config has 61D observation space (verify by checking len of observation group terms)
  - Each config has 16D action space (12 leg + 4 wheel)
  QA scenarios:
  - Happy: import succeeds, configs instantiate
  - Failure: missing box import → ImportError
  - Evidence: .omo/evidence/task-3-taskd-finetune-configs.txt
  Commit: N

- [x] 4. Create `agents/rsl_rl_ppo_taskd_cfg.py` — PPO runner config
  What to do / Must NOT do: Create file at `ATEC2026/source/atec_rl_lab/atec_rl_lab/train/locomotion/velocity/config/quadruped/unitree_b2/agents/rsl_rl_ppo_taskd_cfg.py`. Implement 3 runner configs:

  **Base `TaskDOmniPPORunnerCfg(UnitreeB2RoughPPORunnerCfg)`**:
  - `max_iterations = 12000`
  - `save_interval = 100`
  - `experiment_name = "unitree_b2w_taskd"`
  - Policy: inherit [512,256,128] + ELU from parent
  - Algorithm: inherit PPO params from parent

  **`TaskDOmniEasyPPORunnerCfg(TaskDOmniPPORunnerCfg)`**:
  - `max_iterations = 2500`
  - `experiment_name = "unitree_b2w_taskd_easy"`

  **`TaskDOmniMediumPPORunnerCfg(TaskDOmniPPORunnerCfg)`**:
  - `max_iterations = 4000`
  - `experiment_name = "unitree_b2w_taskd_medium"`

  **`TaskDOmniOfficialPPORunnerCfg(TaskDOmniPPORunnerCfg)`**:
  - `max_iterations = 7000`
  - `experiment_name = "unitree_b2w_taskd_official"`

  Must NOT change actor architecture (keep [512,256,128]).
  Parallelization: Wave 1 | Blocked by: — | Blocks: T5 | Can parallelize with: T1, T2, T3
  References:
  - `ATEC2026/source/atec_rl_lab/atec_rl_lab/train/locomotion/velocity/config/quadruped/unitree_b2/agents/rsl_rl_ppo_omni_cfg.py` — parent config to extend
  - `ATEC2026/source/atec_rl_lab/atec_rl_lab/train/locomotion/velocity/config/quadruped/unitree_b2/agents/rsl_rl_ppo_cfg.py` — base PPO params
  Acceptance criteria:
  - `python -c "from atec_rl_lab.train.locomotion.velocity.config.quadruped.unitree_b2.agents.rsl_rl_ppo_taskd_cfg import *; print('OK')"` succeeds
  - Each config has correct experiment_name and max_iterations
  QA scenarios:
  - Happy: import succeeds
  - Evidence: .omo/evidence/task-4-taskd-finetune-runner.txt
  Commit: N

### Wave 2: Integration (parallel where possible)

- [x] 5. Register 3 new Task D training envs
  What to do / Must NOT do: Edit `ATEC2026/source/atec_rl_lab/atec_rl_lab/train/locomotion/velocity/config/quadruped/unitree_b2/__init__.py`. Add 3 new `gym.register()` calls AFTER the existing 4:

  ```python
  gym.register(
      id="ATEC-Isaac-TaskD-FixedArm-B2W-Easy-v0",
      entry_point="isaaclab.envs:ManagerBasedRLEnv",
      disable_env_checker=True,
      kwargs={
          "env_cfg_entry_point": f"{__name__}.taskd_omni_env_cfg:TaskDOmniEnvEasyCfg",
          "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_taskd_cfg:TaskDOmniEasyPPORunnerCfg",
      },
  )
  ```

  Repeat for Medium and Official. Must NOT modify existing 4 registrations.
  Parallelization: Wave 2 | Blocked by: T3, T4 | Blocks: T9, T11 | Can parallelize with: T6, T7, T8
  References:
  - `ATEC2026/source/atec_rl_lab/atec_rl_lab/train/locomotion/velocity/config/quadruped/unitree_b2/__init__.py` — existing 4 registrations
  Acceptance criteria:
  - `python scripts/list_envs.py 2>/dev/null | grep TaskD` shows 3 new IDs
  - `python -c "import atec_rl_lab.train; import gymnasium as gym; spec = gym.spec('ATEC-Isaac-TaskD-FixedArm-B2W-Easy-v0'); print(spec.id)"` succeeds
  QA scenarios:
  - Happy: list_envs shows new IDs
  - Failure: import error → check __init__.py syntax
  - Evidence: .omo/evidence/task-5-taskd-finetune-registration.txt
  Commit: Y | feat(train): register 3 Task D fine-tuning envs

- [x] 6. Add `_segmented_col_transfer_53_61` to train.py
  What to do / Must NOT do: Edit `ATEC2026/scripts/rsl_rl/train.py`. Add new function after `_segmented_col_transfer_45_53`:

  ```python
  def _segmented_col_transfer_53_61(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
      """Map B2W Omni 53-D observation columns to Task D 61-D columns.

      Source (53)                    Target (61)
      [0:53]  omni obs (identical)  →  [0:53]  copy all
                                   →  [53:54] score_norm          init random
                                   →  [54:58] stage_one_hot       init random
                                   →  [58:59] box_detected        init random
                                   →  [59:60] box_bearing         init random
                                   →  [60:61] box_distance_norm   init random
      """
      merged = target.clone()
      merged[:, :53] = source[:, :53]
      merged[:, 53:61] = torch.randn(source.shape[0], 8, device=merged.device, dtype=merged.dtype) * 0.01
      return merged
  ```

  Also update `_load_actor_weights_only` to detect 53→61 input expansion and call the new function (add clause after the 45→53 check).

  Must NOT modify the existing `_segmented_col_transfer_45_53` function.
  Parallelization: Wave 2 | Blocked by: — | Blocks: T9 | Can parallelize with: T5, T7, T8
  References:
  - `ATEC2026/scripts/rsl_rl/train.py:117-143` — existing _segmented_col_transfer_45_53 pattern
  - `ATEC2026/scripts/rsl_rl/train.py:222-253` — input expansion detection in _load_actor_weights_only
  Acceptance criteria:
  - `_segmented_col_transfer_53_61` exists and handles 53→61 case
  - `_load_actor_weights_only` detects 53→61 and calls the new function
  - Existing 45→53 path still works (no regression)
  QA scenarios:
  - Happy: function exists and is callable
  - Regression: existing 45→53 still works
  - Evidence: .omo/evidence/task-6-taskd-finetune-warmstart.txt
  Commit: Y | feat(train): add 53→61 obs expansion for Task D fine-tuning

- [x] 7. Update `mdp/__init__.py` imports
  What to do / Must NOT do: Edit `ATEC2026/source/atec_rl_lab/atec_rl_lab/train/locomotion/velocity/mdp/__init__.py`. Add imports for the new modules:

  ```python
  from .taskd_observations import *  # noqa: F401
  from .taskd_rewards import *  # noqa: F401
  ```

  Must NOT modify existing imports. Must NOT remove any existing lines.
  Parallelization: Wave 2 | Blocked by: T1, T2 | Blocks: T3, T5 | Can parallelize with: T5, T6, T8
  References:
  - `ATEC2026/source/atec_rl_lab/atec_rl_lab/train/locomotion/velocity/mdp/__init__.py` — existing imports
  Acceptance criteria:
  - `python -c "import atec_rl_lab.train.locomotion.velocity.mdp; print(dir(atec_rl_lab.train.locomotion.velocity.mdp))"` includes `taskd_score_norm`
  QA scenarios:
  - Happy: import succeeds, new functions accessible
  - Evidence: .omo/evidence/task-7-taskd-finetune-mdp-imports.txt
  Commit: N (part of T5 commit)

### Wave 3: Inference

- [x] 8. Add `b2w_taskd61` mode to solution.py
  What to do / Must NOT do: Edit `ATEC2026/demo/solution.py`. Add new policy mode:

  1. In `__init__`, detect `ATEC_POLICY_MODE == "b2w_taskd61"`:
     - Set `self.policy_out_dim = 16` (same as omni16)
     - Set `self._policy_mode = "b2w_taskd61"`

  2. Add `_extract_policy_obs_taskd61(self, obs, action_dim, current_score)` method (note: takes `current_score` from `predicts()`):
     - Same 53D base as `_extract_policy_obs_omni16` (indices 0-52)
     - Append 8D Task D features:
       - `score_norm = current_score / 36.0` (clamped [0,1])
       - `stage_one_hot`: 4D one-hot based on score thresholds [1.9, 15.0, 21.0]
       - `box_detected`: from `_parse_lidar(extero)` → 1.0 or 0.0
       - `box_bearing`: from `_parse_lidar(extero)` → normalized to [-1,1]
       - `box_distance_norm`: from `_parse_lidar(extero)` → distance/5.0, clamped [0,1]
     - Total: 61D

  3. In `predicts()`, add branch for `b2w_taskd61`:
     - Call `_extract_policy_obs_taskd61(obs, action_dim)` instead of `_extract_policy_obs_omni16`
     - Action mapping: same as omni16 (16D → 24D with arms=0)

  Must NOT modify existing omni16 or legacy 12D code paths.
  Parallelization: Wave 3 | Blocked by: T1 | Blocks: T15 | Can parallelize with: T5, T6, T7
  References:
  - `ATEC2026/demo/solution.py:40-45` — existing policy mode detection
  - `ATEC2026/demo/solution.py:218-280` — _extract_policy_obs_omni16 (53D layout)
  - `ATEC2026/demo/solution.py:338-364` — _parse_lidar output format
  - `ATEC2026/demo/solution.py:282-335` — _map_policy_action_to_env_action
  Acceptance criteria:
  - `ATEC_POLICY_MODE=b2w_taskd61` is recognized by AlgSolution.__init__
  - `_extract_policy_obs_taskd61` returns [1, 61] tensor
  - `_map_policy_action_to_env_action` returns [1, 24] with arms=0
  QA scenarios:
  - Happy: AlgSolution instantiates with b2w_taskd61 mode
  - Failure: wrong mode string → ValueError
  - Evidence: .omo/evidence/task-8-taskd-finetune-solution.txt
  Commit: Y | feat(solution): add b2w_taskd61 inference mode for 61D Task D policy

### Wave 4: Scripts

- [x] 9. Create `train_taskd_finetune.sh` training script
  What to do / Must NOT do: Create `ATEC2026/scripts/training/train_taskd_finetune.sh`. Structure:

  ```bash
  #!/usr/bin/env bash
  set -eo pipefail
  cd /home/1ctnltug/atec2026/ATEC2026
  source /home/1ctnltug/atec2026/scripts/env/activate_atec2026_sim.sh

  STAGE="${1:-easy}"  # easy | medium | official
  MAX_ITERS="${ATEC_TASKD_ITERS:-}"
  NUM_ENVS="${ATEC_TRAIN_NUM_ENVS:-1024}"

  case "$STAGE" in
    easy)    TASK="ATEC-Isaac-TaskD-FixedArm-B2W-Easy-v0";    EXP="unitree_b2w_taskd_easy";    PREV_EXP="";                        DEFAULT_ITERS=2500 ;;
    medium)  TASK="ATEC-Isaac-TaskD-FixedArm-B2W-Medium-v0";  EXP="unitree_b2w_taskd_medium";  PREV_EXP="unitree_b2w_taskd_easy";  DEFAULT_ITERS=4000 ;;
    official) TASK="ATEC-Isaac-TaskD-FixedArm-B2W-Official-v0"; EXP="unitree_b2w_taskd_official"; PREV_EXP="unitree_b2w_taskd_medium"; DEFAULT_ITERS=7000 ;;
    *) echo "Usage: $0 [easy|medium|official]"; exit 1 ;;
  esac

  MAX_ITERS="${MAX_ITERS:-$DEFAULT_ITERS}"

  # Find warm-start checkpoint
  if [[ "$STAGE" == "easy" ]]; then
    SOURCE_CKPT="demo/policy_taskd_omni.pt"
  else
    # Look in PREVIOUS stage's directory, not current
    SOURCE_CKPT="$(find "logs/rsl_rl/$PREV_EXP" -maxdepth 2 -type f -name 'model_*.pt' | sort -V | tail -1)"
  fi

  if [[ -z "$SOURCE_CKPT" || ! -f "$SOURCE_CKPT" ]]; then
    echo "ERROR: Warm-start checkpoint not found for stage '$STAGE'"
    if [[ -n "$PREV_EXP" ]]; then
      echo "  Expected in: logs/rsl_rl/$PREV_EXP/"
      echo "  Train the previous stage first."
    else
      echo "  Expected: demo/policy_taskd_omni.pt"
    fi
    exit 1
  fi

  echo "Stage: $STAGE | Iters: $MAX_ITERS | Envs: $NUM_ENVS"
  echo "Warm-start from: $SOURCE_CKPT"

  python scripts/rsl_rl/train.py \
    --task "$TASK" \
    --headless --enable_cameras --disable_fabric \
    --num_envs "$NUM_ENVS" \
    --max_iterations "$MAX_ITERS" \
    --actor_checkpoint "$SOURCE_CKPT" \
    --run_name "from_${STAGE}"
  ```

  Must NOT hardcode checkpoint paths. Must support stage selection via argument.
  Parallelization: Wave 4 | Blocked by: T5, T6 | Blocks: T11 | Can parallelize with: T10
  References:
  - `scripts/training/train_b2w_rough_omni_from_straight.sh` — existing pattern
  - `scripts/training/train_b2_rough_straight_from_flat.sh` — existing pattern
  Acceptance criteria:
  - Script is executable, handles all 3 stages
  - `bash scripts/training/train_taskd_finetune.sh easy --help` doesn't crash
  QA scenarios:
  - Happy: script runs, finds checkpoint, starts training
  - Failure: missing checkpoint → clear error message
  - Evidence: .omo/evidence/task-9-taskd-finetune-train-script.txt
  Commit: Y | feat(training): add Task D fine-tuning script

- [x] 10. Create `export_taskd_finetune_policy.sh` export script
  What to do / Must NOT do: Create `ATEC2026/scripts/training/export_taskd_finetune_policy.sh`. Structure:

  ```bash
  #!/usr/bin/env bash
  set -eo pipefail
  cd /home/1ctnltug/atec2026/ATEC2026
  source /home/1ctnltug/atec2026/scripts/env/activate_atec2026_sim.sh

  STAGE="${1:-official}"
  EXP="unitree_b2w_taskd_${STAGE}"

  LATEST_CKPT="$(find "logs/rsl_rl/$EXP" -name 'model_*.pt' | sort -V | tail -1)"
  # ... standard export flow (play.py --video --video_length 2, then copy)
  TARGET="demo/policy_taskd_finetuned.pt"
  ```

  Parallelization: Wave 4 | Blocked by: T5 | Blocks: T12-T14 | Can parallelize with: T9
  References:
  - `scripts/training/export_latest_b2w_omni_policy_to_demo.sh` — existing pattern
  Acceptance criteria:
  - Script is executable, finds checkpoint, exports to demo/
  - Evidence: .omo/evidence/task-10-taskd-finetune-export-script.txt
  Commit: Y | feat(training): add Task D policy export script

### Wave 5: Testing

- [x] 11. Smoke test (64 envs, 10 iters)
  What to do / Must NOT do: Run:
  ```bash
  ATEC_TASKD_ITERS=10 ATEC_TRAIN_NUM_ENVS=64 ./scripts/training/train_taskd_finetune.sh easy
  ```
  Verify:
  - Training starts without shape mismatch errors
  - Observation shape is [64, 61]
  - Action shape is [64, 16]
  - At least 1 checkpoint saved (model_0.pt)
  - No crashes in first 10 iterations

  Parallelization: Wave 5 | Blocked by: T9 | Blocks: T12 | Can parallelize with: T10
  References:
  - `scripts/training/train_b2w_rough_omni_from_straight.sh` — existing smoke test pattern
  Acceptance criteria:
  - Training completes 10 iterations without error
  - Checkpoint file exists in logs/rsl_rl/unitree_b2w_taskd_easy/
  QA scenarios:
  - Happy: training runs, checkpoints saved
  - Failure: shape mismatch → check _segmented_col_transfer_53_61
  - Evidence: .omo/evidence/task-11-taskd-finetune-smoke.txt
  Commit: N

- [ ] 12. Easy stage training (1024 envs, 1500-2500 iters)
  What to do / Must NOT do: Run:
  ```bash
  ATEC_TASKD_ITERS=2500 ATEC_TRAIN_NUM_ENVS=1024 ./scripts/training/train_taskd_finetune.sh easy
  ```
  Monitor:
  - Average return increasing over iterations
  - Box x-progression reward increasing
  - No excessive falls (fall penalty should decrease)
  Export checkpoint after completion.

  Parallelization: Wave 5 | Blocked by: T11 | Blocks: T13 | Can parallelize with: T10
  Acceptance criteria:
  - Training completes 2500 iterations
  - Average return > 0 (positive learning signal)
  - Checkpoint exported to demo/policy_taskd_finetuned.pt
  Evidence: .omo/evidence/task-12-taskd-finetune-easy.txt
  Commit: N

- [ ] 13. Medium stage training (1024 envs, 2500-4000 iters)
  What to do / Must NOT do: Run:
  ```bash
  ATEC_TASKD_ITERS=4000 ATEC_TRAIN_NUM_ENVS=1024 ./scripts/training/train_taskd_finetune.sh medium
  ```
  Warm-start from Easy stage checkpoint.
  Parallelization: Wave 5 | Blocked by: T12 | Blocks: T14
  Acceptance criteria:
  - Training completes 4000 iterations
  - Robot crosses x=-1.4 threshold more frequently than Easy stage
  Evidence: .omo/evidence/task-13-taskd-finetune-medium.txt
  Commit: N

- [ ] 14. Official stage training (1024-2048 envs, 4000-7000 iters)
  What to do / Must NOT do: Run:
  ```bash
  ATEC_TASKD_ITERS=7000 ATEC_TRAIN_NUM_ENVS=2048 ./scripts/training/train_taskd_finetune.sh official
  ```
  Warm-start from Medium stage checkpoint.
  Parallelization: Wave 5 | Blocked by: T13 | Blocks: T15
  Acceptance criteria:
  - Training completes 7000 iterations
  - Robot reaches x=2.0 threshold in some episodes
  - Box pushed into target zone in some episodes
  Evidence: .omo/evidence/task-14-taskd-finetune-official.txt
  Commit: N

- [ ] 15. Headless evaluation on official Task D env
  What to do / Must NOT do: Run:
  ```bash
  ATEC_POLICY_MODE=b2w_taskd61 ATEC_POLICY_PATH=demo/policy_taskd_finetuned.pt \
    python scripts/play_atec_task.py --task ATEC-TaskD-B2wPiper \
    --headless --enable_cameras --disable_fabric --num_envs 4 --debug
  ```
  Record video:
  ```bash
  ATEC_POLICY_MODE=b2w_taskd61 ATEC_POLICY_PATH=demo/policy_taskd_finetuned.pt \
    ./scripts/task_d/record_task_d_b2w_video.sh
  ```
  Verify:
  - Policy loads without errors
  - Robot walks and approaches box
  - Box moves forward when pushed
  - No premature falls
  - Score > 0 (at least crosses x=-1.4)

  Parallelization: Wave 5 | Blocked by: T8, T14 | Blocks: —
  References:
  - `scripts/task_d/record_task_d_b2w_video.sh` — existing recording script
  - `ATEC2026/demo/solution.py` — inference entry point
  Acceptance criteria:
  - Headless run completes without crash
  - Video recorded successfully
  - Score > 0 in at least 1 of 4 envs
  Evidence: .omo/evidence/task-15-taskd-finetune-eval.txt
  Commit: N

## Final verification wave
> Runs in parallel after ALL todos. ALL must APPROVE. Surface results and wait for the user's explicit okay before declaring complete.
- [ ] F1. Plan compliance audit — verify all 15 todos completed, no modifications to tasks/task_d/
- [ ] F2. Code quality review — grep for TODO/FIXME in new files, verify no hardcoded paths
- [ ] F3. Real manual QA — verify smoke test passes, eval produces score > 0
- [ ] F4. Scope fidelity — no changes to IsaacLab/, no changes to existing omni env

## Commit strategy
- Wave 1-2: Single commit `feat(train): add Task D fine-tuning envs, obs, rewards, and warm-start`
- Wave 3: Single commit `feat(solution): add b2w_taskd61 inference mode`
- Wave 4: Single commit `feat(training): add Task D fine-tuning and export scripts`
- Wave 5: No commits (training runs)

## Success criteria
1. `python scripts/list_envs.py` shows 3 new Task D training envs
2. Smoke test (64 envs, 10 iters) completes without shape errors
3. Easy stage training shows positive learning signal (return > 0)
4. Official stage training produces checkpoint that scores > 0 on official Task D eval
5. solution.py with `b2w_taskd61` mode produces valid 24D actions with arms=0
6. No modifications to `tasks/task_d/` or `IsaacLab/`
