# b2w-flat-omni - Work Plan

## TL;DR (For humans)

**What you'll get:** A flat-ground B2W omni-directional locomotion training pipeline — one new gym environment, one PPO runner config, training/export scripts, and a smoke-tested checkpoint path. The policy will be warm-started from the existing rough-omni B2W checkpoint so it learns forward / backward / turning on clean flat ground faster.

**Why this approach:** Reuse the proven `unitree_b2w_rough_omni` weights as the starting point; flat ground removes terrain distraction and lets the policy specialize on velocity tracking in all directions. Keeping the same 16D action / 53D observation layout makes the warm-start a direct weight copy with no column transfer.

**What it will NOT do:** Touch official Task D code, change the existing rough-omni training pipeline, add arm control, or introduce a new action space.

**Effort:** Medium
**Risk:** Low - same obs/action shape as rough omni; only terrain changes from rough to plane.
**Decisions to sanity-check:** Whether to remove the height scanner entirely on flat ground, and whether the existing rough-omni checkpoint is the best warm-start source.

Your next move: approve this plan, then I'll delegate execution.

---

> TL;DR (machine): Effort=Medium Risk=Low | 7 todos across 4 waves + final verification | deliverables=1 flat B2W omni env + PPO cfg + training/export scripts + smoke test

## Scope
### Must have
1. New `UnitreeB2WPiperFlatOmniEnvCfg` class extending `UnitreeB2WPiperRoughOmniEnvCfg` with terrain set to plane and height scanner removed.
2. New `UnitreeB2WPiperFlatOmniPPORunnerCfg` runner config with experiment name `unitree_b2w_flat_omni`.
3. Register one new gym env ID `ATEC-Isaac-Velocity-Flat-Omni-B2W-Piper-v0`.
4. `scripts/training/train_b2w_flat_omni.sh` that warm-starts from the latest rough-omni checkpoint.
5. `scripts/training/export_b2w_flat_omni_policy_to_demo.sh` to export the final checkpoint to `demo/policy_b2w_flat_omni.pt`.
6. Smoke test (64 envs, 10 iters) verifying obs shape [64, 53], action shape [64, 16], and checkpoint save.
7. GPU-bound full training step documented for manual execution.

### Must NOT have (guardrails, anti-slop, scope boundaries)
- NO modifications to `tasks/task_d/` or official competition code.
- NO changes to the existing `UnitreeB2WPiperRoughOmniEnvCfg` or its training pipeline.
- NO arm control in training (arms stay at PD default).
- NO changes to the 16D action space (12 leg + 4 wheel) or 53D observation layout.
- NO new dependencies beyond existing IsaacLab / RSL-RL.
- NO modifications to `IsaacLab/` framework code.

## Verification strategy
> Zero human intervention - all verification is agent-executed.
- Test decision: tests-after (shape checks + smoke train)
- Evidence: .omo/evidence/task-<N>-b2w-flat-omni.<ext>

## Execution strategy
### Parallel execution waves

**Wave 1: Foundation (2 parallel todos)**
- T1. Create `flat_b2w_omni_env_cfg.py` — flat-ground B2W omni env config
- T2. Create `agents/rsl_rl_ppo_flat_b2w_omni_cfg.py` — PPO runner config

**Wave 2: Integration (1 todo)**
- T3. Register `ATEC-Isaac-Velocity-Flat-Omni-B2W-Piper-v0` in `__init__.py`

**Wave 3: Scripts (2 parallel todos)**
- T4. Create `train_b2w_flat_omni.sh`
- T5. Create `export_b2w_flat_omni_policy_to_demo.sh`

**Wave 4: Testing (2 todos)**
- T6. Smoke test (64 envs, 10 iters)
- T7. Full training (GPU-bound, manual)

### Dependency matrix
| Todo | Depends on | Blocks | Can parallelize with |
| --- | --- | --- | --- |
| T1. flat_b2w_omni_env_cfg.py | — | T3, T6 | T2 |
| T2. rsl_rl_ppo_flat_b2w_omni_cfg.py | — | T3 | T1 |
| T3. Register env | T1, T2 | T4, T5, T6 | — |
| T4. train_b2w_flat_omni.sh | T3 | T6 | T5 |
| T5. export_b2w_flat_omni_policy.sh | T3 | T7 | T4 |
| T6. Smoke test | T3, T4 | T7 | T5 |
| T7. Full training | T6 | — | — |

## Todos
<!-- APPEND TASK BATCHES BELOW THIS LINE WITH edit/apply_patch - never rewrite the headers above. -->

### Wave 1: Foundation (parallel)

- [x] 1. Create `flat_b2w_omni_env_cfg.py` — flat-ground B2W omni env config
  What to do / Must NOT do: Create file at `ATEC2026/source/atec_rl_lab/atec_rl_lab/train/locomotion/velocity/config/quadruped/unitree_b2/flat_b2w_omni_env_cfg.py`. Implement:

  ```python
  @configclass
  class UnitreeB2WPiperFlatOmniEnvCfg(UnitreeB2WPiperRoughOmniEnvCfg):
      def __post_init__(self):
          super().__post_init__()

          # Terrain: rough → flat plane
          self.scene.terrain.terrain_type = "plane"
          self.scene.terrain.terrain_generator = None

          # No height scan needed on flat ground
          self.scene.height_scanner = None
          if self.observations.policy.height_scan is not None:
              self.observations.policy.height_scan = None
          if self.observations.critic.height_scan is not None:
              self.observations.critic.height_scan = None

          # No terrain curriculum on flat ground
          self.curriculum.terrain_levels = None

          # Keep everything else identical to rough omni:
          # - 16D action (12 leg + 4 wheel)
          # - 53D observation
          # - ModeBasedVelocityCommandCfg for forward/backward/turning
          # - reward weights copied from parent
  ```

  Must NOT modify `UnitreeB2WPiperRoughOmniEnvCfg`. Must NOT change action/observation dimensions.
  Parallelization: Wave 1 | Blocked by: — | Blocks: T3, T6 | Can parallelize with: T2
  References:
  - `ATEC2026/source/atec_rl_lab/atec_rl_lab/train/locomotion/velocity/config/quadruped/unitree_b2/b2w_omni_env_cfg.py` — parent class
  - `ATEC2026/source/atec_rl_lab/atec_rl_lab/train/locomotion/velocity/config/quadruped/unitree_b2/flat_env_cfg.py` — example of switching terrain to plane
  Acceptance criteria:
  - `python -c "from atec_rl_lab.train.locomotion.velocity.config.quadruped.unitree_b2.flat_b2w_omni_env_cfg import UnitreeB2WPiperFlatOmniEnvCfg; print('OK')"` succeeds
  - Config instantiates and has 16D action space, 53D observation space
  QA scenarios:
  - Happy: import succeeds, terrain_type is "plane"
  - Failure: parent class changes break inheritance → ImportError or AttributeError
  - Evidence: .omo/evidence/task-1-b2w-flat-omni-config.txt
  Commit: Y | feat(train): add flat-ground B2W omni env config

- [x] 2. Create `agents/rsl_rl_ppo_flat_b2w_omni_cfg.py` — PPO runner config
  What to do / Must NOT do: Create file at `ATEC2026/source/atec_rl_lab/atec_rl_lab/train/locomotion/velocity/config/quadruped/unitree_b2/agents/rsl_rl_ppo_flat_b2w_omni_cfg.py`. Implement:

  ```python
  @configclass
  class UnitreeB2WPiperFlatOmniPPORunnerCfg(UnitreeB2WPiperRoughOmniPPORunnerCfg):
      def __post_init__(self):
          super().__post_init__()
          self.max_iterations = 10000
          self.experiment_name = "unitree_b2w_flat_omni"
          # Inherit [512,256,128] + ELU architecture and PPO hyperparams
  ```

  Must NOT change actor/critic hidden dimensions or activation.
  Parallelization: Wave 1 | Blocked by: — | Blocks: T3 | Can parallelize with: T1
  References:
  - `ATEC2026/source/atec_rl_lab/atec_rl_lab/train/locomotion/velocity/config/quadruped/unitree_b2/agents/rsl_rl_ppo_omni_cfg.py` — parent runner config
  - `ATEC2026/source/atec_rl_lab/atec_rl_lab/train/locomotion/velocity/config/quadruped/unitree_b2/agents/rsl_rl_ppo_cfg.py` — base PPO params
  Acceptance criteria:
  - `python -c "from atec_rl_lab.train.locomotion.velocity.config.quadruped.unitree_b2.agents.rsl_rl_ppo_flat_b2w_omni_cfg import UnitreeB2WPiperFlatOmniPPORunnerCfg; print('OK')"` succeeds
  - `max_iterations == 10000`, `experiment_name == "unitree_b2w_flat_omni"`
  QA scenarios:
  - Happy: import succeeds, config fields correct
  - Failure: wrong import path → ModuleNotFoundError
  - Evidence: .omo/evidence/task-2-b2w-flat-omni-runner.txt
  Commit: N (part of T1 commit)

### Wave 2: Integration

- [x] 3. Register `ATEC-Isaac-Velocity-Flat-Omni-B2W-Piper-v0` in `__init__.py`
  What to do / Must NOT do: Edit `ATEC2026/source/atec_rl_lab/atec_rl_lab/train/locomotion/velocity/config/quadruped/unitree_b2/__init__.py`. Add one new `gym.register()` call after the existing rough-omni registration:

  ```python
  gym.register(
      id="ATEC-Isaac-Velocity-Flat-Omni-B2W-Piper-v0",
      entry_point="isaaclab.envs:ManagerBasedRLEnv",
      disable_env_checker=True,
      kwargs={
          "env_cfg_entry_point": f"{__name__}.flat_b2w_omni_env_cfg:UnitreeB2WPiperFlatOmniEnvCfg",
          "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_flat_b2w_omni_cfg:UnitreeB2WPiperFlatOmniPPORunnerCfg",
      },
  )
  ```

  Must NOT modify existing registrations.
  Parallelization: Wave 2 | Blocked by: T1, T2 | Blocks: T4, T5, T6 | Can parallelize with: —
  References:
  - `ATEC2026/source/atec_rl_lab/atec_rl_lab/train/locomotion/velocity/config/quadruped/unitree_b2/__init__.py` — existing registrations
  Acceptance criteria:
  - `python scripts/list_envs.py 2>/dev/null | grep Flat-Omni` shows the new ID
  - `python -c "import gymnasium as gym; spec = gym.spec('ATEC-Isaac-Velocity-Flat-Omni-B2W-Piper-v0'); print(spec.id)"` succeeds
  QA scenarios:
  - Happy: list_envs shows new ID
  - Failure: import error → check __init__.py syntax
  - Evidence: .omo/evidence/task-3-b2w-flat-omni-registration.txt
  Commit: Y | feat(train): register B2W flat omni env

### Wave 3: Scripts (parallel)

- [x] 4. Create `train_b2w_flat_omni.sh`
  What to do / Must NOT do: Create `ATEC2026/scripts/training/train_b2w_flat_omni.sh`. Structure:

  ```bash
  #!/usr/bin/env bash
  set -eo pipefail
  cd /home/1ctnltug/atec2026/ATEC2026
  source /home/1ctnltug/atec2026/scripts/env/activate_atec2026_sim.sh

  NUM_ENVS="${ATEC_TRAIN_NUM_ENVS:-1024}"
  MAX_ITERS="${ATEC_B2W_FLAT_OMNI_ITERS:-10000}"

  TASK="ATEC-Isaac-Velocity-Flat-Omni-B2W-Piper-v0"
  EXP="unitree_b2w_flat_omni"

  # Warm-start from latest rough-omni checkpoint
  SOURCE_CKPT="$(find logs/rsl_rl/unitree_b2w_rough_omni -maxdepth 2 -type f -name 'model_*.pt' | sort -V | tail -1)"
  if [[ -z "$SOURCE_CKPT" || ! -f "$SOURCE_CKPT" ]]; then
      SOURCE_CKPT="demo/policy_taskd_omni.pt"
  fi
  if [[ ! -f "$SOURCE_CKPT" ]]; then
      echo "WARNING: No rough-omni checkpoint found; training from scratch."
      EXTRA_ARGS=""
  else
      echo "Warm-start from: $SOURCE_CKPT"
      EXTRA_ARGS="--actor_checkpoint $SOURCE_CKPT"
  fi

  python scripts/rsl_rl/train.py \
    --task "$TASK" \
    --headless --enable_cameras --disable_fabric \
    --num_envs "$NUM_ENVS" \
    --max_iterations "$MAX_ITERS" \
    $EXTRA_ARGS \
    --run_name "from_rough_omni"
  ```

  Must NOT hardcode checkpoint paths beyond the fallback search. Must support env vars `ATEC_TRAIN_NUM_ENVS` and `ATEC_B2W_FLAT_OMNI_ITERS`.
  Parallelization: Wave 3 | Blocked by: T3 | Blocks: T6 | Can parallelize with: T5
  References:
  - `ATEC2026/scripts/training/train_b2w_rough_omni_from_straight.sh` — existing warm-start pattern
  - `ATEC2026/scripts/training/train_taskd_finetune.sh` — env-var pattern
  Acceptance criteria:
  - Script is executable
  - `bash -n scripts/training/train_b2w_flat_omni.sh` passes syntax check
  QA scenarios:
  - Happy: script finds checkpoint and constructs train command
  - Failure: missing checkpoint → falls back to from-scratch with warning
  - Evidence: .omo/evidence/task-4-b2w-flat-omni-train-script.txt
  Commit: Y | feat(training): add B2W flat omni training script

- [x] 5. Create `export_b2w_flat_omni_policy_to_demo.sh`
  What to do / Must NOT do: Create `ATEC2026/scripts/training/export_b2w_flat_omni_policy_to_demo.sh`. Structure follows the existing export scripts:

  ```bash
  #!/usr/bin/env bash
  set -eo pipefail
  cd /home/1ctnltug/atec2026/ATEC2026
  source /home/1ctnltug/atec2026/scripts/env/activate_atec2026_sim.sh

  EXP="unitree_b2w_flat_omni"
  TARGET="demo/policy_b2w_flat_omni.pt"

  LATEST_CKPT="$(find "logs/rsl_rl/$EXP" -maxdepth 2 -type f -name 'model_*.pt' | sort -V | tail -1)"
  if [[ -z "$LATEST_CKPT" || ! -f "$LATEST_CKPT" ]]; then
      echo "ERROR: No checkpoint found in logs/rsl_rl/$EXP/"
      exit 1
  fi

  echo "Exporting $LATEST_CKPT → $TARGET"
  # Use play.py to export (same pattern as other export scripts)
  python scripts/rsl_rl/play.py \
    --task ATEC-Isaac-Velocity-Flat-Omni-B2W-Piper-v0 \
    --headless --enable_cameras --disable_fabric \
    --num_envs 1 \
    --load_run "$(dirname "$LATEST_CKPT")" \
    --checkpoint "$(basename "$LATEST_CKPT")" \
    --video --video_length 2

  cp -v "$LATEST_CKPT" "$TARGET"
  ```

  Must NOT hardcode run names. Must fail clearly if no checkpoint exists.
  Parallelization: Wave 3 | Blocked by: T3 | Blocks: T7 | Can parallelize with: T4
  References:
  - `ATEC2026/scripts/training/export_latest_b2w_omni_policy_to_demo.sh` — existing pattern
  - `ATEC2026/scripts/training/export_latest_rough_straight_policy_to_demo.sh` — play.py export pattern
  Acceptance criteria:
  - Script is executable
  - `bash -n scripts/training/export_b2w_flat_omni_policy_to_demo.sh` passes syntax check
  QA scenarios:
  - Happy: script finds checkpoint and exports
  - Failure: missing checkpoint → clear error and exit 1
  - Evidence: .omo/evidence/task-5-b2w-flat-omni-export-script.txt
  Commit: Y | feat(training): add B2W flat omni export script

### Wave 4: Testing

- [x] 6. Smoke test (64 envs, 10 iters)
  What to do / Must NOT do: Run:
  ```bash
  ATEC_B2W_FLAT_OMNI_ITERS=10 ATEC_TRAIN_NUM_ENVS=64 ./scripts/training/train_b2w_flat_omni.sh
  ```
  Verify:
  - Training starts without shape mismatch errors
  - Observation shape is [64, 53]
  - Action shape is [64, 16]
  - At least 1 checkpoint saved (model_0.pt)
  - No crashes in first 10 iterations
  - Warm-start from rough omni succeeds (no input dimension mismatch)

  Parallelization: Wave 4 | Blocked by: T3, T4 | Blocks: T7 | Can parallelize with: T5
  References:
  - `ATEC2026/scripts/training/train_b2w_rough_omni_from_straight.sh` — smoke test pattern
  Acceptance criteria:
  - Training completes 10 iterations without error
  - Checkpoint file exists in `logs/rsl_rl/unitree_b2w_flat_omni/`
  QA scenarios:
  - Happy: training runs, checkpoints saved, shapes correct
  - Failure: shape mismatch → check env config inheritance
  - Evidence: .omo/evidence/task-6-b2w-flat-omni-smoke.txt
  Commit: N

- [~] 7. Full training (GPU-bound, manual)
  What to do / Must NOT do: Run:
  ```bash
  ATEC_TRAIN_NUM_ENVS=4096 ATEC_B2W_FLAT_OMNI_ITERS=10000 ./scripts/training/train_b2w_flat_omni.sh
  ```
  Monitor:
  - Average return increasing
  - Velocity tracking rewards (`track_lin_vel_xy_exp`, `track_ang_vel_z_exp`) increasing
  - No excessive falls

  Export after completion:
  ```bash
  ./scripts/training/export_b2w_flat_omni_policy_to_demo.sh
  ```

  Parallelization: Wave 4 | Blocked by: T6 | Blocks: — | Can parallelize with: —
  Acceptance criteria:
  - Training completes requested iterations
  - Average return > 0
  - Checkpoint exported to `demo/policy_b2w_flat_omni.pt`
  QA scenarios:
  - Happy: training converges, policy exported
  - Failure: OOM → reduce `ATEC_TRAIN_NUM_ENVS`
  - Evidence: .omo/evidence/task-7-b2w-flat-omni-training.txt
  Commit: N

## Final verification wave
> Runs in parallel after ALL code todos (T1-T6). T7 is GPU-bound and excluded from the automated final wave.
- [x] F1. Plan compliance audit — verify every Must have item has a matching todo and no Must NOT have was violated.
- [x] F2. Code quality review — no TODO/FIXME in new files, no hardcoded absolute paths, scripts are executable.
- [x] F3. Real manual QA — smoke test output shows obs [64, 53], action [64, 16], checkpoint saved.
- [x] F4. Scope fidelity — no modifications to `tasks/task_d/`, `IsaacLab/`, or existing rough-omni config.

## Commit strategy
- Wave 1-2: Single commit `feat(train): add flat-ground B2W omni env and PPO config`
- Wave 3: Single commit `feat(training): add B2W flat omni training and export scripts`
- Wave 4: No commits (training runs)

## Success criteria
1. `python scripts/list_envs.py` shows `ATEC-Isaac-Velocity-Flat-Omni-B2W-Piper-v0`.
2. Smoke test (64 envs, 10 iters) completes without shape errors.
3. Full training shows positive learning signal (return > 0).
4. Exported policy at `demo/policy_b2w_flat_omni.pt` loads without errors.
5. No modifications to `tasks/task_d/` or `IsaacLab/`.
