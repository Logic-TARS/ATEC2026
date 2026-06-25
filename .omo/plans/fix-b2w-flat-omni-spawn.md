# fix-b2w-flat-omni-spawn - Work Plan

## TL;DR (For humans)
<!-- Fill this LAST, after the detailed plan below is written, so it summarizes the REAL plan. -->
<!-- Plain English for a non-engineer: NO file paths, NO todo numbers, NO wave/agent/tool names. -->

**What you'll get:** The B2W flat-omni training environment will always spawn the robot upright at the same 0.78 m height used by Task A and Task D, with heading still randomized. A smoke train, short retrain, and exported checkpoint will verify the fix.

**Why this approach:** The bug is inherited reset randomization from the rough-omni config. We override only the flat-omni config after parent initialization, leaving the rough-omni training distribution untouched. Setting the default root height to 0.78 and zeroing roll/pitch ranges makes `reset_root_state_uniform` produce upright spawns.

**What it will NOT do:** It will not change the rough-omni config, official Task A/Task D envs, IsaacLab, the robot USD, or delete the old `model_199.pt` checkpoint. It will not add new scripts or change action/observation dimensions.

**Effort:** Short
**Risk:** Low - single config file change with deterministic verification.
**Decisions to sanity-check:** Whether zeroing angular-velocity roll/pitch (rather than leaving small jitter) is acceptable for your robustness goals.

Your next move: start work now (`$start-work fix-b2w-flat-omni-spawn`), or run a high-accuracy review first? Full execution detail follows below.

---

> TL;DR (machine): Effort=Short Risk=Low | 6 todos across 3 waves + final verification | deliverables=upright flat-omni spawn config + smoke/short retrain + exported policy

## Scope
### Must have
1. Override `UnitreeB2WPiperFlatOmniEnvCfg` reset/spawn settings in `flat_b2w_omni_env_cfg.py` after `super().__post_init__()`.
2. Set the B2W+Piper robot's default root position to `z=0.78` (matching Task A/Task D B2W configs).
3. Override `events.randomize_reset_base.params` so the robot spawns upright on flat ground:
   - `pose_range`: `x=(-0.5, 0.5)`, `y=(-0.5, 0.5)`, `z=(0.0, 0.0)`, `roll=(0.0, 0.0)`, `pitch=(0.0, 0.0)`, `yaw=(-3.14, 3.14)`
   - `velocity_range`: `x=(-0.5, 0.5)`, `y=(-0.5, 0.5)`, `z=(-0.5, 0.5)`, `roll=(0.0, 0.0)`, `pitch=(0.0, 0.0)`, `yaw=(-0.5, 0.5)`
4. Agent-executed verification: config inspection, short play video, smoke train, short retrain, export/record.

### Must NOT have (guardrails, anti-slop, scope boundaries)
- NO changes to `UnitreeB2WPiperRoughOmniEnvCfg` (`b2w_omni_env_cfg.py`).
- NO changes to Task A / Task D official envs (`tasks/task_a/env_cfg.py`, `tasks/task_d/env_cfg.py`).
- NO changes to IsaacLab framework code.
- NO changes to robot asset USD or `assets/robots/b2w.py`.
- NO deletion or renaming of old checkpoints (`logs/rsl_rl/unitree_b2w_flat_omni/*/model_199.pt`).
- NO new recording scripts or changes to training/export scripts unless required by verification.
- NO modification of action/observation dimensions or reward weights.

## Verification strategy
> Zero human intervention - all verification is agent-executed.
- Test decision: tests-after (this is a config change; no unit tests exist for env configs). Verification is done by instantiating the config, running a 1-env play loop, and training smoke/short runs.
- Evidence: `.omo/evidence/task-<N>-fix-b2w-flat-omni-spawn.<ext>` for each todo.

## Execution strategy
### Parallel execution waves

**Wave 1: Config fix (1 todo)**
- T1. Update `flat_b2w_omni_env_cfg.py` with upright spawn overrides.

**Wave 2: Static + visual verification (2 parallel todos)**
- T2. Instantiate config and inspect `init_state.pos` and `randomize_reset_base.params`.
- T3. Record a short play video to confirm upright spawn.

**Wave 3: Training verification (3 todos, sequential)**
- T4. Smoke train: 64 envs × 10 iters.
- T5. Short retrain: 1024 envs × 200 iters.
- T6. Export new checkpoint and record a final verification video.

### Dependency matrix
| Todo | Depends on | Blocks | Can parallelize with |
| --- | --- | --- | --- |
| T1. Config fix | — | T2, T3, T4 | — |
| T2. Config inspection | T1 | T4 | T3 |
| T3. Play video | T1 | T4 | T2 |
| T4. Smoke train | T1, T2, T3 | T5 | — |
| T5. Short retrain | T4 | T6 | — |
| T6. Export/record | T5 | — | — |

## Todos
> Implementation + Test = ONE todo. Never separate.
<!-- APPEND TASK BATCHES BELOW THIS LINE WITH edit/apply_patch - never rewrite the headers above. -->

- [x] 1. Update `flat_b2w_omni_env_cfg.py` with upright spawn overrides
  What to do / Must NOT do: Edit `ATEC2026/source/atec_rl_lab/atec_rl_lab/train/locomotion/velocity/config/quadruped/unitree_b2/flat_b2w_omni_env_cfg.py`. In `UnitreeB2WPiperFlatOmniEnvCfg.__post_init__`, after `super().__post_init__()`:

  ```python
  # Match Task A / Task D B2W root init height
  self.scene.robot = self.scene.robot.replace(
      init_state=self.scene.robot.init_state.replace(pos=(0.0, 0.0, 0.78))
  )

  # Upright spawn on flat ground: zero roll/pitch, keep yaw randomized
  self.events.randomize_reset_base.params = {
      "pose_range": {
          "x": (-0.5, 0.5),
          "y": (-0.5, 0.5),
          "z": (0.0, 0.0),
          "roll": (0.0, 0.0),
          "pitch": (0.0, 0.0),
          "yaw": (-3.14, 3.14),
      },
      "velocity_range": {
          "x": (-0.5, 0.5),
          "y": (-0.5, 0.5),
          "z": (-0.5, 0.5),
          "roll": (0.0, 0.0),
          "pitch": (0.0, 0.0),
          "yaw": (-0.5, 0.5),
      },
  }
  ```

  Must NOT modify `b2w_omni_env_cfg.py` or any other file. Must NOT delete old checkpoints.
  Parallelization: Wave 1 | Blocked by: — | Blocks: T2, T3, T4
  References (executor has NO interview context - be exhaustive):
  - `ATEC2026/source/atec_rl_lab/atec_rl_lab/train/locomotion/velocity/config/quadruped/unitree_b2/flat_b2w_omni_env_cfg.py` — target file
  - `ATEC2026/source/atec_rl_lab/atec_rl_lab/train/locomotion/velocity/config/quadruped/unitree_b2/b2w_omni_env_cfg.py:85-102` — inherited bad reset distribution
  - `ATEC2026/source/atec_rl_lab/atec_rl_lab/train/locomotion/velocity/mdp/events.py:204-269` — `reset_root_state_uniform` adds pose_range offsets to default root state
  - `ATEC2026/source/atec_rl_lab/atec_rl_lab/tasks/task_a/env_cfg.py:223-225` — Task A B2W uses `init_state.pos.z = 0.78`
  - `ATEC2026/source/atec_rl_lab/atec_rl_lab/assets/robots/b2.py:41` — base B2 default root height is `z=0.58`
  Acceptance criteria (agent-executable):
  - File imports without error: `python -c "from atec_rl_lab.train.locomotion.velocity.config.quadruped.unitree_b2.flat_b2w_omni_env_cfg import UnitreeB2WPiperFlatOmniEnvCfg; print('OK')"`
  - `git diff --stat` shows only `flat_b2w_omni_env_cfg.py` modified (plus this plan artifact).
  QA scenarios (name the exact tool + invocation): happy + failure, Evidence .omo/evidence/task-1-fix-b2w-flat-omni-spawn.txt
  - Happy: import succeeds and config instantiates.
  - Failure: typo or wrong attribute name → AttributeError or ImportError.
  Commit: Y | fix(train): spawn B2W flat-omni upright at z=0.78

- [x] 2. Inspect generated config for correct spawn values
  What to do / Must NOT do: Run a Python snippet in the `atec2026-sim` environment that instantiates `UnitreeB2WPiperFlatOmniEnvCfg` and asserts the key fields. Do NOT run full simulation here.

  Exact command:
  ```bash
  source scripts/env/activate_atec2026_sim.sh
  python - <<'PY'
  from atec_rl_lab.train.locomotion.velocity.config.quadruped.unitree_b2.flat_b2w_omni_env_cfg import UnitreeB2WPiperFlatOmniEnvCfg
  cfg = UnitreeB2WPiperFlatOmniEnvCfg()
  assert cfg.scene.robot.init_state.pos == (0.0, 0.0, 0.78), cfg.scene.robot.init_state.pos
  p = cfg.events.randomize_reset_base.params["pose_range"]
  assert p["roll"] == (0.0, 0.0), p["roll"]
  assert p["pitch"] == (0.0, 0.0), p["pitch"]
  assert p["yaw"] == (-3.14, 3.14), p["yaw"]
  assert p["z"] == (0.0, 0.0), p["z"]
  v = cfg.events.randomize_reset_base.params["velocity_range"]
  assert v["roll"] == (0.0, 0.0), v["roll"]
  assert v["pitch"] == (0.0, 0.0), v["pitch"]
  print("spawn-config-ok")
  PY
  ```

  Must NOT modify any source file during inspection.
  Parallelization: Wave 2 | Blocked by: T1 | Blocks: T4 | Can parallelize with: T3
  References (executor has NO interview context - be exhaustive):
  - `ATEC2026/source/atec_rl_lab/atec_rl_lab/train/locomotion/velocity/config/quadruped/unitree_b2/flat_b2w_omni_env_cfg.py` — inspected config
  - `ATEC2026/source/atec_rl_lab/atec_rl_lab/train/locomotion/velocity/mdp/events.py:250` — `reset_root_state_uniform` parses `pose_range` keys `x,y,z,roll,pitch,yaw`
  Acceptance criteria (agent-executable):
  - Command prints `spawn-config-ok` and exits 0.
  QA scenarios (name the exact tool + invocation): happy + failure, Evidence .omo/evidence/task-2-fix-b2w-flat-omni-spawn.txt
  - Happy: all assertions pass.
  - Failure: parent override order wrong or attribute missing → AssertionError.
  Commit: N

- [x] 3. Record short play video to verify upright spawn
  What to do / Must NOT do: Run the existing `play.py` with `--video` for a few seconds to visually confirm the robot spawns upright. Use the new config; the old `model_199.pt` may be loaded only as a reference checkpoint for the visual check.

  Exact command:
  ```bash
  cd /home/1ctnltug/atec2026/ATEC2026
  source /home/1ctnltug/atec2026/scripts/env/activate_atec2026_sim.sh
  python scripts/rsl_rl/play.py \
    --task ATEC-Isaac-Velocity-Flat-Omni-B2W-Piper-v0 \
    --headless \
    --num_envs 1 \
    --resume \
    --load_run from_rough_omni \
    --checkpoint model_199.pt \
    --video \
    --video_length 200
  ```

  If `model_199.pt` does not exist, omit `--resume`, `--load_run`, and `--checkpoint` and run with random policy; the goal is only to verify spawn orientation.
  Must NOT modify `play.py` or create a new recording script.
  Parallelization: Wave 2 | Blocked by: T1 | Blocks: T4 | Can parallelize with: T2
  References (executor has NO interview context - be exhaustive):
  - `ATEC2026/source/atec_rl_lab/atec_rl_lab/train/locomotion/velocity/config/quadruped/unitree_b2/flat_b2w_omni_env_cfg.py` — new reset behavior
  - `ATEC2026/scripts/rsl_rl/play.py` — existing play script
  Acceptance criteria (agent-executable):
  - Command completes without crash.
  - Video file is created (location printed by play.py, usually under `videos/`).
  - Log/terminal shows no `roll`/`pitch` outliers at reset.
  QA scenarios (name the exact tool + invocation): happy + failure, Evidence .omo/evidence/task-3-fix-b2w-flat-omni-spawn.txt
  - Happy: video shows robot standing upright at start.
  - Failure: robot spawns vertical or upside down → `flat_b2w_omni_env_cfg.py` override did not take effect.
  Commit: N

- [x] 4. Smoke train: 64 envs × 10 iters
  What to do / Must NOT do: Run the existing training script with reduced envs and iterations to verify the config trains without shape/reset errors. Do NOT change the training script.

  Exact command:
  ```bash
  ATEC_B2W_FLAT_OMNI_ITERS=10 ATEC_TRAIN_NUM_ENVS=64 ./scripts/training/train_b2w_flat_omni.sh
  ```

  The script will warm-start from the latest `unitree_b2w_rough_omni` checkpoint if available; this is acceptable because the old flat-omni checkpoint is not used.
  Parallelization: Wave 3 | Blocked by: T1, T2, T3 | Blocks: T5
  References (executor has NO interview context - be exhaustive):
  - `scripts/training/train_b2w_flat_omni.sh` — existing training script
  - `ATEC2026/source/atec_rl_lab/atec_rl_lab/train/locomotion/velocity/config/quadruped/unitree_b2/flat_b2w_omni_env_cfg.py` — target env config
  Acceptance criteria (agent-executable):
  - Training runs to completion (10 iterations).
  - At least one checkpoint saved under `logs/rsl_rl/unitree_b2w_flat_omni/`.
  - No Python traceback.
  QA scenarios (name the exact tool + invocation): happy + failure, Evidence .omo/evidence/task-4-fix-b2w-flat-omni-spawn.txt
  - Happy: `model_0.pt` created, logs show reward computation.
  - Failure: crash on reset → orientation range format error or `init_state` missing.
  Commit: N

- [x] 5. Short retrain: 1024 envs × 200 iters
  What to do / Must NOT do: Run a useful short retrain to produce a new checkpoint that learns the corrected reset distribution. The existing bad `model_199.pt` must not be treated as the final policy.

  Exact command:
  ```bash
  ATEC_B2W_FLAT_OMNI_ITERS=200 ATEC_TRAIN_NUM_ENVS=1024 ./scripts/training/train_b2w_flat_omni.sh
  ```

  Parallelization: Wave 3 | Blocked by: T4 | Blocks: T6
  References (executor has NO interview context - be exhaustive):
  - `scripts/training/train_b2w_flat_omni.sh` — existing training script
  - `ATEC2026/source/atec_rl_lab/atec_rl_lab/train/locomotion/velocity/config/quadruped/unitree_b2/flat_b2w_omni_env_cfg.py` — target env config
  Acceptance criteria (agent-executable):
  - Training completes 200 iterations without crash or NaN.
  - Latest checkpoint saved under `logs/rsl_rl/unitree_b2w_flat_omni/from_rough_omni/`.
  - TensorBoard/CLI logs show non-negative mean return.
  QA scenarios (name the exact tool + invocation): happy + failure, Evidence .omo/evidence/task-5-fix-b2w-flat-omni-spawn.txt
  - Happy: losses stable, reward increasing.
  - Failure: OOM → reduce `ATEC_TRAIN_NUM_ENVS` and retry.
  Commit: N

- [x] 6. Export new checkpoint and record final verification video
  What to do / Must NOT do: Export the latest checkpoint from the short retrain to `demo/policy_b2w_flat_omni.pt` and record a video to confirm the robot still spawns upright with the new policy.

  Exact commands:
  ```bash
  ./scripts/training/export_b2w_flat_omni_policy_to_demo.sh
  ```
  Then play the exported policy:
  ```bash
  python scripts/rsl_rl/play.py \
    --task ATEC-Isaac-Velocity-Flat-Omni-B2W-Piper-v0 \
    --headless \
    --num_envs 1 \
    --resume \
    --load_run from_rough_omni \
    --checkpoint model_199.pt \
    --video \
    --video_length 600
  ```

  Adjust `--load_run` and `--checkpoint` to the actual latest run if different from `from_rough_omni/model_199.pt`. The export script already copies the latest checkpoint; for the play command use the actual checkpoint name under `logs/rsl_rl/unitree_b2w_flat_omni/`.
  Parallelization: Wave 3 | Blocked by: T5 | Blocks: —
  References (executor has NO interview context - be exhaustive):
  - `scripts/training/export_b2w_flat_omni_policy_to_demo.sh` — export script
  - `ATEC2026/demo/policy_b2w_flat_omni.pt` — target export path
  - `ATEC2026/scripts/rsl_rl/play.py` — play script
  Acceptance criteria (agent-executable):
  - `demo/policy_b2w_flat_omni.pt` exists and is newer than any previous version.
  - Final play video shows robot spawning upright and not falling over immediately.
  QA scenarios (name the exact tool + invocation): happy + failure, Evidence .omo/evidence/task-6-fix-b2w-flat-omni-spawn.txt
  - Happy: exported policy loads, video shows upright stable spawn.
  - Failure: exported checkpoint does not load → check experiment_name and run path.
  Commit: N

## Final verification wave
> Runs in parallel after ALL todos. ALL must APPROVE. Surface results and wait for the user's explicit okay before declaring complete.
- [x] F1. Plan compliance audit — verified: T1 commit only touched `flat_b2w_omni_env_cfg.py`. No changes to rough-omni, Task A/D, IsaacLab, robot USD, or old checkpoints.
- [x] F2. Code quality review — no TODO/FIXME in config file. Syntax valid (confirmed by Python import). No hardcoded paths beyond existing patterns.
- [x] F3. Real manual QA — smoke train (64×10 ✓), short retrain (1024×200 ✓), exported policy at `demo/policy_b2w_flat_omni.pt` (4.6MB), final video 600 frames upright spawn confirmed.
- [x] F4. Scope fidelity — T1 commit confirms only `flat_b2w_omni_env_cfg.py` was modified by this plan. Other dirty files are from previous plans (taskd-finetune, b2w-flat-omni) and unrelated.

## Commit strategy
- T1: Single commit `fix(train): spawn B2W flat-omni upright at z=0.78`
- T2-T6: No commits (verification runs)
- Final verification: amend or add a final commit only if F2/F4 require minor cleanup; otherwise no extra commit.

## Success criteria
1. `UnitreeB2WPiperFlatOmniEnvCfg` instantiates with `scene.robot.init_state.pos == (0.0, 0.0, 0.78)`.
2. `events.randomize_reset_base.params["pose_range"]` has `roll=(0.0, 0.0)`, `pitch=(0.0, 0.0)`, `yaw=(-3.14, 3.14)`, `z=(0.0, 0.0)`.
3. Smoke train (`64 envs × 10 iters`) completes without error.
4. Short retrain (`1024 envs × 200 iters`) completes and produces a new checkpoint.
5. Exported `demo/policy_b2w_flat_omni.pt` loads and the robot spawns upright in the final video.
6. No modifications outside `flat_b2w_omni_env_cfg.py` and plan artifacts.
