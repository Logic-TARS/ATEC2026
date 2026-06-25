---
slug: fix-b2w-flat-omni-spawn
status: approved
intent: clear
pending-action: plan written to .omo/plans/fix-b2w-flat-omni-spawn.md; wait for user to start work
approach: In `UnitreeB2WPiperFlatOmniEnvCfg.__post_init__`, after `super().__post_init__()`, override the robot's default root position to z=0.78 and override `events.randomize_reset_base.params` so roll/pitch are fixed at 0, yaw stays fully randomized, x/y jitter remains small, and angular-velocity roll/pitch are zeroed. Leave rough-omni and all official envs untouched. Verify via config inspection, a short play video, and smoke/short retrain.
---

# Draft: fix-b2w-flat-omni-spawn

## Components (topology ledger)
| id | outcome | status | evidence path |
| --- | --- | --- | --- |
| cfg | `UnitreeB2WPiperFlatOmniEnvCfg` reset/spawn overrides added | active | `.omo/evidence/fix-b2w-flat-omni-spawn-config.txt` |
| inspect | 1-env config inspection confirms z=0.78, roll=0, pitch=0, yaw random | active | `.omo/evidence/fix-b2w-flat-omni-spawn-inspect.txt` |
| play | Short play video shows upright spawn | active | `.omo/evidence/fix-b2w-flat-omni-spawn-play.txt` |
| smoke | 64 envs × 10 iters smoke train passes | active | `.omo/evidence/fix-b2w-flat-omni-spawn-smoke.txt` |
| retrain | 1024 envs × 200 iters short retrain completes | active | `.omo/evidence/fix-b2w-flat-omni-spawn-retrain.txt` |
| export | New checkpoint exported and verified | active | `.omo/evidence/fix-b2w-flat-omni-spawn-export.txt` |

## Open assumptions (announced defaults)
| assumption | adopted default | rationale | reversible? |
| --- | --- | --- | --- |
| Reset angular-velocity roll/pitch | `(0.0, 0.0)` | Maximally stable flat-ground startup; user listed `0.0` as an acceptable option | Yes |
| x/y position jitter | Keep parent's `(-0.5, 0.5)` | Already small; user said "keep small" without requesting a new value | Yes |
| z position jitter | `(0.0, 0.0)` | Keeps nominal spawn height exactly at the configured `z=0.78` | Yes |
| Warm-start source | Continue using latest `unitree_b2w_rough_omni` checkpoint per existing script | Existing `model_199.pt` learned from bad reset distribution and should not be used as final policy | Yes — script can be invoked without warm-start by ensuring no source checkpoint exists |
| Old flat-omni checkpoint | Treat `logs/rsl_rl/unitree_b2w_flat_omni/*/model_199.pt` as reference only, not final policy | User explicitly says it was trained on bad reset distribution | Yes — file is not deleted |

## Findings (cited - path:lines)
- `ATEC2026/source/atec_rl_lab/atec_rl_lab/train/locomotion/velocity/config/quadruped/unitree_b2/flat_b2w_omni_env_cfg.py:9` — `UnitreeB2WPiperFlatOmniEnvCfg` extends `UnitreeB2WPiperRoughOmniEnvCfg`; only terrain/height-scan/curriculum are overridden today.
- `ATEC2026/source/atec_rl_lab/atec_rl_lab/train/locomotion/velocity/config/quadruped/unitree_b2/b2w_omni_env_cfg.py:85-102` — `randomize_reset_base.params` currently sets `z=(0.0, 0.2)`, `roll=(-3.14, 3.14)`, `pitch=(-3.14, 3.14)`, `yaw=(-3.14, 3.14)`.
- `ATEC2026/source/atec_rl_lab/atec_rl_lab/train/locomotion/velocity/mdp/events.py:204-269` — `reset_root_state_uniform` adds `pose_range` offsets to the asset's default root state, so setting the robot's `init_state.pos.z = 0.78` makes that the nominal reset height.
- `ATEC2026/source/atec_rl_lab/atec_rl_lab/tasks/task_a/env_cfg.py:223-225` — Task A B2W sets `init_state.pos=(-141, 0, 0.78)` via `UNITREE_B2W_PIPER_CFG.init_state.replace(...)`; this is the reference height we will match.
- `ATEC2026/source/atec_rl_lab/atec_rl_lab/assets/robots/b2.py:41` — Base `UNITREE_B2_CFG` default root height is `z=0.58`, which the B2W configs inherit before this fix.

## Decisions (with rationale)
1. **Only override in flat-omni, not rough-omni** — User explicitly requires leaving `UnitreeB2WPiperRoughOmniEnvCfg` unchanged to preserve prior training assumptions.
2. **Set default root height, then zero roll/pitch ranges** — `reset_root_state_uniform` adds sampled offsets to the asset's default root state; fixing orientation is done by setting orientation ranges to `(0.0, 0.0)` while leaving yaw randomized.
3. **Zero angular-velocity roll/pitch at reset** — Removes startup tumble on flat ground without removing all velocity randomization.
4. **Do not delete or rename existing `model_199.pt`** — User wants it preserved as reference; the new policy will come from a fresh short retrain.

## Scope IN
- One-file config change in `flat_b2w_omni_env_cfg.py`
- Config inspection to confirm values
- Short play video for visual verification
- Smoke train (64 envs × 10 iters)
- Short retrain (1024 envs × 200 iters)
- Export/record new checkpoint

## Scope OUT (Must NOT have)
- No changes to `UnitreeB2WPiperRoughOmniEnvCfg`
- No changes to Task A / Task D official envs
- No changes to IsaacLab framework
- No changes to robot asset USD
- No deletion of old checkpoints

## Open questions
- None remaining.

## Approval gate
status: awaiting-approval
