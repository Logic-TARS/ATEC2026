---
slug: b2w-flat-omni
status: awaiting-approval
intent: clear
pending-action: user approval of .omo/plans/b2w-flat-omni.md, then execute
approach: Create a flat-ground variant of the existing B2W rough-omni env by switching terrain to plane, removing the height scanner, and keeping the same 16D action / 53D observation layout. Warm-start training from the latest rough-omni checkpoint. Add PPO runner config, gym registration, training script, and export script. Smoke test 64 envs × 10 iters; full training is GPU-bound and manual.
---

# Draft: b2w-flat-omni

## Components (topology ledger)
| id | outcome | status | evidence path |
| --- | --- | --- | --- |
| env | `UnitreeB2WPiperFlatOmniEnvCfg` class created | active | `.omo/evidence/task-1-b2w-flat-omni-config.txt` |
| runner | `UnitreeB2WPiperFlatOmniPPORunnerCfg` created | active | `.omo/evidence/task-2-b2w-flat-omni-runner.txt` |
| register | New gym env ID registered | active | `.omo/evidence/task-3-b2w-flat-omni-registration.txt` |
| scripts | Train + export scripts created | active | `.omo/evidence/task-4-b2w-flat-omni-train-script.txt`, `.omo/evidence/task-5-b2w-flat-omni-export-script.txt` |
| smoke | 64 envs × 10 iters smoke test passes | active | `.omo/evidence/task-6-b2w-flat-omni-smoke.txt` |
| training | Full GPU training | deferred | `.omo/evidence/task-7-b2w-flat-omni-training.txt` |

## Open assumptions (announced defaults)
| assumption | adopted default | rationale | reversible? |
| --- | --- | --- | --- |
| Robot variant | B2W (16D leg+wheel) | User explicitly chose B2W | No — user decided |
| Warm-start source | Latest `unitree_b2w_rough_omni` checkpoint, fallback to `demo/policy_taskd_omni.pt` | Same 53D obs / 16D action layout; no column transfer needed | Yes — can train from scratch by omitting `--actor_checkpoint` |
| Terrain | Flat plane (`terrain_type="plane"`) | Matches existing `UnitreeB2FlatEnvCfg` pattern | Yes |
| Height scanner | Removed entirely on flat ground | Not used in observations; saves GPU resources | Yes — can keep if needed |
| Max iterations | 10000 | Flat ground is easier than rough; fewer iterations than rough omni's 20000 | Yes — env var override |
| Num envs default | 4096 for full training | Existing scripts use 4096 default | Yes — env var override |

## Findings (cited - path:lines)
- `ATEC2026/source/atec_rl_lab/atec_rl_lab/train/locomotion/velocity/config/quadruped/unitree_b2/b2w_omni_env_cfg.py:11` — `UnitreeB2WPiperRoughOmniEnvCfg` defines 16D action (12 leg + 4 wheel) and 53D observation.
- `ATEC2026/source/atec_rl_lab/atec_rl_lab/train/locomotion/velocity/config/quadruped/unitree_b2/flat_env_cfg.py:17-18` — Example of switching terrain to plane: `terrain_type = "plane"`, `terrain_generator = None`.
- `ATEC2026/source/atec_rl_lab/atec_rl_lab/train/locomotion/velocity/config/quadruped/unitree_b2/__init__.py:41-49` — Existing rough-omni registration pattern.
- `ATEC2026/source/atec_rl_lab/atec_rl_lab/train/locomotion/velocity/config/quadruped/unitree_b2/agents/rsl_rl_ppo_omni_cfg.py:7` — Existing rough-omni runner config with `max_iterations = 20000`.
- Existing rough-omni checkpoint available at `logs/rsl_rl/unitree_b2w_rough_omni/2026-06-21_19-02-24_from_straight/model_2800.pt` and exported at `demo/policy_taskd_omni.pt`.

## Decisions (with rationale)
1. **Extend rough-omni, not rough-straight** — Rough omni already has ModeBasedVelocityCommandCfg for forward/backward/turning; rough-straight is heading-only and unsuitable.
2. **Same obs/action dimensions as rough omni** — Enables direct warm-start weight copy without column transfer.
3. **Remove height scanner** — Observations don't use height scan in rough omni config; removing the sensor on flat ground reduces GPU overhead.
4. **No reward re-tuning for flat ground in Wave 1** — Parent rough-omni rewards already include velocity tracking and penalties; we will tune only if smoke test or early training shows issues.

## Scope IN
- Flat B2W omni env config
- PPO runner config
- Gym registration
- Training script with warm-start
- Export script
- Smoke test

## Scope OUT (Must NOT have)
- No Task D code changes
- No existing rough-omni config changes
- No arm control
- No new dependencies
- No IsaacLab framework changes

## Open questions
- None remaining. User confirmed B2W variant.

## Approval gate
status: completed
