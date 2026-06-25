---
slug: taskd-flat-from-taska
status: awaiting-approval
intent: clear
pending-action: write .omo/plans/taskd-flat-from-taska.md
approach: Mirror Task A's flat-first curriculum for the B2W+Piper (Task D pre-training) pipeline: train flat omni from scratch, then warm-start rough omni from the flat omni checkpoint, then keep Task D fine-tuning unchanged. Fix the existing backwards warm-start in train_b2w_flat_omni.sh and add the missing flat→rough omni script.
---

# Draft: taskd-flat-from-taska

## Components (topology ledger)
| id | outcome | status | evidence path |
|---|---|---|---|
| C1 | Flat B2W omni env config aligned with Task A flat patterns | active | ATEC2026/source/atec_rl_lab/atec_rl_lab/train/locomotion/velocity/config/quadruped/unitree_b2/flat_b2w_omni_env_cfg.py |
| C2 | Flat B2W omni training script trains from scratch (not from rough omni) | active | scripts/training/train_b2w_flat_omni.sh |
| C3 | New rough B2W omni training script warm-started from flat omni checkpoint | active | scripts/training/train_b2w_rough_omni_from_flat.sh (to create) |
| C4 | Task D fine-tune script can consume the new rough omni checkpoint | active | scripts/training/train_taskd_finetune.sh |
| C5 | Export scripts updated to match the new pipeline | deferred | scripts/training/export_*.sh |

## Open assumptions (announced defaults)
| assumption | adopted default | rationale | reversible? |
|---|---|---|---|
| Meaning of "Task D flat training" | B2W+Piper flat-omni pre-training for Task D (existing ATEC-Isaac-Velocity-Flat-Omni-B2W-Piper-v0), not a flat-terrain variant of the full Task D env | README describes this as "训练 B2W 平地全向策略（用于 Task D）"; a full Task D flat env would need box + flat terrain which is a larger scope | yes — user can override |
| Keep existing Task A scripts untouched | Yes | User asked to port approach, not modify Task A | yes |
| Keep B2W rough omni env config unchanged | Yes | Existing rough omni config is the target; only the warm-start source changes | yes |

## Findings (cited - path:lines)
- **Task A training flow (flat → rough straight):**
  - Flat env: `ATEC2026/source/atec_rl_lab/atec_rl_lab/train/locomotion/velocity/config/quadruped/unitree_b2/flat_env_cfg.py:9` inherits from rough, sets `terrain_type="plane"`, disables height scan, disables terrain curriculum, calls `disable_zero_weight_rewards()`.
  - Flat runner: `.../agents/rsl_rl_ppo_cfg.py:39` `UnitreeB2FlatPPORunnerCfg` uses `max_iterations=5000`, `experiment_name="unitree_b2_flat"`.
  - Rough-straight runner: `.../agents/rsl_rl_ppo_cfg.py:48` `UnitreeB2RoughStraightPPORunnerCfg` uses `max_iterations=8000`, `experiment_name="unitree_b2_rough_straight"`.
  - Training script: `scripts/training/train_b2_rough_straight_from_flat.sh:12` finds latest checkpoint in `logs/rsl_rl/unitree_b2_flat` and warm-starts rough-straight from it with `--actor_checkpoint`.

- **Existing Task D pre-training flow:**
  - Flat env: `.../flat_b2w_omni_env_cfg.py:9` `UnitreeB2WPiperFlatOmniEnvCfg` already sets plane terrain, disables height scan, sets upright spawn, robot init height 0.78.
  - Flat runner: `.../agents/rsl_rl_ppo_flat_b2w_omni_cfg.py:7` `UnitreeB2WPiperFlatOmniPPORunnerCfg` uses `max_iterations=10000`, `experiment_name="unitree_b2w_flat_omni"`.
  - Rough omni runner: `.../agents/rsl_rl_ppo_omni_cfg.py:7` `UnitreeB2WPiperRoughOmniPPORunnerCfg` uses `max_iterations=20000`, `experiment_name="unitree_b2w_rough_omni"`.
  - Registration: `.../unitree_b2/__init__.py:52` registers `ATEC-Isaac-Velocity-Flat-Omni-B2W-Piper-v0` and `ATEC-Isaac-Velocity-Rough-Omni-B2W-Piper-v0`.

- **Identified issue:** `scripts/training/train_b2w_flat_omni.sh:7-33` currently warm-starts flat omni from the **rough omni** checkpoint (`logs/rsl_rl/unitree_b2w_rough_omni`). This is the reverse of Task A's flat-first curriculum.

- **Missing piece:** There is no script that warm-starts rough omni from a flat omni checkpoint. The existing `scripts/training/train_b2w_rough_omni_from_straight.sh:7-37` warm-starts from B2 rough-straight (`logs/rsl_rl/unitree_b2_rough_straight`), which is a cross-robot transfer (B2 12D → B2W 16D).

- **Checkpoint loading:** `ATEC2026/scripts/rsl_rl/train.py:168-312` already supports actor-only warm-start with observation expansion (45D→53D, 53D→61D), so flat omni → rough omni (same 53D input, same 16D output) should be an exact match.

## Decisions (with rationale)
- Default approach: mirror Task A's curriculum exactly for the B2W+Piper pre-training pipeline.
- Keep the flat B2W omni env config mostly as-is; it already follows Task A flat patterns. Only consider adding `disable_zero_weight_rewards()` and/or `base_height_l2.params["sensor_cfg"] = None` for strict parity if the user agrees.
- Do not create a full Task D flat-terrain env variant unless the user explicitly asks; that would introduce box/terrain coupling and is outside the default "pre-training" interpretation.

## Scope IN
- Modify `scripts/training/train_b2w_flat_omni.sh` to train flat B2W omni from scratch (remove warm-start from rough omni; keep optional fallback to demo policy only as a convenience, not primary flow).
- Add `scripts/training/train_b2w_rough_omni_from_flat.sh` that warm-starts rough omni from the latest flat omni checkpoint.
- Update `scripts/training/train_taskd_finetune.sh` (easy stage) to prefer flat-omni-derived rough omni checkpoint while still falling back to the existing source if needed.
- Optionally align `flat_b2w_omni_env_cfg.py` with Task A flat conventions (disable_zero_weight_rewards, base_height sensor config).
- Add/export script adjustments so the new flat→rough pipeline can be exported cleanly.

## Scope OUT (Must NOT have)
- No changes to Task A files (`flat_env_cfg.py`, `rough_env_cfg.py`, `rsl_rl_ppo_cfg.py`, `train_b2_rough_straight_from_flat.sh`).
- No changes to the full Task D env (`taskd_omni_env_cfg.py`) or Task D reward/observation terms.
- No changes to robot asset configs or solution.py.

## Open questions
1. **What does "Task D flat ground training" mean to you?**
   - **(A — recommended)** Fix/port the existing B2W+Piper flat-omni pre-training pipeline so it follows Task A's flat-first curriculum (this is the default above).
   - **(B)** Create a brand-new Task D environment variant that keeps the box and task rewards but replaces pit-and-platform terrain with flat ground.
   - **(C)** Both A and B.

2. **For the flat B2W omni script, what should the warm-start fallback be?**
   - **(A — recommended)** Train truly from scratch; remove the current "warm-start from rough omni" logic entirely.
   - **(B)** Keep a fallback to `demo/policy_taskd_omni.pt` if it exists, so a user without a fresh flat checkpoint can still run something.
   - **(C)** Keep the existing backwards behavior as-is and only add the new flat→rough script.

3. **Should the flat B2W omni env config be made stricter like Task A flat?**
   - **(A — recommended)** Leave it as-is; the parent rough omni config already disables zero-weight rewards and base_height is zero-weight.
   - **(B)** Add explicit `disable_zero_weight_rewards()` and `base_height_l2.params["sensor_cfg"] = None` for visual parity with Task A flat.

## Approval gate
status: awaiting-approval
<!-- When exploration is exhausted and unknowns are answered, set status: awaiting-approval. -->
<!-- That durable record is the loop guard: on a later turn read it and resume at the gate instead of re-running exploration. -->
