# Draft: Task A (L0) Full Score - 26 Points

## Requirements (confirmed)
- Complete Task A (L0: Off-road Navigation) to score maximum 26 points
- Task A scoring: 5 checkpoints at x = [-115, -35, 45, 125, 140] with rewards [2, 4, 8, 8, 4] = 26 total
- Robot: B2wPiper (Unitree B2W + AgileX Piper arm)
- Training uses 12-DOF B2 config, solution.py maps to 24-DOF B2wPiper

## Technical Decisions
- Use existing training pipeline: train rough-straight from flat checkpoint
- Export via play.py JIT export, copy to demo/policy.pt
- solution.py already handles B2wPiper with fixed velocity command [1.5, 0, 0]

## Research Findings
- Terrain: 15 tiles × 20m = 300m course from x=-141 to x=145
  - Flat → random_rough × 4 → slopes × 4 → stairs × 4 → flat (goal)
- Current training status: rough_straight only 9 iterations (essentially not started)
- Flat training checkpoint exists at logs/rsl_rl/unitree_b2_flat/
- Training config: 8000 iterations default, 4096 envs, PPO with 24 steps/env
- PPO config: [512, 256, 128] actor/critic, lr=1e-3, adaptive schedule

## Open Questions
- Current baseline score unknown (need to run test first)
- Whether 8000 iterations is sufficient or need more
- Whether training terrain matches Task A terrain well enough

## Scope Boundaries
- INCLUDE: Train policy, export, test, iterate until 26 points
- EXCLUDE: Solution.py architecture changes (already working), new robot configs
