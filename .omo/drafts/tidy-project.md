---
slug: tidy-project
status: awaiting-approval
intent: clear
pending-action: write .omo/plans/tidy-project.md
approach: Two-phase tidy: (1) physical file reorganization of root-level artifacts, (2) document dedup/staleness cleanup.
---

# Draft: tidy-project

## Components (topology ledger)
<!-- Lock the SHAPE before depth. One row per top-level component that can succeed or fail independently. -->
<!-- id | outcome (one line) | status: active|deferred | evidence path -->

| id | outcome | status | evidence |
|---|---|---|---|
| root-clutter | 6 misplaced entries moved/deleted | active | ls /home/1ctnltug/atec2026/ |
| doc-dedup | stale/duplicate docs cleaned or consolidated | active | wc -l each .md |
| script-path | record_task_d_b2w_video.sh relocated to scripts/task_d/ | active | read of root ls |
| git-history | final commit of tidy changes | deferred | |

## Open assumptions (announced defaults)
<!-- Record any default you adopt instead of asking, so the user can veto it at the gate. -->
<!-- assumption | adopted default | rationale | reversible? -->

| assumption | adopted default | rationale | reversible? |
|---|---|---|---|
| `key.txt` location | Move to `notes/` — API keys don't belong at project root | Security hygiene, already in .gitignore | Yes |
| `issue.md` (0 bytes) | Delete — empty file with trivial note | No value to keep | Yes (git) |
| `SETUP_STATUS.md` (stale) | Delete — info fully covered by CLAUDE.md | Last updated 2026-05-21, duplicates CLAUDE.md | Yes (git) |
| `notes/ATEC2026_DEPLOYMENT_GUIDE.md` | Update stale content + add deprecation header | References wrong Python version (3.12 vs 3.11), wrong entry point (run_env.py) | Yes |
| `record_task_d_b2w_video.sh` | Move to `scripts/task_d/` | All other scripts under scripts/ | Yes |
| `README.md` | Keep but add cross-ref to CLAUDE.md at top | Chinese human docs are useful, but avoid drift | Yes |

## Findings (cited - path:lines)

1. **Root-level clutter**: 6 entries at `/home/1ctnltug/atec2026/` don't belong at root:
   - `key.txt` (API key, .gitignored)
   - `issue.md` (0 bytes, single line)
   - `record_task_d_b2w_video.sh` (script outside scripts/)
   - `task_d_video.mp4` (video outside artifacts/)
   - `submission_taskd.zip` (submission zip at root)
   - `submission_taskd/` (submission dir at root)

2. **Document overlap analysis** (wc -l):
   - `CLAUDE.md` = 228 lines — most comprehensive, covers all
   - `AGENTS.md` = 104 lines — subset for AI agent config (~60% overlap with CLAUDE.md)
   - `README.md` = 187 lines — Chinese human guide, heavy overlap with CLAUDE.md
   - `SETUP_STATUS.md` = 115 lines — stale (last updated 2026-05-21), content fully covered by CLAUDE.md
   - `notes/ATEC2026_DEPLOYMENT_GUIDE.md` = 214 lines — stale deployment guide
   - `ONBOARDING.md` = 62 lines — team process doc

3. **notes/ATEC2026_DEPLOYMENT_GUIDE.md staleness**:
   - Line 12: says Python 3.12.x required → workspace uses `atec2026-sim` (Python 3.11)
   - Line 136: references `demo/run_env.py` → workspace uses `scripts/play_atec_task.py`
   - Line 164: interface is `predict(self, obs, current_score)` → actual is `predicts(self, obs, current_score)`
   - Lines 10-16: version constraints (Cuda 12.8.1, etc.) are for fresh install, not current workspace

4. **script path inconsistency**: All task scripts under `scripts/task_a/`, `scripts/training/`, `scripts/env/` but `record_task_d_b2w_video.sh` is at root. CLAUDE.md line 104 references it as `./record_task_d_b2w_video.sh`.

5. **No docs/ directory**: Standard for project documentation per neat-freak conventions, but `notes/` serves this role.

## Decisions (with rationale)

1. **Delete SETUP_STATUS.md** — All info (IsaacLab paths, conda env details, critical flags) is in CLAUDE.md. Keeping it risks future drift.
2. **Update notes/ATEC2026_DEPLOYMENT_GUIDE.md** — Mark as "reference only, actual workspace may differ" rather than full rewrite. This preserves the original deployment procedure while warning about drift.
3. **Keep README.md mostly as-is** — Chinese human onboarding is valuable. Add a single-line cross-ref to CLAUDE.md for authoritative info.
4. **Keep AGENTS.md concise** — Its purpose as AI agent config is served well at 104 lines.
5. **Keep ONBOARDING.md untouched** — Team process doc, not project-specific.

## Scope IN

1. Physical reorganization of root-level files
2. Cleanup of stale/duplicate markdown documentation
3. Path updates in CLAUDE.md if scripts move
4. One tidy-up commit

## Scope OUT (Must NOT have)

- Do NOT modify solution.py or any Python code
- Do NOT modify training configs
- Do NOT touch IsaacLab/ or ATEC2026_Simulation_Challenge/ content beyond updating .md references
- Do NOT add new features or change any behavior
- Do NOT create new documentation directories (keep notes/)

## Open questions

None — all findings resolved through exploration.

## Approval gate
status: awaiting-approval
<!-- When exploration is exhausted and unknowns are answered, set status: awaiting-approval. -->
<!-- That durable record is the loop guard: on a later turn read it and resume at the gate instead of re-running exploration. -->
