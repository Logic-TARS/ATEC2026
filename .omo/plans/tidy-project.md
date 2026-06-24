# tidy-project - Work Plan

## TL;DR (For humans)

**What you'll get:** A clean project root with no stray files — scripts under `scripts/`, videos and submissions under `artifacts/`, stale setup status deleted. Documentation deduplicated so `CLAUDE.md` is the single source of truth, and old deployment guide marked as historical reference.

**Why this approach:** Two-phase (physical first, docs second) makes each step independently verifiable. Deleting stale docs instead of rewriting prevents future drift; updating the deployment guide as "reference only" preserves the original procedure without misleading new readers.

**What it will NOT do:** Touch any Python code, training configs, IsaacLab, or the ATEC challenge project source. No behavior changes, no new features.

**Effort:** Short — 7 focused file operations + 3 doc edits  
**Risk:** Low — all changes are reversible via git; no functional code touched  
**Decisions to sanity-check:** Deleting SETUP_STATUS.md (is all its info truly covered by CLAUDE.md?) and moving `record_task_d_b2w_video.sh` (needs the referencing CLAUDE.md line updated)

Your next move: approve this plan, then I'll execute it with `$start-work`.

---

> TL;DR (machine): Effort=Short Risk=Low | 7 file ops + 3 doc edits | deliverables=clean root + deduped docs

## Scope
### Must have
1. Remove 6 stray root-level files/dirs
2. Delete stale `SETUP_STATUS.md`
3. Update `notes/ATEC2026_DEPLOYMENT_GUIDE.md` with deprecation header
4. Update `CLAUDE.md` path references if scripts move
5. Delete empty `issue.md`
6. One tidy-up commit

### Must NOT have (guardrails, anti-slop, scope boundaries)
- NO modifications to .py, .pt, .sh logic (except path moves)
- NO changes to IsaacLab/ or ATEC2026_Simulation_Challenge/ source
- NO creation of new docs/ directory
- NO edits to ONBOARDING.md, AGENTS.md, README.md content (only cross-refs if needed)
- NO functional changes — tidy only

## Verification strategy
> Zero human intervention - all verification is agent-executed.
- Test decision: none (file operations, no logic)
- Evidence: ls output before/after for each file operation

## Execution strategy
### Parallel execution waves

**Wave 1: physical cleanup** (parallel — all independent file moves/deletes)
- T1. Delete `issue.md`
- T2. Create `scripts/task_d/` + move `record_task_d_b2w_video.sh` into it
- T3. Move `task_d_video.mp4` → `artifacts/task_d_videos/`
- T4. Move `submission_taskd.zip` + `submission_taskd/` → `artifacts/submissions/`
- T5. Move `key.txt` → `notes/`

**Wave 2: documentation cleanup** (sequential — T7 depends on T2 completion)
- T6. Delete `SETUP_STATUS.md`
- T7. Update `CLAUDE.md` path ref (line 104: `./record_task_d_b2w_video.sh` → `./scripts/task_d/record_task_d_b2w_video.sh`)
- T8. Add deprecation header to `notes/ATEC2026_DEPLOYMENT_GUIDE.md`

**Wave 3: commit**
- T9. Git commit all changes

### Dependency matrix
| Todo | Depends on | Blocks | Can parallelize with |
| --- | --- | --- | --- |
| T1. delete issue.md | — | — | T2, T3, T4, T5 |
| T2. move record script | — | T7 | T1, T3, T4, T5 |
| T3. move video | — | — | T1, T2, T4, T5 |
| T4. move submission files | — | — | T1, T2, T3, T5 |
| T5. move key.txt | — | — | T1, T2, T3, T4 |
| T6. delete SETUP_STATUS.md | — | — | T1..T5 |
| T7. update CLAUDE.md ref | T2 | — | T6, T8 |
| T8. update deployment guide | — | — | T6, T7 |
| T9. git commit | T1..T8 | — | — |

## Todos

<!-- APPEND TASK BATCHES BELOW THIS LINE WITH edit/apply_patch - never rewrite the headers above. -->
- [x] 1. Delete empty `issue.md`
  What to do / Must NOT do: Remove the 0-byte file `/home/1ctnltug/atec2026/issue.md`. Must NOT modify any other file.
  Parallelization: Wave 1 | Blocked by: — | Blocks: —
  References: ls shows issue.md at root, 0 bytes, single-line content
  Acceptance criteria: `ls /home/1ctnltug/atec2026/issue.md` returns "No such file"
  QA scenarios: `ls -la /home/1ctnltug/atec2026/issue.md` → must fail with ENOENT
  Commit: N (will be part of final T9)

- [x] 2. Create `scripts/task_d/` and move `record_task_d_b2w_video.sh`
  What to do / Must NOT do: Create directory `scripts/task_d/`; `mv /home/1ctnltug/atec2026/record_task_d_b2w_video.sh /home/1ctnltug/atec2026/scripts/task_d/record_task_d_b2w_video.sh`. Must NOT modify the script content.
  Parallelization: Wave 1 | Blocked by: — | Blocks: T7
  References: Root ls shows record_task_d_b2w_video.sh; scripts/task_a/ exists as pattern
  Acceptance criteria: Script exists at new path and is executable; old path does not exist
  QA scenarios: `ls -la /home/1ctnltug/atec2026/scripts/task_d/record_task_d_b2w_video.sh` → executable; `ls /home/1ctnltug/atec2026/record_task_d_b2w_video.sh` → ENOENT
  Commit: N

- [x] 3. Move `task_d_video.mp4` into `artifacts/task_d_videos/`
  What to do / Must NOT do: `mv /home/1ctnltug/atec2026/task_d_video.mp4 /home/1ctnltug/atec2026/artifacts/task_d_videos/task_d_video.mp4`. Must NOT modify file content.
  Parallelization: Wave 1 | Blocked by: — | Blocks: —
  References: Root ls shows task_d_video.mp4 (456K); artifacts/task_d_videos/ exists
  Acceptance criteria: File exists at target; absent from root
  QA scenarios: `ls /home/1ctnltug/atec2026/artifacts/task_d_videos/task_d_video.mp4` → exists; `ls /home/1ctnltug/atec2026/task_d_video.mp4` → ENOENT
  Commit: N

- [x] 4. Move `submission_taskd.zip` and `submission_taskd/` into `artifacts/submissions/`
  What to do / Must NOT do: Create `artifacts/submissions/` if needed; `mv submission_taskd.zip artifacts/submissions/`; `mv submission_taskd/ artifacts/submissions/`. Must NOT modify zip or directory contents.
  Parallelization: Wave 1 | Blocked by: — | Blocks: —
  References: Root ls shows submission_taskd.zip (720K) and submission_taskd/ (808K) at root
  Acceptance criteria: Both exist under artifacts/submissions/; absent from root
  QA scenarios: `ls /home/1ctnltug/atec2026/artifacts/submissions/` → shows both; `ls /home/1ctnltug/atec2026/submission_taskd*` → ENOENT
  Commit: N

- [x] 5. Move `key.txt` into `notes/`
  What to do / Must NOT do: `mv /home/1ctnltug/atec2026/key.txt /home/1ctnltug/atec2026/notes/key.txt`. Must NOT modify file content. Must NOT commit API key to git.
  Parallelization: Wave 1 | Blocked by: — | Blocks: —
  References: .gitignore already ignores key.txt at any path
  Acceptance criteria: File exists at notes/key.txt; absent from root
  QA scenarios: `ls /home/1ctnltug/atec2026/notes/key.txt` → exists; `ls /home/1ctnltug/atec2026/key.txt` → ENOENT
  Commit: N

- [x] 6. Delete stale `SETUP_STATUS.md`
  What to do / Must NOT do: `rm /home/1ctnltug/atec2026/SETUP_STATUS.md`. Info is fully covered by CLAUDE.md (IsaacLab paths, conda env details, critical flags). Must NOT delete any other file.
  Parallelization: Wave 2 | Blocked by: — | Blocks: —
  References: SETUP_STATUS.md = 115 lines, last updated 2026-05-21; content overlaps with CLAUDE.md lines 36-68
  Acceptance criteria: File does not exist at root
  QA scenarios: `ls /home/1ctnltug/atec2026/SETUP_STATUS.md` → ENOENT
  Commit: N

- [x] 7. Update CLAUDE.md reference for moved script
  What to do / Must NOT do: In `/home/1ctnltug/atec2026/CLAUDE.md` line 104, change `./record_task_d_b2w_video.sh` to `./scripts/task_d/record_task_d_b2w_video.sh`. Must NOT change any other content.
  Parallelization: Wave 2 | Blocked by: T2 | Blocks: —
  References: CLAUDE.md line 104: `./record_task_d_b2w_video.sh`
  Acceptance criteria: grep shows the new path only, not the old one
  QA scenarios: `grep record_task_d /home/1ctnltug/atec2026/CLAUDE.md` → shows `./scripts/task_d/record_task_d_b2w_video.sh`; `grep './record_task_d' /home/1ctnltug/atec2026/CLAUDE.md` → empty
  Commit: N

- [x] 8. Add deprecation header to `notes/ATEC2026_DEPLOYMENT_GUIDE.md`
  What to do / Must NOT do: Prepend a deprecation notice block at the top of the file noting: (a) this is the original deployment guide from competition instructions, (b) actual workspace uses `atec2026-sim` env (Python 3.11), not the spec env (Python 3.12), (c) entry point is `scripts/play_atec_task.py`, not `demo/run_env.py`, (d) interface method is `predicts()` not `predict()`. Must NOT delete any original content.
  Parallelization: Wave 2 | Blocked by: — | Blocks: —
  References: notes/ATEC2026_DEPLOYMENT_GUIDE.md line 12 (Python 3.12), line 136 (run_env.py), line 164 (predict)
  Acceptance criteria: File starts with a deprecation notice block
  QA scenarios: `head -10 /home/1ctnltug/atec2026/notes/ATEC2026_DEPLOYMENT_GUIDE.md` → shows deprecation header; total line count increased by 6-8 lines
  Commit: N

- [x] 9. Git commit all tidy changes
  What to do / Must NOT do: `git add -A` (respects .gitignore; key.txt at new path is still gitignored); `git commit -m "chore: tidy project root - move artifacts, delete stale docs, update refs"`. Must NOT include any unstaged unrelated changes.
  Parallelization: Wave 3 | Blocked by: T1..T8 | Blocks: —
  References: All previous tasks
  Acceptance criteria: `git status` is clean; `git log --oneline -1` shows the commit message
  QA scenarios: `git status --porcelain` → empty; `git log --oneline -1` → contains "tidy project root"
  Commit: Y | chore: tidy project root - move artifacts, delete stale docs, update refs

## Final verification wave
> Runs in parallel after ALL todos. ALL must APPROVE. Surface results and wait for the user's explicit okay before declaring complete.
- [x] F1. Plan compliance audit — verify all 9 todos completed
- [x] F2. Root dir check — `ls /home/1ctnltug/atec2026/` shows only clean entries
- [x] F3. Path consistency — CLAUDE.md references match actual file locations
- [x] F4. Scope fidelity — no Python/IsaacLab/ATEC source was modified

## Commit strategy
Single commit after all changes: `chore: tidy project root - move artifacts, delete stale docs, update refs`

## Success criteria
1. `ls /home/1ctnltug/atec2026/` shows 14 clean entries (down from 21)
2. `git status --porcelain` is clean after commit
3. No stale/empty files remain at root
4. CLAUDE.md path references match actual file locations
5. Original deployment guide marked as historical reference
