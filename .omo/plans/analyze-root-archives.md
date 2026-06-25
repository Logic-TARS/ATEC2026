# analyze-root-archives - Work Plan

## TL;DR (For humans)
<!-- Fill this LAST, after the detailed plan below is written, so it summarizes the REAL plan. -->
<!-- Plain English for a non-engineer: NO file paths, NO todo numbers, NO wave/agent/tool names. -->

**What you'll get:** A Chinese report explaining what each of the four tar.gz files in the workspace root contains, which files are duplicates, and the checksums of the key model weights inside.

**Why this approach:** Unpacking the archives to a temporary analysis folder and reading their README/MANIFEST files is the fastest, safest way to identify them without touching the originals or running any simulation.

**What it will NOT do:** It will not modify or delete the original archives, install anything, run Isaac Sim, execute the policies, or submit anything to the ATEC platform.

**Effort:** Quick
**Risk:** Low - only read-only `tar` extraction and checksum checks
**Decisions to sanity-check:** Whether to use the default analysis directory `/tmp/opencode/root_archive_analysis` instead of extracting under the workspace root.

Your next move: approve this plan so a worker can extract and analyze the archives. Full execution detail follows below.

---

> TL;DR (machine): Quick/Low - read-only extraction of four workspace-root tar.gz archives into /tmp/opencode/root_archive_analysis, produce Chinese summary report at .omo/evidence/root-archive-analysis.md.

## Scope
### Must have
- Decompress all four `*.tar.gz` archives at the workspace root into a safe analysis directory.
- Inspect each archive’s directory layout, metadata files (README/MANIFEST/MODEL_INFO/DEPLOYMENT), binary artifacts (`policy.pt`, checkpoints), and scripts.
- Record file counts, key MD5 checksums, and a human-readable classification of what each archive represents.
- Identify duplicate/identical archives and cross-archive relationships (e.g., handover `submission/` vs. standalone submit packages).
- Produce a consolidated markdown report under `.omo/evidence/root-archive-analysis.md`.

### Must NOT have (guardrails, anti-slop, scope boundaries)
- Do NOT modify, move, or delete the original `*.tar.gz` files in `/home/1ctnltug/atec2026/`.
- Do NOT install packages, run Isaac Sim, execute `solution.py`, or submit anything to the ATEC platform.
- Do NOT unpack into the workspace root or overwrite existing project files (`ATEC2026/`, `IsaacLab/`, `scripts/`, `demo/`, etc.).
- No network access; analysis is read-only on the local filesystem.

## Verification strategy
> Zero human intervention - all verification is agent-executed.
- Test decision: none (read-only inspection task; verification uses `tar`, `find`, `md5sum`, `diff`)
- Evidence: `.omo/evidence/root-archive-analysis.md` plus per-archive inspection logs under `.omo/evidence/task-<N>-analyze-root-archives.{txt,md}`

## Execution strategy
### Parallel execution waves
- Wave 1 (parallel): Extract and inspect each of the 4 archives as independent todos.
- Wave 2 (sequential): Generate consolidated comparison report and validate duplicates.

### Dependency matrix
| Todo | Depends on | Blocks | Can parallelize with |
| --- | --- | --- | --- |
| T1 | — | T5 | T2, T3, T4 |
| T2 | — | T5 | T1, T3, T4 |
| T3 | — | T5 | T1, T2, T4 |
| T4 | — | T5 | T1, T2, T3 |
| T5 | T1–T4 | F1–F4 | — |

## Todos
> Implementation + Test = ONE todo. Never separate.
<!-- APPEND TASK BATCHES BELOW THIS LINE WITH edit/apply_patch - never rewrite the headers above. -->
- [x] 1. Extract & inspect `atec_b2w_handover_20260610_1427_score26.00.tar.gz`
  What to do / Must NOT do: Unpack the handover archive to the analysis directory. Read `README.md`, `MANIFEST.txt`, `DEPLOYMENT.md`, `TRAINING.md`, `requirements_full.txt`, and list `submission/`, `checkpoints/`, `training_overlay/`, `scripts/`. Must NOT execute any scripts or install anything.
  Parallelization: Wave 1 | Blocked by: — | Blocks: T5
  References (executor has NO interview context - be exhaustive): `/home/1ctnltug/atec2026/atec_b2w_handover_20260610_1427_score26.00.tar.gz`
  Acceptance criteria (agent-executable):
    - `tar -tzf` shows the top-level directory `atec_b2w_handover/`
    - Extraction succeeds: `tar -xzf ... -C <analysis_dir>` exits 0
    - `find <analysis_dir>/atec_b2w_handover -type f | wc -l` ≥ 30
    - `md5sum <analysis_dir>/atec_b2w_handover/submission/policy.pt` matches `19c573ba32b773c73b27ef491af548bf`
    - Capture one-paragraph summary in `.omo/evidence/task-1-analyze-root-archives.md`
  QA scenarios (name the exact tool + invocation): happy: `tar -xzf` extracts without error and `find` returns expected file count; failure: corrupted archive → `tar` exits non-zero, abort. Evidence `.omo/evidence/task-1-analyze-root-archives.md`
  Commit: N | —

- [x] 2. Extract & inspect `atec_taskA_b2w_best_20260610_1543_score26.00_with_video.tar.gz`
  What to do / Must NOT do: Unpack the “best with video” archive. Read `README.txt`, `MANIFEST.txt`, `MODEL_INFO.txt`, `VIDEO_INFO.txt`, list `logs/`, `video/`, and verify `policy.pt`/`solution.py`. Must NOT play the video or run the policy.
  Parallelization: Wave 1 | Blocked by: — | Blocks: T5
  References (executor has NO interview context - be exhaustive): `/home/1ctnltug/atec2026/atec_taskA_b2w_best_20260610_1543_score26.00_with_video.tar.gz`
  Acceptance criteria (agent-executable):
    - Top-level directory is `atec_taskA_b2w_best_26/`
    - Extraction succeeds and contains `video/taskA_b2w_score26.00.mp4`, `logs/eval_26.00*.log`, `policy.pt`, `solution.py`
    - `md5sum policy.pt` matches `19c573ba32b773c73b27ef491af548bf`
    - Capture summary in `.omo/evidence/task-2-analyze-root-archives.md`
  QA scenarios: happy: all expected files present and md5 matches; failure: missing video or checksum mismatch. Evidence `.omo/evidence/task-2-analyze-root-archives.md`
  Commit: N | —

- [x] 3. Extract & inspect `atec_taskA_b2w_submit_20260610_1428_score26.00.tar.gz`
  What to do / Must NOT do: Unpack the first submit archive. Confirm it contains a minimal `atec_taskA_b2w/` directory with `solution.py`, `policy.pt`, `requirements.txt` only. Must NOT execute `solution.py`.
  Parallelization: Wave 1 | Blocked by: — | Blocks: T5
  References (executor has NO interview context - be exhaustive): `/home/1ctnltug/atec2026/atec_taskA_b2w_submit_20260610_1428_score26.00.tar.gz`
  Acceptance criteria (agent-executable):
    - Top-level directory is `atec_taskA_b2w/` (not bare files)
    - Contains exactly `solution.py`, `policy.pt`, `requirements.txt`
    - `md5sum policy.pt` matches `19c573ba32b773c73b27ef491af548bf`
    - Capture summary in `.omo/evidence/task-3-analyze-root-archives.md`
  QA scenarios: happy: 3 files present and checksum matches; failure: extra/missing files or bad checksum. Evidence `.omo/evidence/task-3-analyze-root-archives.md`
  Commit: N | —

- [x] 4. Extract & inspect `atec_taskA_b2w_submit_20260610_1641_score26.00.tar.gz` and compare to submit 1428
  What to do / Must NOT do: Unpack the second submit archive. Confirm it is a duplicate or near-duplicate of the 1428 submit by comparing file lists and md5sums. Must NOT modify either submit.
  Parallelization: Wave 1 | Blocked by: — | Blocks: T5
  References (executor has NO interview context - be exhaustive): `/home/1ctnltug/atec2026/atec_taskA_b2w_submit_20260610_1641_score26.00.tar.gz`
  Acceptance criteria (agent-executable):
    - Top-level contains `solution.py`, `policy.pt`, `requirements.txt`
    - `md5sum` of all three files equals those from T3
    - `diff -rq <t3_dir> <t4_dir>` reports no file differences
    - Capture summary and duplicate verdict in `.omo/evidence/task-4-analyze-root-archives.md`
  QA scenarios: happy: bit-identical to T3; failure: checksum mismatch triggers note of divergence. Evidence `.omo/evidence/task-4-analyze-root-archives.md`
  Commit: N | —

- [x] 5. Generate consolidated root-archive analysis report
  What to do / Must NOT do: Read the four per-archive evidence files and write `.omo/evidence/root-archive-analysis.md` with a classification table, checksum cross-reference, duplicate findings, and plain-English description of each package. Must NOT include runnable commands that mutate the workspace.
  Parallelization: Wave 2 | Blocked by: T1–T4 | Blocks: F1–F4
  References (executor has NO interview context - be exhaustive): `.omo/evidence/task-{1..4}-analyze-root-archives.md`
  Acceptance criteria (agent-executable):
    - Report file exists and contains one row per archive with: archive name, unpacked size, top-level dir, key artifacts, `policy.pt` md5, classification
    - Report explicitly states whether the two submit archives are identical
    - Report is ≤ 200 lines and written in Chinese (matching user language)
  QA scenarios: happy: report is created and readable; failure: missing evidence file → abort. Evidence `.omo/evidence/root-archive-analysis.md`
  Commit: N | —

## Final verification wave
> Runs in parallel after ALL todos. ALL must APPROVE. Surface results and wait for the user's explicit okay before declaring complete.
- [x] F1. Plan compliance audit: Confirm all four archives were extracted, originals untouched, and no project files modified. Evidence: `ls -la /home/1ctnltug/atec2026/*.tar.gz` and `git status --short` show no changes in `ATEC2026/`, `IsaacLab/`, `scripts/`, `demo/`.
- [x] F2. Code quality review: N/A (no code changes). Confirm no executable code was run.
- [x] F3. Real manual QA: Open `.omo/evidence/root-archive-analysis.md` and verify it matches the observed archive contents and checksums.
- [x] F4. Scope fidelity: Confirm the analysis did not install packages, run Isaac Sim, or submit anything.

## Commit strategy
No git commits. This is a read-only analysis task.

## Success criteria
- All four root archives are extracted to a dedicated analysis directory without altering the originals.
- Each archive’s purpose, key contents, and `policy.pt` checksum are documented.
- The duplicate relationship between the two submit archives is identified.
- A single consolidated Chinese report exists under `.omo/evidence/root-archive-analysis.md`.
