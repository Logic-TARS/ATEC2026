---
slug: analyze-root-archives
status: awaiting-approval
intent: clear
pending-action: write .omo/plans/analyze-root-archives.md
approach: Read-only extraction of the four root tar.gz archives into a temporary analysis directory; inspect README/MANIFEST/MODEL_INFO/DEPLOYMENT files and checksum key artifacts; produce a consolidated Chinese report.
---

# Draft: analyze-root-archives

## Components (topology ledger)
<!-- Lock the SHAPE before depth. One row per top-level component that can succeed or fail independently. -->
<!-- id | outcome (one line) | status: active|deferred | evidence path -->
- handover | Extract & classify the 26.00 handover package (full code + weights + docs) | active | .omo/evidence/task-1-analyze-root-archives.md
- best_with_video | Extract & classify the best-run package with evaluation logs and video | active | .omo/evidence/task-2-analyze-root-archives.md
- submit_1428 | Extract & classify the 14:28 submission package | active | .omo/evidence/task-3-analyze-root-archives.md
- submit_1641 | Extract & classify the 16:41 submission package and compare to 1428 | active | .omo/evidence/task-4-analyze-root-archives.md
- report | Consolidated Chinese report summarizing all four archives | active | .omo/evidence/root-archive-analysis.md

## Open assumptions (announced defaults)
<!-- Record any default you adopt instead of asking, so the user can veto it at the gate. -->
<!-- assumption | adopted default | rationale | reversible? -->
- Extraction target | `/tmp/opencode/root_archive_analysis` | Keeps the workspace root clean and avoids overwriting project files | reversible
- Report language | Chinese | Matches the user’s query language | reversible
- Scope | Inspect only; do not execute code or install dependencies | User asked “decompress and analyze what they are,” not run them | reversible

## Findings (cited - path:lines)
- `/home/1ctnltug/atec2026/atec_b2w_handover_20260610_1427_score26.00.tar.gz` contains `atec_b2w_handover/` with `README.md` (lines 1-94), `DEPLOYMENT.md`, `TRAINING.md`, `submission/` (solution.py + policy.pt), `checkpoints/`, `training_overlay/`, `scripts/`, `eval_logs/`.
- `/home/1ctnltug/atec2026/atec_taskA_b2w_best_20260610_1543_score26.00_with_video.tar.gz` contains `atec_taskA_b2w_best_26/` with `README.txt`, `MANIFEST.txt`, `MODEL_INFO.txt`, `policy.pt`, `solution.py`, `checkpoint_model_8797.pt`, `logs/eval_26.00*.log`, `video/taskA_b2w_score26.00.mp4`.
- `/home/1ctnltug/atec2026/atec_taskA_b2w_submit_20260610_1428_score26.00.tar.gz` contains `atec_taskA_b2w/{solution.py, policy.pt, requirements.txt}`.
- `/home/1ctnltug/atec2026/atec_taskA_b2w_submit_20260610_1641_score26.00.tar.gz` contains bare `{solution.py, policy.pt, requirements.txt}` at the archive root.
- All submit/policy files share md5 `19c573ba32b773c73b27ef491af548bf`; all solution.py md5 `9fe4ea578686f8e090973c761fbe127a`.

## Decisions (with rationale)
- Use `/tmp/opencode/root_archive_analysis` as extraction target to avoid mutating the workspace root.
- Do not ask the user for a target directory; the default is reversible and clearly safer.
- Treat the request as CLEAR: outcome is “decompress and identify the four packages,” with only the target directory as a preference best handled by default.

## Scope IN
- Extract and catalog the four root archives.
- Read metadata and list key files.
- Compute checksums for key artifacts.
- Identify duplicate submit packages.
- Write a consolidated Chinese report.

## Scope OUT (Must NOT have)
- No modification/deletion of original archives.
- No installation or execution of code.
- No simulation, video playback, or platform submission.
- No changes to tracked project files.

## Open questions
- None. All unknowns resolved by listing and reading metadata.

## Approval gate
status: awaiting-approval
<!-- When exploration is exhausted and unknowns are answered, set status: awaiting-approval. -->
<!-- That durable record is the loop guard: on a later turn read it and resume at the gate instead of re-running exploration. -->
- User must approve before the worker extracts the archives and writes the report.
- Default extraction directory can be overridden if the user replies with a different path.
