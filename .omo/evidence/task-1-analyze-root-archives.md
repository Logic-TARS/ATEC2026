# Task 1: Analyze Root Archives — atec_b2w_handover

## Archive Info

| Field | Value |
|---|---|
| Filename | `atec_b2w_handover_20260610_1427_score26.00.tar.gz` |
| Size | 20.0 MB |
| Extracted to | `/tmp/opencode/root_archive_analysis/atec_b2w_handover/` |
| Total files | 41 |

## Classification

该存档为 **ATEC 2026 仿真挑战赛 Task A（越野导航）B2wPiper 交接包**，包含一套在 Unitree B2W + Piper 轮足四足机器人上取得赛题满分 26/26 的完整方案。内容涵盖可直接提交的推理代码与策略权重（`submission/`）、多版训练中间权重与实验室头（`checkpoints/`，含 JIT 导出和原始 RSL-RL checkpoint）、对官方训练仓库的配置修改叠加层（`training_overlay/`）、自动化训练/导出/评测流水线脚本（`scripts/`）、以及从 18.10 到 26.00 分的完整评测日志（`eval_logs/`）。核心策略采用「踏步反射（feet_air_time）+ 楼梯课程续训」的 RL 训练方案，部署侧附加航向保持、roll 居中、坡道降速和卡住时冲刺指令的规则兜底层。

## Key Contents

### `submission/` (4 files)
- `solution.py` — AlgSolution 部署推理代码（159 lines，含航向保持/坡道降速/卡住冲刺）
- `policy.pt` — JIT 策略权重（778.1K）
- `requirements.txt` — 空依赖（平台已预装 torch+numpy）
- `README.md` — 方案说明

### `checkpoints/` (7 files, 3 raw + 4 JIT)
| File | Size | Type |
|---|---|---|
| `model_5798_v2_raw.pt` | 5.6M | RSL-RL raw checkpoint |
| `model_8797_stairstep_raw.pt` | 5.6M | RSL-RL raw checkpoint |
| `model_8797_stepstair_v3_raw.pt` | 5.6M | RSL-RL raw checkpoint |
| `policy_stairstep_2175_jit.pt` | 778.1K | JIT export |
| `policy_stairstep_8797_jit.pt` | 778.1K | JIT export |
| `policy_stepstair_v3_jit.pt` | 778.1K | JIT export |
| `policy_v2_5798_jit.pt` | 778.1K | JIT export |

### `training_overlay/` (16 files)
- Training configs for unitree_b2w (flat/rough/stair/taska/taska_slope/hard/teacher/distill env cfgs, PPO cfg)
- `demo/` — alternative solution.py + policy.pt
- `scripts/play_atec_task.py` — modified play script

### `scripts/` (6 files)
- `env.sh`, `auto_queue.sh`, `export_student.py`, `monitor_eval.sh`, `pipeline_monitor.sh`, `watch_stair_step.sh`

### `eval_logs/` (3 files, 120K/31K/44K lines)
- `eval_18.10.log`, `eval_21.75.log`, `eval_26.00.log`

## Checksum Verification

```text
Expected: 19c573ba32b773c73b27ef491af548bf
Got:      19c573ba32b773c73b27ef491af548bf
Status:   ✅ MATCH
```

(`md5sum /tmp/opencode/root_archive_analysis/atec_b2w_handover/submission/policy.pt`)

MANIFEST.txt also confirms this md5 for `./submission/policy.pt`, `./checkpoints/policy_stairstep_2175_jit.pt`, `./checkpoints/policy_stairstep_8797_jit.pt`, `./checkpoints/policy_stepstair_v3_jit.pt`, and `./training_overlay/demo/policy.pt`.

## Notable Observations

1. **满分方案**（26/26）：在 Task A 的平地 +2、粗糙 +4、上坡 +8、下坡 +8、终点 +4 五段全部拿满，用时 443.54s。
2. **多版本迭代**：从 ~5.84 → 18.10 → 21.75 → 26.00，实验路径清晰（v2 粗糙 → 楼梯专项 → 踏步反射 → 踏步+卡住冲刺）。
3. **双备份**：`submission/policy.pt` 与 `checkpoints/policy_*_jit.pt` 中 3 个 JIT 文件共享同一 md5，说明它们本质是同一策略权重的拷贝。
4. **两个 model_8797 raw checkpoint 的 md5 相同**（`ff9779e5`），表明使用了同一原始 checkpoint 进行不同方向的 fine-tune（stairstep vs stepstair_v3），但最终导出 JIT 的 md5 一致，暗示后续导出流程收敛到同一权重选择。
5. **训练配置完善**：`training_overlay/` 包含完整的单元树 B2W 训练配置族（flat/rough/stair/taska/taska_slope/hard/teacher/distill），覆盖从平坦到极端地形的全部场景。
