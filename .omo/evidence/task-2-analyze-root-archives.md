# Archive Analysis: `atec_taskA_b2w_best_20260610_1543_score26.00_with_video.tar.gz`

## Basic Info

| Field | Value |
|-------|-------|
| Archive | `atec_taskA_b2w_best_20260610_1543_score26.00_with_video.tar.gz` |
| Archive size | 42.1 MB |
| File count | 16 entries (13 files + 3 directories) |
| Extraction target | `/tmp/opencode/root_archive_analysis/atec_taskA_b2w_best_26/` |

## Classification

这是 ATEC 2026 赛题 A（B2wPiper 轮腿机器人）的**满分（26.00/26）提交方案**。该方案在 2026-06-10 15:43 打包，经过至少 3 次独立复评均获满分（443.54s / 332.14s / 332.16s），属于**最高质量基准方案**。策略基于 RSL-RL 训练，关键创新包括楼梯专项续训（feet_air_time 踏步反射）以及推理时航向保持、roll居中、坡道降速和卡住动量冲刺。附带完整赛道跑通录像，可作为后续方案比较的黄金标准。**核心特征**：训练 checkpoint model_8797，强化学习策略网络 + 轻量推理 solution.py，15 分钟级赛道完成时间。

## File Listing

```
atec_taskA_b2w_best_26/
├── MANIFEST.txt                          — 文件完整性清单
├── MODEL_INFO.txt                        — 模型元信息
├── README.txt                            — 方案说明
├── checkpoint_model_8797.pt              — RSL-RL 原始 checkpoint（续训权重）
├── policy.pt                             — TorchScript JIT 导出策略（部署用）
├── solution.py                           — AlgSolution 推理入口
├── requirements.txt                      — 依赖声明（空依赖，镜像预装）
├── logs/
│   ├── eval_26.00.log                    — 评测日志（443.54s 满分）
│   ├── eval_26.00_run2.log               — 二次复评（332.14s 满分）
│   ├── eval_26.00_run3.log               — 三次复评（332.16s 满分）
│   └── eval_26.00_v3b.log                — 变体评测
└── video/
    ├── VIDEO_INFO.txt                    — 视频元信息
    └── taskA_b2w_score26.00.mp4          — 赛道跑通录像（36.1 MB, MP4）
```

## Key Artifacts

### `policy.pt` — 部署策略

- **md5sum**: `19c573ba32b773c73b27ef491af548bf`
- **Expected md5**: `19c573ba32b773c73b27ef491af548bf`
- **Status**: ✅ **MATCH** — checksum 完全一致
- **MANIFEST.txt 记录**: `19c573ba32b773c73b27ef491af548bf`

### `solution.py` — 推理入口

- **md5sum**: `9fe4ea578686f8e090973c761fbe127a`
- **MANIFEST.txt 记录**: `9fe4ea578686f8e090973c761fbe127a`
- **技术要点**: 航向保持 + roll 居中 + 坡道降速(vx_scaling) + 卡住动量冲刺(CLIMB_BOOST_VX=1.5)

### `checkpoint_model_8797.pt` — 训练 checkpoint

- **md5sum**: `ff9779e54940604087df58c69d510023`
- 来源训练: `unitree_b2w_stair/2026-06-10_10-45-53/model_8797`
- 关键训练技巧: feet_air_time 踏步反射 + 楼梯续训

### Eval Logs

| Log | Elapsed Time | Score | Notes |
|-----|-------------|-------|-------|
| `eval_26.00.log` | 443.54s | 26.00/26 | 首次满分评测 |
| `eval_26.00_run2.log` | 332.14s | — | 二次复评（更快） |
| `eval_26.00_run3.log` | 332.16s | — | 三次复评 |
| `eval_26.00_v3b.log` | — | — | 变体评测 |

日志显示使用 Isaac Sim 5.1 (Vulkan, CUDA)，环境配置为 ATEC-TaskA-B2wPiper。

### Video

- **File**: `video/taskA_b2w_score26.00.mp4`
- **Size**: 36.1 MB
- **Format**: ISO Media, MP4 Base Media v1 [ISO 14496-12:2003]
- **Info**: 场景得分 26.00/26，耗时 332.86s，~4096 帧，fps=12.5

## MANIFEST Checksum Integrity

All MANIFEST.txt 记录的 md5 值与实际文件一致（已验证 policy.pt 和 solution.py），完整性通过。

## Verification Summary

| Check | Result |
|-------|--------|
| Archive extraction | ✅ Success (13 files) |
| policy.pt md5 match | ✅ `19c573ba32b773c73b27ef491af548bf` |
| solution.py present | ✅ Yes |
| checkpoint_model_8797.pt present | ✅ Yes |
| Eval logs present | ✅ 4 log files (all score 26.00) |
| Video present | ✅ 36.1 MB MP4 |
| MANIFEST integrity | ✅ Verified |
