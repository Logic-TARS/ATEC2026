# 根目录 4 个安装包分析报告

## 总览表

| 文件名 | 解压大小 | 文件数 | 顶级目录 | 分类 | policy.pt md5 |
|---|---|---|---|---|---|
| `atec_b2w_handover_20260610_1427_score26.00.tar.gz` | 27M | 41 | `atec_b2w_handover/` | 满分交接包（完整方案+权重+文档+脚本+评测日志） | `19c573ba32b773c73b27ef491af548bf` |
| `atec_taskA_b2w_best_20260610_1543_score26.00_with_video.tar.gz` | 47M | 13 | `atec_taskA_b2w_best_26/` | 满分评测记录包（含 36M 赛道录像） | `19c573ba32b773c73b27ef491af548bf` |
| `atec_taskA_b2w_submit_20260610_1428_score26.00.tar.gz` | 796K | 3 | `atec_taskA_b2w_submit_1428/` | 平台提交包（最小化提价结构） | `19c573ba32b773c73b27ef491af548bf` |
| `atec_taskA_b2w_submit_20260610_1641_score26.00.tar.gz` | 796K | 3 | `atec_taskA_b2w_submit_1641/` | 平台提交包（与 1428 完全重复） | `19c573ba32b773c73b27ef491af548bf` |

## 各包详细信息

### 包 1：`atec_b2w_handover_*.tar.gz` — 满分交接包

**这是本次分析的原始包中最完整、最核心的一个。** 解压后 27M、41 个文件，结构包含 6 个目录：`submission/`（可直接提交平台的部署入口：`solution.py` + `policy.pt` + `requirements.txt` + `README.md`）、`checkpoints/`（7 个训练中间权重，包含 3 个 RSL-RL raw checkpoint 和 4 个 JIT 导出）、`training_overlay/`（16 个训练配置叠加文件，覆盖平底/粗糙/楼梯/教师/蒸馏等各种环境配置）、`scripts/`（6 个自动化脚本）、`eval_logs/`（从 18.10 到 26.00 分的完整评测日志，产出迭代轨迹清晰）以及 `docs/`（TRAINING.md 和 DEPLOYMENT.md 文档）。

评测得分 **26.00/26 满分**（五段：平地 +2、粗糙 +4、上坡 +8、下坡 +8、终点 +4，用时 443.54s）。该包的 `submission/` 目录与另外两个提交包（1428/1641）的文件完全一致，是原始素材包。

### 包 2：`atec_taskA_b2w_best_*_with_video.tar.gz` — 满分评测记录包

**这是包含赛道录像的补充证据包。** 解压后 47M（其中视频占 36.1 MB）、13 个文件。除了与上面相同的 `solution.py`、`policy.pt`、`requirements.txt` 外，额外提供 4 份评测日志（首次满分 443.54s + 三次复评 332.14s/332.16s/变体）和一个完整赛道跑通 MP4 录像（36.1 MB, 12.5fps, ~4096 帧）。另含原始 RSL-RL checkpoint `model_8797`（5.6M，md5: `ff9779e5`）作为训练存档。**作用**：为评分结果提供可复现的视频佐证。

### 包 3：`atec_taskA_b2w_submit_1428_*.tar.gz` — 平台提交包

**这是最简化的 ATEC 平台提交结构。** 解压后仅 796K、3 个文件：`solution.py`（159 行部署推理代码，含航向保持、横向阻尼、坡道降速和卡住冲刺等规则兜底）、`policy.pt`（778.1K JIT 导出权重）和 `requirements.txt`（空依赖）。这是标准提价格式，与交接包 `submission/` 目录内容完全一致。

### 包 4：`atec_taskA_b2w_submit_1641_*.tar.gz` — 重复提交包

**此为 1428 的字节级重复。** 896K、3 个文件，`diff -rq` 结果无差异（exit 0），所有文件 md5 完全匹配。`requirements.txt` 也与 1428 一致（md5: `90fa3a`）。推测为同一提交的备份或重传，本身无新增信息量。

## 交叉引用

### 全局共享的政策权重

所有 4 个包的 `policy.pt` 共享**完全相同**的 md5 哈希值：

```
19c573ba32b773c73b27ef491af548bf
```

### solution.py 全局匹配

所有 4 个包的 `solution.py` 共享相同的 md5：

```
9fe4ea578686f8e090973c761fbe127a
```

### requirements.txt 三包一致

除 best 包不包含 `requirements.txt` 外，其余 3 包的文件完全一致（md5: `90fa3a985d79808229411109f1f32b0b`）。

### 重复发现

- **1428 vs 1641**：字节级重复（`diff -r` 无任何输出，exit 0）
- **handover `submission/` 与 1428**：4 个文件（solution.py, policy.pt, requirements.txt, README.md）与 1428 完全一致，其中 `README.md` 为 handover 独有

### 原始存档保留状态

所有原始 `.tar.gz` 文件保持解压前的原始位置和内容不变。本次分析仅进行只读解包和 md5 校验，未执行任何代码、未修改工作区内容。

## 结论

四个根目录安装包实质上是**同一满分方案（Task A B2wPiper，26.00/26）的不同打包形式**：一个完整交接包（含全量中间产物和文档）、一个带录像的评测复现包、和两个平台提交包（互为重复）。核心产出物 `policy.pt` 和 `solution.py` 在全部包中完全一致，证明它们是同一策略的多份拷贝，而非独立方案。
