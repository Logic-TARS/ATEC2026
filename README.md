# 🤖 ATEC 2026 具身强化学习仿真挑战项目

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![Isaac Lab](https://img.shields.io/badge/Isaac%20Lab-v2.3.2-76B900)
![RL](https://img.shields.io/badge/RL-RSL--RL%20PPO-FF6F00)
![Robot](https://img.shields.io/badge/Robot-Unitree%20B2W%20%2B%20AgileX%20Piper-4B8BBE)
![Deployment](https://img.shields.io/badge/Deployment-AlgSolution.predicts-555555)

> 本仓库用于展示我在 **ATEC 2026 具身智能仿真挑战赛** 中的强化学习训练、仿真部署与官方评测接口适配工程。  
> 项目面向 **Unitree B2W + AgileX Piper** 机器人，基于 **Isaac Lab v2.3.2 + RSL-RL PPO** 构建 Task A 越野导航、Task D 推箱越障、Task F 平地预训练与提交部署链路。

<p align="center">
  <img src="ATEC2026/doc/b2w.png" width="220" alt="Unitree B2W with Piper">
  <img src="ATEC2026/doc/task_a.gif" width="220" alt="Task A off-road navigation">
  <img src="ATEC2026/doc/task_d.gif" width="220" alt="Task D obstacle traversal">
</p>

---

## ✅ 项目亮点 / 可验证结果

- **完整具身 RL 工程链路**：完成 Isaac Lab 环境注册、RSL-RL PPO 训练、策略导出、Task A / Task D 本地播放、视频录制与官方提交接口适配。
- **多阶段课程学习**：构建 flat locomotion → rough straight walking → rough omni B2W policy → Task D fine-tuning 的递进式训练流程。
- **Task D 推箱越障专项设计**：围绕箱子接触、推箱、平台导航和终点通过设计任务观测、奖励与高层控制逻辑。
- **61D 观测 + 16D 动作接口**：Task D 保持 61D policy observation 与 16D locomotion action，便于从 Task F 平地预训练 checkpoint 迁移到官方 Task D。
- **16D → 24D 官方动作适配**：将训练得到的 12 维腿部动作 + 4 维轮速动作扩展到官方 24D 动作接口，Piper 手臂 8 维动作固定 / 置零。
- **在线控制与鲁棒性逻辑**：`solution.py` 集成高层状态机、LiDAR / height-scan 修正、heading lock、speed correction、stuck recovery 与 score-aware phase switching。
- **可复现实验脚本**：提供环境激活、smoke test、训练、回放、视频录制、policy export、submission packaging 等脚本。

---

## 🧩 问题—方法—效果

| 问题 | 解决方法 | 产生效果 |
|---|---|---|
| Task D 推箱越障任务同时包含接近箱子、稳定接触、推箱、过坑 / 上平台和终点通过，直接端到端训练难度高 | 将任务拆分为 flat pre-train、rough omni locomotion、Task D easy / medium / official fine-tuning 等阶段，采用课程学习逐步增加任务难度 | 降低从零训练复杂任务的难度，形成从基础运动到官方 Task D 的递进式训练流程 |
| 官方评测接口为 24D action，但训练策略更适合输出紧凑的 16D locomotion action | 设计 deployment adapter：12 维 leg position + 4 维 wheel velocity 来自 policy，8 维 Piper arm action 固定 / 置零 | 解决训练动作空间与官方提交动作空间不一致的问题，使策略可直接接入 `AlgSolution.predicts` |
| Task D 中机器人容易出现推箱偏航、接触丢失、速度不足或卡死 | 在 `solution.py` 中加入高层状态机、heading lock、speed correction、LiDAR / height-scan 修正和 stuck recovery | 提升策略在官方评测接口下的鲁棒性，减少卡死和错误阶段切换 |
| Task F 平地预训练和官方 Task D 容易因观测 / 动作维度不一致导致 checkpoint 迁移困难 | 统一 Task F 与 Task D 的 61D 观测和 16D 动作维度，保证 actor checkpoint 可以 exact match 加载 | 支持从平地推箱预训练平滑迁移到官方 Task D fine-tuning |
| Isaac Lab / Isaac Sim 训练、播放、录像和 GUI 依赖复杂，复现实验成本高 | 封装 `scripts/env`、`scripts/training`、`scripts/task_a`、`scripts/task_d` 和通用 `train_env.sh` / `play_env.sh` / `view_env.sh` | 降低复现实验门槛，支持 smoke test、视频验证和提交前本地检查 |

---

## 📊 实验结果 / 工程结果

| 指标 | 结果 |
|---|---|
| 比赛名称 | ATEC 2026 Simulation Challenge |
| 任务方向 | Task A 越野导航；Task D 推箱越障；Task F 平地预训练 |
| 仿真平台 | Isaac Sim + Isaac Lab v2.3.2 |
| 强化学习算法 | RSL-RL PPO |
| 机器人平台 | Unitree B2W + AgileX Piper |
| Task D policy observation | 61D |
| Task D locomotion action | 16D |
| 官方 action interface | 24D |
| 动作适配方式 | 16D policy output → 12D legs + 4D wheels + 8D fixed arm |
| 训练流程 | flat locomotion → rough straight → rough omni → Task D fine-tuning |
| 复现入口 | `scripts/train_env.sh`、`scripts/play_env.sh`、`scripts/view_env.sh` |
| 部署入口 | `ATEC2026/demo/solution.py` / `AlgSolution.predicts` |
| 评测得分 / 排名 | 待补充 |
| Task D 任务完成率 | 待补充 |
| 最优提交包 | 待补充 |

> 可继续补充：最终评测得分、排行榜排名、Task A / Task D 回放视频、训练曲线、submission zip 版本、关键消融对比。

---

## 🧾 简历表述

> 面向 ATEC 2026 具身智能仿真挑战赛 Task A 越野导航与 Task D 推箱越障任务，基于 Isaac Lab v2.3.2 与 RSL-RL PPO 搭建 Unitree B2W + AgileX Piper 强化学习训练与部署链路，完成环境注册、课程学习训练、策略导出、本地回放和官方 `AlgSolution.predicts` 接口适配；针对训练策略 16D 输出与官方 24D 动作接口不一致的问题，设计 deployment adapter，将 12 维腿部动作与 4 维轮速动作扩展到官方动作空间；针对 Task D 接触不稳定、推箱偏航和卡死问题，在 `solution.py` 中实现高层状态机、LiDAR / height-scan 修正、heading lock、speed correction 与 stuck recovery，提升策略在评测接口下的鲁棒性。

---

## 🎯 任务说明与技术难点

### Task A：越野导航

Task A 关注 B2W + Piper 机器人在复杂地形中的运动能力，重点验证策略在非平整地形上的稳定移动、姿态保持和导航能力。

### Task D：推箱越障

Task D 要求机器人完成推箱、越障、平台导航和终点通过等组合行为。该任务不是单纯 locomotion，而是具有明显阶段结构的 loco-manipulation 任务。

主要难点：

- **长时序任务**：从接近箱子到最终通过终点，任务链路长，直接强化学习训练难度高。
- **接触不稳定**：推箱需要稳定接触，机器人容易顶偏、滑开或丢失箱子。
- **动作空间不一致**：训练时希望使用紧凑 16D locomotion action，但官方接口要求 24D action。
- **地形和障碍影响**：坑、平台和高度变化会导致速度衰减、偏航和卡死。
- **部署约束明确**：最终必须通过 `AlgSolution.predicts(obs, current_score)` 返回官方动作格式。

---

## 🧠 核心方案

### 1. 多阶段课程学习

训练流程按任务难度逐步递进：

```text
flat locomotion
  -> rough straight walking
  -> rough omni B2W policy
  -> Task F flat push pre-training
  -> Task D easy / medium / official fine-tuning
  -> policy export
  -> official submission adapter
```

对应脚本：

```bash
# Rough straight walking from a flat checkpoint
./scripts/training/train_b2_rough_straight_from_flat.sh

# B2W + Piper rough omni policy
./scripts/training/train_b2w_rough_omni_from_straight.sh

# Task D official fine-tuning
./scripts/training/train_taskd_finetune.sh official

# Task F flat pre-training -> official Task D transfer
ATEC_TASKD_ITERS=7000 ATEC_TRAIN_NUM_ENVS=1024 \
  ./scripts/training/train_taskd_from_flat_pretrain.sh
```

### 2. Task D 观测与动作设计

Task D 采用紧凑的 locomotion policy 形式：

| 模块 | 维度 | 说明 |
|---|---:|---|
| Policy observation | 61D | 包含机器人状态、命令、关节状态、历史动作和任务相关信息 |
| Locomotion action | 16D | 12 维腿部动作 + 4 维轮速动作 |
| Official action | 24D | 12 维腿部动作 + 4 维轮速动作 + 8 维 Piper 手臂动作 |

这样做的目的：

- policy 只学习和移动 / 推箱直接相关的 leg + wheel 控制；
- Piper arm 在该任务中固定，减少不必要的动作维度；
- 训练策略与官方评测接口通过 adapter 解耦。

### 3. 官方提交接口适配

官方评测调用：

```python
class AlgSolution:
    def predicts(self, obs, current_score):
        return {"action": action, "giveup": False}
```

本项目将训练策略输出映射到官方动作空间：

| Slice | 含义 | 部署行为 |
|---|---|---|
| `0:12` | Leg joint position commands | 来自 locomotion policy |
| `12:16` | Wheel velocity commands | 来自 locomotion policy |
| `16:24` | Piper arm position commands | 固定 / 置零 |

### 4. `solution.py` 在线控制逻辑

`ATEC2026/demo/solution.py` 是官方提交入口，集成：

- policy loading / inference；
- Task D high-level state machine；
- approach box / push box / nav platform / climb finish 阶段切换；
- scripted path / side push / pit push / teleop / waypoint route 多种调试模式；
- heading lock；
- speed correction；
- LiDAR / height-scan correction；
- stuck recovery；
- score-aware phase switching；
- 16D policy action 到 24D official action 的映射。

---

## 🛠 技术栈

| Area | Tools / Components |
|---|---|
| Simulator | Isaac Sim, Isaac Lab v2.3.2 |
| RL | RSL-RL PPO, TorchScript policy export |
| Robots | Unitree B2 / B2W, AgileX Piper |
| Tasks | Task A off-road navigation, Task D box-pushing obstacle traversal, Task F flat pre-training |
| Engineering | Gym environment registration, shell automation, local video capture, submission adapter |

---

## 📁 系统结构

```text
ATEC2026/source/atec_rl_lab/
  train/locomotion/velocity/
    config/quadruped/unitree_b2/      custom training envs and PPO configs
    mdp/                              rewards, commands, observations, events

scripts/
  env/                                conda + Isaac Lab workspace activation
  training/                           curriculum training and export scripts
  task_a/, task_d/                    local play and video recording helpers

ATEC2026/demo/
  solution.py                         official submission entrypoint
  policy*.pt                          local policy artifacts, not intended for GitHub release
```

训练环境注册入口：

```text
ATEC2026/source/atec_rl_lab/atec_rl_lab/train/locomotion/velocity/config/quadruped/unitree_b2/__init__.py
```

官方提交部署入口：

```text
ATEC2026/demo/solution.py
```

---

## 🚀 核心运行流程

### 1. 环境检查

```bash
source scripts/env/activate_atec2026_sim.sh
python scripts/list_envs.py
```

### 2. Task A：越野导航回放

```bash
python scripts/play_atec_task.py \
  --task ATEC-TaskA-B2wPiper \
  --headless --enable_cameras --disable_fabric \
  --num_envs 1 --debug
```

录像：

```bash
./scripts/task_a/record_task_a_b2w_video.sh
```

### 3. Task D：推箱越障训练

```bash
# Smoke test
ATEC_TASKD_ITERS=10 ATEC_TRAIN_NUM_ENVS=64 \
  ./scripts/training/train_taskd_finetune.sh official

# Full fine-tuning
./scripts/training/train_taskd_finetune.sh official

# Export policy
./scripts/training/export_taskd_finetune_policy.sh official
```

### 4. Task F 平地预训练 → Task D 迁移

Task F 被用作平地形 Task D 预训练阶段。它与官方 Task D 保持 **61D observation + 16D action**，因此 actor checkpoint 可以 exact match 加载。

```bash
ATEC_TASKD_ITERS=7000 ATEC_TRAIN_NUM_ENVS=1024 \
  ./scripts/training/train_taskd_from_flat_pretrain.sh
```

---

## 🧪 Key Environment IDs

| Environment | Purpose |
|---|---|
| `ATEC-Isaac-Velocity-Rough-Straight-Unitree-B2-v0` | B2 rough straight curriculum |
| `ATEC-Isaac-Velocity-Rough-Omni-B2W-Piper-v0` | B2W + Piper rough omni locomotion |
| `ATEC-Isaac-Velocity-Flat-Omni-B2W-Piper-v0` | Flat omni baseline / smoke tests |
| `ATEC-Isaac-TaskD-FixedArm-B2W-Easy-v0` | Task D easy fine-tuning |
| `ATEC-Isaac-TaskD-FixedArm-B2W-Medium-v0` | Task D medium fine-tuning |
| `ATEC-Isaac-TaskD-FixedArm-B2W-Official-v0` | Task D official fine-tuning |
| `ATEC-Isaac-Velocity-Flat-TaskF-Unitree-B2W-Piper-v0` | Flat Task D pre-training |
| `ATEC-Isaac-Velocity-ShortOmniDR-TaskF-Unitree-B2W-Piper-v0` | Task F domain-randomized hardening |

---

## 📝 后续可补充材料

为了让该项目更适合简历和面试展示，建议继续补充：

- 最终评测得分 / 排名 / 提交截图；
- Task A 与 Task D 策略回放 GIF / MP4；
- TensorBoard 训练曲线；
- Task D 失败案例与修复前后对比；
- baseline 对比：无课程学习 vs Task F 预训练迁移；
- 消融对比：无 heading lock / 无 speed correction / 无 stuck recovery；
- 最优 policy 与 submission zip 的版本说明。

---

## Repository Notes

- `ATEC2026/readme.md` is the original challenge README and is kept as upstream-facing reference.
- `scripts/README.md` is the command quick reference for training, playback, and environment viewing.
- `docs/workspace-runbook.md` keeps local workspace operations that are useful for reproducing runs but too detailed for a portfolio homepage.
- Large assets are intentionally excluded from GitHub: `IsaacLab/`, `ATEC2026/logs/`, `artifacts/`, `archives/`, robot model downloads, submission zips, and local checkpoints.

---

## License

The challenge project includes its own MIT license in `ATEC2026/LICENSE`. Third-party components such as Isaac Lab and robot assets follow their respective upstream licenses.
