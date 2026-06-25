# ATEC 2026 工作区说明

这个目录不是一个单独的 Python 包，而是一个“工作区”。里面同时放了比赛项目、IsaacLab 框架、我给你写的本地脚本、视频和备份文件。

如果你只想跑 Task A，看下面这一段就够了。

## 我现在该从哪里开始？

先进入工作区：

```bash
cd /home/1ctnltug/atec2026
```

录一段 Task A B2W 视频：

```bash
./scripts/task_a/record_task_a_b2w_video.sh
```

运行 Task A B2W，不一定录视频：

```bash
./scripts/task_a/run_task_a_b2w_gui.sh
```

如果要打开 GUI 画面运行：

```bash
ATEC_GUI=1 ./scripts/task_a/run_task_a_b2w_gui.sh
```

视频参数在这里改：

```text
scripts/task_a/task_a_video_config.sh
```

视频会保存到这里：

```text
artifacts/task_a_videos/
```

最新视频快捷入口在这里：

```text
artifacts/latest_task_a_video.mp4
```

## 继续训练坑洼地直行策略

从已有平地模型继续训练到粗糙地形：

```bash
./scripts/training/train_b2_rough_straight_from_flat.sh
```

先短跑测试，不想一上来训很久：

```bash
ATEC_ROUGH_STRAIGHT_ITERS=200 ATEC_TRAIN_NUM_ENVS=1024 ./scripts/training/train_b2_rough_straight_from_flat.sh
```

训练完以后，把最新模型导出并替换 `demo/policy.pt`：

```bash
./scripts/training/export_latest_rough_straight_policy_to_demo.sh
```

然后再录视频检查效果：

```bash
./scripts/task_a/record_task_a_b2w_video.sh
```

## 训练 B2W 平地全向策略

从零开始训练 B2W 平地全向运动策略（用于 Task D）：

```bash
./scripts/training/train_b2w_flat_omni.sh
```

先短跑测试：

```bash
ATEC_B2W_FLAT_OMNI_ITERS=10 ATEC_TRAIN_NUM_ENVS=64 ./scripts/training/train_b2w_flat_omni.sh
```

训练完后导出到 `demo/policy_b2w_flat_omni.pt`：

```bash
./scripts/training/export_b2w_flat_omni_policy_to_demo.sh
```

从官方 checkpoint 精调 Task D 策略（`easy|medium|official`）：

```bash
./scripts/training/train_taskd_finetune.sh official
```

精调完后导出到 `demo/policy_taskd_finetuned.pt`：

```bash
./scripts/training/export_taskd_finetune_policy.sh official
```

## 每个文件夹是干什么的？

```text
ATEC2026/
```

比赛官方项目。真正的任务代码、提交代码、训练入口都在这里。平时不要随便移动这个目录。

最常改的是：

```text
ATEC2026/demo/solution.py
ATEC2026/demo/policy.pt
```

```text
IsaacLab/
```

Isaac Lab 框架。它不是我们的比赛代码，但训练和仿真要依赖它。

为什么放根目录？因为这个工作区用的是本地 IsaacLab 2.3.2，而系统里还有另一个 `/opt/IsaacLab`。放在根目录可以明确告诉脚本：ATEC 用这个版本，不要混到系统版本里。

```text
scripts/
```

我给你写的本地快捷脚本。你日常基本只需要跑这里面的脚本。

```text
scripts/env/
```

环境激活脚本。一般不用手动跑，因为其他脚本会自动 source 它们。需要手动进环境时运行：

```bash
source /home/1ctnltug/atec2026/scripts/env/activate_atec2026_sim.sh
```

```text
scripts/task_a/
```

Task A 相关脚本和配置。

里面这些最重要：

```text
record_task_a_b2w_video.sh   录视频
run_task_a_b2w_gui.sh        运行 Task A
task_a_video_config.sh       改视频长度、相机模式、输出目录
```

```text
scripts/training/
```

训练相关脚本：粗糙地形直行课程学习、B2W 平地全向策略训练、Task D 精调。

```text
artifacts/
```

生成出来的东西，不是源码。比如视频、提交包、最新视频快捷链接。

```text
archives/
```

原始压缩包和安装包。比如比赛 zip、Miniconda 安装器。一般不用打开，但先留着，环境坏了还能救。

```text
reports/
```

HTML 报告和临时分析结果。

```text
notes/
```

长文档、部署说明。

```text
SETUP_STATUS.md
```

这台机器当前环境状态记录。忘了 Python 环境、IsaacLab 路径时看它。

## 最常用的三个命令

录视频：

```bash
cd /home/1ctnltug/atec2026
./scripts/task_a/record_task_a_b2w_video.sh
```

跑仿真：

```bash
cd /home/1ctnltug/atec2026
./scripts/task_a/run_task_a_b2w_gui.sh
```

训练坑洼地直行：

```bash
cd /home/1ctnltug/atec2026
./scripts/training/train_b2_rough_straight_from_flat.sh
```
