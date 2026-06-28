# Scripts Quick Reference

这些脚本会自动激活 `atec2026-sim`，并切到 `ATEC2026/` 目录执行。直接在仓库根目录运行即可。

## List Environments

```bash
python ATEC2026/scripts/list_envs.py
```

## Train

训练必须启用 cameras，并建议关闭 Fabric：

```bash
./scripts/train_env.sh \
  --task ATEC-Isaac-Velocity-Flat-TaskF-Unitree-B2W-Piper-v0 \
  --headless --enable_cameras --disable_fabric \
  --num_envs 64 --max_iterations 1000
```

从已有 actor checkpoint 热启动：

```bash
./scripts/train_env.sh \
  --task ATEC-Isaac-Velocity-Flat-TaskF-Unitree-B2W-Piper-v0 \
  --headless --enable_cameras --disable_fabric \
  --num_envs 64 --max_iterations 1000 \
  --actor_checkpoint logs/rsl_rl/unitree_b2w_flat_omni/<run>/model_1000.pt
```

## Play

播放策略。`--checkpoint` 使用完整路径：

```bash
./scripts/play_env.sh \
  --task ATEC-Isaac-Velocity-Flat-Omni-B2W-Piper-v0 \
  --num_envs 1 \
  --checkpoint logs/rsl_rl/unitree_b2w_flat_omni/<run>/model_1000.pt
```

也可以让底层按 run 名查找：

```bash
./scripts/play_env.sh \
  --task ATEC-Isaac-Velocity-Flat-Omni-B2W-Piper-v0 \
  --num_envs 1 \
  --load_run <run>
```

录制播放视频：

```bash
./scripts/play_env.sh \
  --task ATEC-Isaac-Velocity-Flat-Omni-B2W-Piper-v0 \
  --num_envs 1 \
  --checkpoint logs/rsl_rl/unitree_b2w_flat_omni/<run>/model_1000.pt \
  --video --video_length 300 \
  --video_name play_flat_omni.mp4
```

输出默认复制到：

```text
artifacts/play_env_videos/
```

## View Env Config

查看环境初始状态建议用 headless 录像模式，避免当前 GUI `.kit` 依赖不匹配问题。

训练 Task F（实验性平 B2W + Piper 配置）：

```bash
./scripts/view_env.sh \
  --env_cfg atec_rl_lab.train.locomotion.velocity.config.quadruped.unitree_b2.task_f_flat_env_cfg:UnitreeB2WTaskFFlatEnvCfg \
  --video --video_length 300 --num_envs 1
```

Task A B2W：

```bash
./scripts/view_env.sh \
  --env_cfg atec_rl_lab.tasks.task_a.env_cfg:TaskAEnvB2WCfg \
  --video --video_length 300 --num_envs 1
```

输出默认保存到：

```text
artifacts/view_env_videos/
```

## TASK F as flat-terrain Task D pre-training

TASK F 的环境 ID 已重新用作平地形 Task D 预训练。相比官方 Task D，平地形上没有坑（pit）和平台（platform），让六足推箱策略先在简单地形上学会基础行为，再迁移到官方 Task D。

关键优势：TASK F 和官方 Task D 都是 **61D 观测 + 16D 动作**，维度完全一致。因此 checkpoint 加载时 actor 权重显示 "Exact match"，无需任何维度映射或补齐。

### Train

```bash
./scripts/train_env.sh \
  --task ATEC-Isaac-Velocity-Flat-TaskF-Unitree-B2W-Piper-v0 \
  --headless --enable_cameras --disable_fabric \
  --num_envs 64 --max_iterations 5000
```

输出保存到 `logs/rsl_rl/unitree_b2w_taskd_flat_pretrain/`。

环境变量覆盖语法：

```bash
ATEC_TRAIN_NUM_ENVS=64 ATEC_TASKD_FLAT_PRETRAIN_ITERS=5000 ./scripts/train_env.sh \
  --task ATEC-Isaac-Velocity-Flat-TaskF-Unitree-B2W-Piper-v0 \
  --headless --enable_cameras --disable_fabric \
  --num_envs 64 --max_iterations 5000
```

### Transfer to official Task D

用 `train_taskd_from_flat_pretrain.sh` 自动找到最新平预训练 checkpoint，通过 `--actor_checkpoint` 加载后继续在官方 Task D 上训练：

```bash
ATEC_TASKD_ITERS=7000 ATEC_TRAIN_NUM_ENVS=1024 ./scripts/training/train_taskd_from_flat_pretrain.sh
```

Transfer 时 actor 网络权重因为维度完全匹配，加载日志显示 "Exact match"。

### Smoke test

快速验证（10 iters, 64 envs）：

```bash
# Flat pre-training smoke test
ATEC_TRAIN_NUM_ENVS=64 ./scripts/train_env.sh \
  --task ATEC-Isaac-Velocity-Flat-TaskF-Unitree-B2W-Piper-v0 \
  --headless --enable_cameras --disable_fabric \
  --num_envs 64 --max_iterations 10

# Transfer to Task D official smoke test
ATEC_TASKD_ITERS=10 ATEC_TRAIN_NUM_ENVS=64 ./scripts/training/train_taskd_from_flat_pretrain.sh
```

## Useful Options

`train_env.sh`:

```text
--task <env_id> or --task=<env_id>
--headless
--enable_cameras
--disable_fabric
--num_envs <N>
--max_iterations <N>
--seed <N>
--actor_checkpoint <path>
--run_name <name>
```

`play_env.sh`:

```text
--task <env_id> or --task=<env_id>
--checkpoint <path>
--load_run <run>
--num_envs <N>
--camera_mode {follow,fixed,none}  Camera follow mode (default: follow)
--video
--video_length <N>
--video_output_dir <path>
--video_name <name>
```

`view_env.sh`:

```text
--env_cfg <module.path:ClassName>
--num_envs <N>
--video
--video_length <N>
--video_output_dir <path>
--video_name <name>
```

## Gotchas

- 训练：使用 `--headless --enable_cameras --disable_fabric`。
- 录像：不要加 `--disable_fabric`，否则画面可能不刷新。
- 普通 GUI：当前环境可能因 Isaac Sim / Isaac Lab GUI `.kit` 依赖不匹配失败，优先用 `--video`。
- checkpoint 必须匹配环境观测/动作维度。例如 53D flat omni 策略不能直接播放 56D Task F 配置。
- `play_env.sh --checkpoint` 是完整路径；不要写成 `--load_run <run> --checkpoint model_1000.pt`。
