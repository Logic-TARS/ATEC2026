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
