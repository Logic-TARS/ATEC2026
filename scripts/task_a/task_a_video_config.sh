#!/usr/bin/env bash

# Task A 录视频配置。以后主要改这个文件就行。

# 视频长度，单位是“帧”。当前录制约 50 帧/秒。
# 500 约 10 秒，1500 约 30 秒，3000 约 60 秒。
ATEC_VIDEO_LENGTH=3000

# 摄像机模式：
# follow = 摄像机固定在狗后方，并跟随狗运动
# fixed  = 远处固定俯视相机
# none   = 不手动控制相机
ATEC_CAMERA_MODE=follow

# 是否禁用 Fabric：
# 0 = 正常，推荐，视频里狗会视觉同步移动
# 1 = 回退模式，可能出现“物理在动但视频里狗不动”
ATEC_DISABLE_FABRIC=0

# 视频最终保存目录。默认放在 artifacts 下面，省得翻很深的 logs 目录。
ATEC_VIDEO_OUTPUT_DIR=/home/1ctnltug/atec2026/artifacts/task_a_videos
