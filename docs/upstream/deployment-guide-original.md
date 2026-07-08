> **⚠️ DEPRECATED — 原始部署指南，仅作参考**
>
> 本文件是比赛官方原始部署指南，保留用于参考。与当前工作区实际配置存在 4 处偏差，以 `CLAUDE.md` / `AGENTS.md` 为准：
> 1. **Python 版本**：本指南要求 3.12.x，实际工作区 `atec2026-sim` 环境使用 **Python 3.11**（匹配 Isaac Sim 二进制）。
> 2. **入口脚本**：本指南第 6 步提到 `demo/run_env.py`，实际入口是 **`scripts/play_atec_task.py`**。
> 3. **接口方法**：本指南代码示例写 `predict(self, obs, current_score)`，实际接口是 **`predicts(self, obs, current_score)`**。
> 4. **环境激活**：以 `scripts/env/activate_atec2026_sim.sh` 为准，而非本指南的 conda 直接安装流程。

# ATEC 2026 线上赛·赛道1 部署指南

最后更新：2026-05-21  
版本：v1.0

## 版本约束

必须严格对应，错版本直接报错。

| 软件 | 版本 | 说明 |
| --- | --- | --- |
| Isaac Lab | v2.3.2 | 官方指定唯一仿真平台 |
| Python | 3.12.x | 禁止使用 3.13 或更低版本 |
| PyTorch | 2.7.1 | 必须配合 CUDA 12.8.1 使用 |
| CUDA | 12.8.1 | 与 PyTorch 版本严格匹配 |
| NVIDIA 驱动 | >= 550.54.15 | 最低支持 CUDA 12.8 |

## 硬件最低要求

- GPU：NVIDIA RTX 3090/4090 或同等算力，>= 24GB 显存，仿真很吃显存
- CPU：16 核以上
- 内存：>= 64GB
- 磁盘：>= 100GB 空闲空间，仿真环境和模型文件占用较大

## 分步部署

### 步骤 1：安装系统基础依赖

```bash
sudo apt update && sudo apt install -y \
    build-essential \
    git \
    wget \
    curl \
    libgl1-mesa-glx \
    libegl1-mesa \
    libxrandr2 \
    libxrandr-dev \
    libxinerama-dev \
    libxcursor-dev \
    libxi-dev
```

### 步骤 2：创建 Python 虚拟环境

安装 Miniconda，如果未安装：

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

创建并激活环境：

```bash
conda create -n atec2026 python=3.12 -y
conda activate atec2026
```

验证 Python 版本，必须显示 3.12.x：

```bash
python --version
```

### 步骤 3：安装 PyTorch 和 CUDA

安装官方指定版本：

```bash
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
```

验证 CUDA 可用，必须返回 True：

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

### 步骤 4：安装 Isaac Lab v2.3.2

克隆官方仓库并切换到指定版本：

```bash
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
git checkout v2.3.2
```

安装 Isaac Sim 依赖：

```bash
./isaaclab.sh --install
```

验证安装，无报错即为成功：

```bash
./isaaclab.sh --verify
```

### 步骤 5：安装 ATEC 赛事扩展

克隆赛事代码仓库，替换为官方提供的仓库地址：

```bash
git clone https://github.com/ATEC-Challenge/ATEC2026_Simulation_Challenge.git
cd ATEC2026_Simulation_Challenge/source/atec_rl_lab
```

安装 ATEC 扩展：

```bash
pip install -e .
```

验证扩展安装成功：

```bash
python -c "import atec_rl_lab; print('ATEC扩展安装成功')"
```

### 步骤 6：环境验证

必须跑通才能开发。

进入赛事根目录：

```bash
cd ATEC2026_Simulation_Challenge
```

测试 L0 任务 A（徒步）环境，替换为你们选的机器人型号：

```bash
python demo/run_env.py --task A --robot B2wPiper
```

如果弹出仿真窗口，机器人能正常加载移动，说明环境部署成功。

## 开发规范

### 代码结构

只需要修改 `solution.py`。

```text
ATEC2026_Simulation_Challenge/
├── demo/
│   ├── solution.py       # 唯一需要修改的文件，实现你的算法
│   └── run_env.py        # 本地测试脚本，不需要改
└── source/
    └── atec_rl_lab/      # 官方扩展，禁止修改
```

### 必须实现的接口

```python
class AlgSolution:
    def __init__(self):
        # 初始化你的模型、参数、加载权重
        pass

    def predict(self, obs, current_score):
        """
        输入参数：
            obs: 观测数据
                - proprio: 本体状态（基座速度、关节位姿）
                - image: 多视角 RGB/深度图
                - extero: 激光雷达点云
            current_score: 当前得分
        返回格式：
            {"action": 动作数组, "giveup": 是否放弃当前 episode}
        """
        # 你的算法逻辑在这里
        return {"action": [...], "giveup": False}
```

### 控制参数规范

- 腿部/臂部关节：位置控制，输出值缩放因子 0.5
- 轮子：速度控制，输出值缩放因子 5.0

## 提交规则

| 限制项 | 要求 |
| --- | --- |
| 每日提交次数 | <= 10 次 |
| 每日成功提交次数 | <= 3 次 |
| 服务启动超时 | 300 秒内，模型加载不要太重量级 |
| 单轮运行最长时间 | 30 分钟 |

提交包结构：

```text
submission.zip/
├── solution.py           # 你的算法实现
├── requirements.txt      # 额外依赖（如果有）
└── models/               # 你的模型权重（可选）
```

## 行走组（任务 A）专项部署与优化建议

1. 机器人选型优先：B2wPiper（四轮足机械臂）越野性能最优，最适合徒步任务。
2. 重点调试方向：位置控制精度、平衡算法、地形适应策略。
3. 地形通关顺序：先过平坦，再过崎岖、斜坡、楼梯，逐步优化，不要贪多。
4. 得分技巧：到达终点额外加 4 分，尽量完成全程，比只过单个地形得分高。

## 常见问题排查

1. Isaac Sim 启动失败：检查 NVIDIA 驱动版本是否 >= 550，CUDA 版本是否匹配。
2. 仿真卡顿：降低仿真分辨率，关闭不必要的可视化选项。
3. 提交超时：`__init__` 函数不要加载过大模型，尽量轻量化。
4. 动作无响应：检查输出维度是否匹配机器人关节数，缩放因子是否正确。
