# Draft: Task L0 (Task A 平坦地形) 达到 26 分

## 需求确认
- 任务: Task A (越野导航) - 平坦地形
- 目标得分: 26 分 (最大值)
- 机器人: Unitree B2W + AgileX Piper

## 得分结构
- x_thresholds: [-115.0, -35.0, 45.0, 125.0, 140.0]
- rewards: [2.0, 4.0, 8.0, 8.0, 4.0]
- 总计: 2 + 4 + 8 + 8 + 4 = 26 分

## 当前状态
- 起始位置: x = -141
- 目标位置: x = 145 (终点)
- 当前策略: demo/policy.pt (baseline flat policy)

## 技术决策
- [待确认]: 使用哪个机器人平台
- [待确认]: 是否需要重新训练策略
- [待确认]: 是否需要调整 solution.py

## 研究发现
- 地形配置: 15 行，包含 flat, random_rough, slopes, stairs
- max_init_terrain_level=0 表示从最简单地形开始
- 现有 baseline policy: atec_robot_model/baseline/unitree_b2_flat/policy.pt

## 待解决问题
- [ ] 当前 baseline policy 能得多少分？
- [ ] 是否需要课程学习训练？
- [ ] 是否需要调整 solution.py 中的参数？
