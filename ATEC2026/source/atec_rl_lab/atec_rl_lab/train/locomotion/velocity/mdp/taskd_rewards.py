# Copyright (c) 2026, ATEC 2026 Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom reward terms for Task D (box-pushing) fine-tuning.

Each function matches the ``RewardTerm.func`` signature
``(env, **kwargs) -> torch.Tensor`` and returns a 1-D tensor of shape
``[num_envs]``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def box_x_progress(
    env: ManagerBasedRLEnv,
    box_cfg: SceneEntityCfg = SceneEntityCfg("box"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward proportional to the box's forward x-velocity.

    Uses a persistent buffer ``env._taskd_prev_box_x`` (shape ``[num_envs]``)
    initialised to the box's current world-x position on the first call.  On each
    subsequent call it computes ``delta_x = box_x - prev_box_x``, clamps to
    ``[0, 0.5]`` (only forward movement is rewarded), updates the buffer, and
    returns ``delta_x * 10.0``.

    Args:
        env: The RL environment instance.
        box_cfg: Scene entity for the pushable box (a ``RigidObject``).
        asset_cfg: Scene entity for the robot (unused, kept for signature
            consistency with other reward terms).

    Returns:
        Reward per environment, shape ``[num_envs]``.
    """
    box: RigidObject = env.scene[box_cfg.name]
    box_x = box.data.root_pos_w[:, 0]

    # Initialise persistent buffer on first call.
    if not hasattr(env, "_taskd_prev_box_x"):
        env._taskd_prev_box_x = box_x.clone().detach()

    # Forward progress (positive delta_x).
    delta_x = box_x - env._taskd_prev_box_x
    delta_x = torch.clamp(delta_x, min=0.0, max=0.5)

    # Update buffer.
    env._taskd_prev_box_x = box_x.clone().detach()

    return delta_x * 10.0


def stage_progress(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    thresholds: Sequence[tuple[float, float]] | None = None,
) -> torch.Tensor:
    """Dense progress reward for traversing x-interval stages.

    For each ``(low, high)`` pair in *thresholds* the contribution is the fraction
    of the interval the robot has traversed, clamped to ``[0, 1]``.  The total
    reward is the sum across all intervals, so the maximum possible value equals
    the number of intervals.

    Args:
        env: The RL environment instance.
        asset_cfg: Scene entity for the robot.
        thresholds: List of ``(low, high)`` x-range pairs.  Defaults to
            ``[(-1.4, 2.0), (2.0, 3.5)]``.

    Returns:
        Reward per environment, shape ``[num_envs]``.
    """
    if thresholds is None:
        thresholds = [(-1.4, 2.0), (2.0, 3.5)]

    asset: Articulation = env.scene[asset_cfg.name]
    robot_x = asset.data.root_pos_w[:, 0]

    reward = torch.zeros(env.num_envs, device=env.device)
    for low, high in thresholds:
        width = high - low
        if width <= 0.0:
            continue
        progress = (robot_x - low) / width
        reward += torch.clamp(progress, min=0.0, max=1.0)

    return reward


def alignment_with_box(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    box_cfg: SceneEntityCfg = SceneEntityCfg("box"),
) -> torch.Tensor:
    """Cosine of the angle between the robot's forward direction and the box.

    Values in ``[-1, 1]``: ``+1`` when the robot faces directly toward the box,
    ``-1`` when it faces directly away.

    Args:
        env: The RL environment instance.
        asset_cfg: Scene entity for the robot.
        box_cfg: Scene entity for the pushable box.

    Returns:
        Reward per environment, shape ``[num_envs]``.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    box: RigidObject = env.scene[box_cfg.name]

    # Robot and box positions (x, y).
    robot_pos = asset.data.root_pos_w[:, :2]
    box_pos = box.data.root_pos_w[:, :2]

    # Unit vector from robot to box.
    dir_to_box = box_pos - robot_pos
    dir_norm = torch.norm(dir_to_box, dim=1)
    dir_normalized = dir_to_box / (dir_norm.unsqueeze(-1) + 1e-8)

    # Robot forward direction in world frame: rotate (1, 0, 0) by root quat.
    forward_vec = torch.zeros(env.num_envs, 3, device=env.device)
    forward_vec[:, 0] = 1.0
    forward_w = math_utils.quat_apply(asset.data.root_quat_w, forward_vec)

    # Dot product of forward (x, y) with the unit direction (x, y).
    dot = forward_w[:, 0] * dir_normalized[:, 0] + forward_w[:, 1] * dir_normalized[:, 1]
    return dot


def taskd_fall_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    min_height: float = 0.25,
) -> torch.Tensor:
    """Positive fall cost when the robot root height drops below *min_height*.

    Pair this with a negative reward weight.

    Args:
        env: The RL environment instance.
        asset_cfg: Scene entity for the robot.
        min_height: Minimum root height before the penalty applies.

    Returns:
        Cost per environment, shape ``[num_envs]``.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    root_height = asset.data.root_pos_w[:, 2]
    return (root_height < min_height).float()


def taskd_action_smoothness(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """Squared L2 difference between current and previous actions.

    Uses a persistent buffer ``env._taskd_prev_action`` (shape ``[num_envs,
    action_dim]``).  On the first call the buffer is initialised from the current
    action so that the penalty is zero for the first step.

    Args:
        env: The RL environment instance.

    Returns:
        Cost per environment, shape ``[num_envs]``. Pair this with a negative reward weight.
    """
    action = env.action_manager.action

    # Initialise persistent buffer on the first call.
    if not hasattr(env, "_taskd_prev_action"):
        env._taskd_prev_action = action.clone().detach()

    # Squared L2 difference.
    diff = action - env._taskd_prev_action
    cost = torch.sum(diff * diff, dim=1)

    # Update buffer.
    env._taskd_prev_action = action.clone().detach()

    return cost
