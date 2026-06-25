"""Custom observation term functions for Task D (B2W omni fine-tuning).

Each function matches ``ObsTerm.func`` signature ``(env, **kwargs) -> torch.Tensor``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _ensure_taskd_score_buffers(env: ManagerBasedRLEnv, num_envs: int, device: torch.device):
    milestones = getattr(env, "_taskd_milestones", None)
    score = getattr(env, "_taskd_score", None)
    if milestones is None or milestones.shape != (num_envs, 3) or milestones.device != device:
        env._taskd_milestones = torch.zeros(num_envs, 3, device=device, dtype=torch.bool)
    if score is None or score.shape != (num_envs,) or score.device != device:
        env._taskd_score = torch.zeros(num_envs, device=device, dtype=torch.float32)


def taskd_reset_buffers(env: ManagerBasedRLEnv, env_ids: torch.Tensor | None = None):
    """Reset Task D fine-tuning buffers for newly reset environments."""
    device = env.device
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=device)
    else:
        env_ids = env_ids.to(device=device)

    _ensure_taskd_score_buffers(env, env.num_envs, device)
    env._taskd_milestones[env_ids] = False
    env._taskd_score[env_ids] = 0.0

    if hasattr(env, "_taskd_prev_action"):
        env._taskd_prev_action[env_ids] = 0.0

    try:
        box: RigidObject = env.scene["box"]
    except KeyError:
        box = None
    if box is not None:
        box_x = box.data.root_pos_w[:, 0]
        if not hasattr(env, "_taskd_prev_box_x") or env._taskd_prev_box_x.shape != (env.num_envs,):
            env._taskd_prev_box_x = box_x.clone().detach()
        else:
            env._taskd_prev_box_x[env_ids] = box_x[env_ids].clone().detach()


def _compute_pseudo_score(env: ManagerBasedRLEnv, robot_x: torch.Tensor, box_x: torch.Tensor) -> torch.Tensor:
    """Update and return cumulative pseudo-score from robot/box x milestones."""
    _ensure_taskd_score_buffers(env, robot_x.shape[0], robot_x.device)
    milestones = env._taskd_milestones
    score = env._taskd_score

    crossed_robot_entry = robot_x > -1.4
    box_in_range = (box_x >= -1.4) & (box_x <= 0.7)
    crossed_finish = robot_x > 2.0

    new_entry = crossed_robot_entry & (~milestones[:, 0])
    new_box = box_in_range & (~milestones[:, 1])
    new_finish = crossed_finish & (~milestones[:, 2])

    score[new_entry] += 2.0
    score[new_box] += 14.0
    score[new_finish] += 20.0

    milestones[new_entry, 0] = True
    milestones[new_box, 1] = True
    milestones[new_finish, 2] = True

    return score


def taskd_score_norm(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    box_cfg: SceneEntityCfg = SceneEntityCfg("box"),
    max_score: float = 36.0,
) -> torch.Tensor:
    """Normalised pseudo-score indicating Task D progress.

    Computes a milestone-based score from robot and box x-positions (no external
    ``current_score`` signal required) and normalises it to ``[0, 1]``.

    Args:
        asset_cfg: Scene entity config for the robot.
        box_cfg: Scene entity config for the box.
        max_score: Maximum possible score used for normalisation.

    Returns:
        Tensor of shape ``[num_envs, 1]`` with values in ``[0, 1]``.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    box: RigidObject = env.scene[box_cfg.name]

    robot_x = asset.data.root_pos_w[:, 0]
    box_x = box.data.root_pos_w[:, 0]

    score = _compute_pseudo_score(env, robot_x, box_x)
    return (score / max_score).clamp(0, 1).unsqueeze(-1)


def taskd_stage_onehot(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    box_cfg: SceneEntityCfg = SceneEntityCfg("box"),
    thresholds: list[float] | tuple[float, ...] = (1.9, 15.0, 21.0),
) -> torch.Tensor:
    """One-hot encoding of the current task stage (0–3).

    Stages are determined by pseudo-score thresholds:

    - Stage 0: score < thresholds[0]  (no milestone)
    - Stage 1: score < thresholds[1]  (robot past -1.4)
    - Stage 2: score < thresholds[2]  (box in delivery zone)
    - Stage 3: score >= thresholds[2] (robot past 2.0)

    Args:
        asset_cfg: Scene entity config for the robot.
        box_cfg: Scene entity config for the box.
        thresholds: Score boundaries separating the four stages.

    Returns:
        Tensor of shape ``[num_envs, 4]`` (one-hot).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    box: RigidObject = env.scene[box_cfg.name]

    robot_x = asset.data.root_pos_w[:, 0]
    box_x = box.data.root_pos_w[:, 0]

    score = _compute_pseudo_score(env, robot_x, box_x)

    # Determine stage index
    stage = torch.zeros(score.shape[0], device=score.device, dtype=torch.long)
    for i, thresh in enumerate(thresholds):
        stage[score >= thresh] = i + 1

    return torch.nn.functional.one_hot(stage, num_classes=4).float()


def taskd_box_detected(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("scan_sensor"),
    box_cfg: SceneEntityCfg = SceneEntityCfg("box"),
) -> torch.Tensor:
    """Whether the box is detected within 3.0 m of the robot.

    Uses a distance proxy (XY-plane Euclidean distance < 3.0 m) as a simple
    stand-in for a full ray-caster check.

    Args:
        sensor_cfg: Scene entity config for the ray-caster sensor (reserved for
            future use; not required by the distance proxy).
        box_cfg: Scene entity config for the box.

    Returns:
        Tensor of shape ``[num_envs, 1]`` with 1.0 if detected, else 0.0.
    """
    asset: Articulation = env.scene["robot"]
    box: RigidObject = env.scene[box_cfg.name]

    robot_xy = asset.data.root_pos_w[:, :2]
    box_xy = box.data.root_pos_w[:, :2]

    dist = torch.norm(box_xy - robot_xy, dim=-1)
    return (dist < 3.0).float().unsqueeze(-1)


def taskd_box_bearing(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("scan_sensor"),
    box_cfg: SceneEntityCfg = SceneEntityCfg("box"),
) -> torch.Tensor:
    """Normalised bearing to the box in ``[-1, 1]``.

    Computes the signed angle from the robot's forward direction (+x) to the
    vector pointing toward the box in the XY plane, normalised by π.

    0 = straight ahead, positive = left, negative = right.

    Args:
        sensor_cfg: Scene entity config for the ray-caster sensor (reserved for
            future use).
        box_cfg: Scene entity config for the box.

    Returns:
        Tensor of shape ``[num_envs, 1]``.
    """
    asset: Articulation = env.scene["robot"]
    box: RigidObject = env.scene[box_cfg.name]

    robot_pos = asset.data.root_pos_w[:, :3]
    box_pos = box.data.root_pos_w[:, :3]
    robot_quat = asset.data.root_quat_w

    # Vector from robot to box in world frame
    to_box = box_pos - robot_pos

    # Rotate into robot's local frame
    local_vec = quat_apply_inverse(robot_quat, to_box)

    dist = torch.norm(to_box[:, :2], dim=-1)
    detected = dist < 3.0
    # Match solution.py suffix semantics: positive values steer right-sector detections back toward center.
    bearing = -torch.atan2(local_vec[:, 1], local_vec[:, 0]) / torch.pi
    bearing = torch.where(detected, bearing.clamp(-1.0, 1.0), torch.zeros_like(bearing))
    return bearing.unsqueeze(-1)


def taskd_box_distance_norm(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("scan_sensor"),
    box_cfg: SceneEntityCfg = SceneEntityCfg("box"),
    max_dist: float = 5.0,
) -> torch.Tensor:
    """Normalised XY-plane distance to the box, clamped to ``[0, 1]``.

    Args:
        sensor_cfg: Scene entity config for the ray-caster sensor (reserved for
            future use).
        box_cfg: Scene entity config for the box.
        max_dist: Maximum expected distance used for normalisation.

    Returns:
        Tensor of shape ``[num_envs, 1]``.
    """
    asset: Articulation = env.scene["robot"]
    box: RigidObject = env.scene[box_cfg.name]

    robot_xy = asset.data.root_pos_w[:, :2]
    box_xy = box.data.root_pos_w[:, :2]

    dist = torch.norm(box_xy - robot_xy, dim=-1)
    detected = dist < 3.0
    distance_norm = (dist / max_dist).clamp(0, 1)
    distance_norm = torch.where(detected, distance_norm, torch.zeros_like(distance_norm))
    return distance_norm.unsqueeze(-1)
