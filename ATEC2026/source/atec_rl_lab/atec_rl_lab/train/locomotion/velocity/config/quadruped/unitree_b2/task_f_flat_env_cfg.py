"""Flat-terrain version of official Task D (B2W omni, pit-and-platform replaced by plane, box pushing).

Extends ``TaskDOmniEnvOfficialCfg`` with:
- Flat plane terrain (no pit, no platform)
- All other official Task D settings unchanged (box, 61D obs, rewards, terminations, episode length)
"""

import isaaclab.sim as sim_utils
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
import torch

import atec_rl_lab.train.locomotion.velocity.mdp as mdp

from .taskd_omni_env_cfg import TaskDOmniEnvOfficialCfg


def _taskf_stage_progress_gated(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    box_cfg: SceneEntityCfg = SceneEntityCfg("box"),
    thresholds: list[tuple[float, float]] | None = None,
    gate_dist: float = 4.0,
):
    """Stage progress using per-env local x coordinates, gated by box proximity."""
    if thresholds is None:
        thresholds = [(-1.4, 2.0), (2.0, 3.5)]

    asset = env.scene[asset_cfg.name]
    box = env.scene[box_cfg.name]
    robot_x = asset.data.root_pos_w[:, 0] - env.scene.env_origins[:, 0]

    reward = torch.zeros(env.num_envs, device=env.device)
    for low, high in thresholds:
        width = high - low
        if width <= 0.0:
            continue
        progress = (robot_x - low) / width
        reward += torch.clamp(progress, min=0.0, max=1.0)

    dist = torch.norm(asset.data.root_pos_w[:, :2] - box.data.root_pos_w[:, :2], dim=1)
    gate = torch.clamp(1.0 - dist / gate_dist, min=0.0, max=1.0)
    return reward * gate


def _taskf_success_termination(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    box_cfg: SceneEntityCfg = SceneEntityCfg("box"),
    robot_x_threshold: float = 2.0,
    box_x_threshold: float = 0.7,
):
    """Return Task F success using per-env local x coordinates."""
    asset = env.scene[asset_cfg.name]
    box = env.scene[box_cfg.name]
    robot_x = asset.data.root_pos_w[:, 0] - env.scene.env_origins[:, 0]
    box_x = box.data.root_pos_w[:, 0] - env.scene.env_origins[:, 0]
    return (robot_x > robot_x_threshold) & (box_x > box_x_threshold)


@configclass
class UnitreeB2WTaskFFlatEnvCfg(TaskDOmniEnvOfficialCfg):
    """Flat-terrain version of official Task D — plane replaces pit-and-platform terrain."""

    def __post_init__(self):
        super().__post_init__()

        # ------------------------------Terrain------------------------------
        # Replace pit-and-platform terrain with a flat plane
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # BetterTerrainImporter's plane path expects a material with diffuse_color.
        self.scene.terrain.visual_material = sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.35, 0.42, 0.34),
            roughness=0.8,
        )

        # ------------------------------------------------------------------ #
        # Task F reward weight overrides
        # ------------------------------------------------------------------ #
        # The inherited official terms use world-frame x thresholds. In a cloned
        # multi-env scene, those thresholds depend on env grid placement.
        self.rewards.official_cross_x.weight = 0
        self.rewards.official_box_in_target_x.weight = 0
        self.rewards.stage_progress = RewTerm(
            func=_taskf_stage_progress_gated,
            weight=3.0,
            params={
                "box_cfg": SceneEntityCfg("box"),
                "gate_dist": 4.0,
            },
        )
        self.rewards.distance_to_box = RewTerm(
            func=mdp.distance_to_box_exp,
            weight=10.0,
            params={
                "box_cfg": SceneEntityCfg("box"),
                "std": 2.0,
            },
        )
        self.rewards.alignment_with_box.weight = 3.0
        self.rewards.taskd_action_smoothness.weight = 0
        self.rewards.joint_pos_limits.weight = -0.5
        self.rewards.taskd_fall_penalty.weight = -2.0

        # ------------------------------------------------------------------ #
        # Task F success termination — end episode when task is achieved
        # ------------------------------------------------------------------ #
        self.terminations.taskd_success = DoneTerm(
            func=_taskf_success_termination,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "box_cfg": SceneEntityCfg("box"),
                "robot_x_threshold": 2.0,
                "box_x_threshold": 0.7,
            },
            time_out=False,
        )

        # ------------------------------Curriculum------------------------------
        # No terrain levels curriculum on flat terrain
        self.curriculum.terrain_levels = None

        # ------------------------------Terminations------------------------------
        # Flat pretraining needs long rollouts; keep success termination for official Task D only.
        self.terminations.x_reached = None

        # ------------------------------------------------------------------ #
        # Task F initialization — match Task A's deterministic reset
        # ------------------------------------------------------------------ #
        # Disable domain randomization to ensure robot starts from correct pose
        # (matching Task A's initialization strategy)
        self.observations.proprio.enable_corruption = False
        self.observations.extero.enable_corruption = False
        self.observations.image.enable_corruption = False

        self.events.physics_material = None
        self.events.base_external_force_torque = None
        self.events.reset_robot_joints = None

        # Disable pose randomization — robot always starts upright, facing +x
        self.events.randomize_reset_base.params = {
            "pose_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }

        # Disable other domain randomization
        self.events.randomize_rigid_body_mass_base = None
        self.events.randomize_rigid_body_mass_others = None
        self.events.randomize_com_positions = None
        self.events.randomize_apply_external_force_torque = None
        self.events.randomize_actuator_gains = None
