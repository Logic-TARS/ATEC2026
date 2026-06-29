"""Flat-terrain version of official Task D (B2W omni, pit-and-platform replaced by plane, box pushing).

Extends ``TaskDOmniEnvOfficialCfg`` with:
- Flat plane terrain (no pit, no platform)
- All other official Task D settings unchanged (box, 61D obs, rewards, terminations, episode length)
"""

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import EventTermCfg as EventTerm
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


def _short_walk_distance_reached(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    target_distance: float = 2.0,
):
    """Terminate once the robot reaches a target local x distance."""
    asset = env.scene[asset_cfg.name]
    robot_x = asset.data.root_pos_w[:, 0] - env.scene.env_origins[:, 0]
    return robot_x >= target_distance


def _short_walk_distance_reward(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    target_distance: float = 2.0,
):
    """One-step success reward when the short-walk target is reached."""
    return _short_walk_distance_reached(
        env,
        asset_cfg=asset_cfg,
        target_distance=target_distance,
    ).to(torch.float32)


def _short_walk_robot_x_progress(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    target_distance: float = 2.0,
):
    """Dense local x progress toward the short-walk target."""
    asset = env.scene[asset_cfg.name]
    robot_x = asset.data.root_pos_w[:, 0] - env.scene.env_origins[:, 0]
    return torch.clamp(robot_x / max(target_distance, 1e-6), min=0.0, max=1.0)


def _short_walk_timeout_penalty(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    target_distance: float = 2.0,
):
    """Penalize timing out before reaching the short-walk target."""
    timed_out = env.termination_manager.time_outs
    reached = _short_walk_distance_reached(
        env,
        asset_cfg=asset_cfg,
        target_distance=target_distance,
    )
    return (timed_out & ~reached).to(torch.float32)


def _taskf_box_local_x_progress(
    env,
    box_cfg: SceneEntityCfg = SceneEntityCfg("box"),
    start_x: float = -3.0,
    target_x: float = 0.7,
):
    """Dense local-frame box progress from reset x toward the target x."""
    box = env.scene[box_cfg.name]
    box_x = box.data.root_pos_w[:, 0] - env.scene.env_origins[:, 0]
    width = max(target_x - start_x, 1e-6)
    return torch.clamp((box_x - start_x) / width, min=0.0, max=1.0)


def _taskf_push_ready_reward(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    box_cfg: SceneEntityCfg = SceneEntityCfg("box"),
    desired_back_dist: float = 1.0,
    back_tolerance: float = 1.0,
    lateral_std: float = 0.75,
):
    """Reward being behind the box, laterally aligned, and facing +x to push."""
    asset = env.scene[asset_cfg.name]
    box = env.scene[box_cfg.name]

    robot_xy = asset.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]
    box_xy = box.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]

    behind_dist = box_xy[:, 0] - robot_xy[:, 0]
    behind_gate = torch.clamp(
        1.0 - torch.abs(behind_dist - desired_back_dist) / back_tolerance,
        min=0.0,
        max=1.0,
    )
    lateral_error = robot_xy[:, 1] - box_xy[:, 1]
    lateral_gate = torch.exp(-(lateral_error * lateral_error) / (lateral_std * lateral_std))

    forward_vec = torch.zeros(env.num_envs, 3, device=env.device)
    forward_vec[:, 0] = 1.0
    forward_w = math_utils.quat_apply(asset.data.root_quat_w, forward_vec)
    forward_x_gate = torch.clamp(forward_w[:, 0], min=0.0, max=1.0)

    return behind_gate * lateral_gate * forward_x_gate


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
        # Match Task A B2W's verified starting height while keeping Task F local x/y.
        self.scene.robot.init_state.pos = (0.0, 0.0, 0.78)

        # ------------------------------Actions------------------------------
        # Keep the 16D interface: 12 leg position actions + 4 wheel velocity actions.
        self.actions.joint_pos.scale = {".*_hip_joint": 0.125, "^(?!.*_hip_joint).*": 0.25}
        self.actions.wheel_vel.scale = 5.0

        # ------------------------------Commands------------------------------
        # Fixed push-box intent instead of random omni velocity commands.
        self.commands.base_velocity.resampling_time_range = (10.0, 10.0)
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.mode_probs = (1.0, 0.0, 0.0, 0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_x = (0.8, 0.8)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)

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
            weight=6.0,
            params={
                "box_cfg": SceneEntityCfg("box"),
                "std": 2.0,
            },
        )
        self.rewards.box_x_progress = RewTerm(
            func=_taskf_box_local_x_progress,
            weight=15.0,
            params={
                "box_cfg": SceneEntityCfg("box"),
                "start_x": -3.0,
                "target_x": 0.7,
            },
        )
        self.rewards.push_ready = RewTerm(
            func=_taskf_push_ready_reward,
            weight=4.0,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "box_cfg": SceneEntityCfg("box"),
                "desired_back_dist": 1.0,
                "back_tolerance": 1.0,
                "lateral_std": 0.75,
            },
        )
        self.rewards.alignment_with_box.weight = 3.0
        self.rewards.taskd_action_smoothness.weight = 0
        self.rewards.ang_vel_xy_l2.weight = -0.1
        self.rewards.flat_orientation_l2 = RewTerm(
            func=mdp.flat_orientation_l2,
            weight=-2.0,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        self.rewards.base_height_l2 = RewTerm(
            func=mdp.base_height_l2,
            weight=-1.0,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=[self.base_link_name]),
                "sensor_cfg": None,
                "target_height": 0.78,
            },
        )
        self.rewards.joint_pos_limits.weight = -5.0
        self.rewards.taskd_fall_penalty = RewTerm(
            func=mdp.taskd_fall_penalty,
            weight=-8.0,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "min_height": 0.45,
            },
        )
        self.rewards.stand_still.weight = 0
        self.rewards.joint_pos_penalty.weight = -1.0
        self.rewards.joint_mirror.weight = -0.05
        self.rewards.feet_height_body.weight = -2.0
        self.rewards.feet_height_body.params["target_height"] = -0.45
        self.rewards.undesired_contacts.weight = -1.0
        self.rewards.contact_forces.weight = -1.5e-4

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
        self.terminations.illegal_contact = DoneTerm(
            func=mdp.illegal_contact,
            params={
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces",
                    body_names=[self.base_link_name, ".*_hip", ".*_thigh"],
                ),
                "threshold": 1.0,
            },
            time_out=False,
        )

        # ------------------------------------------------------------------ #
        # Task F initialization — match Task A's deterministic reset
        # ------------------------------------------------------------------ #
        # Disable domain randomization to ensure robot starts from correct pose
        # (matching Task A's initialization strategy)
        self.observations.policy.enable_corruption = False
        self.observations.critic.enable_corruption = False

        self.events.randomize_rigid_body_material = None
        self.events.randomize_reset_joints = None
        self.events.randomize_push_robot = None

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


@configclass
class UnitreeB2WTaskFShortWalkEnvCfg(UnitreeB2WTaskFFlatEnvCfg):
    """Short-distance walking pretraining for Task F's 61D/16D interface."""

    target_distance: float = 2.0
    short_walk_reward: float = 50.0

    def __post_init__(self):
        super().__post_init__()

        # Stable B2W start height; keep deterministic root reset from Task F flat.
        self.scene.robot.init_state.pos = (0.0, 0.0, 0.58)
        self.episode_length_s = 10.0
        self.rewards.base_height_l2.params["target_height"] = 0.58

        # Restore default joint reset so every episode starts from the nominal stand.
        self.events.randomize_reset_joints = EventTerm(
            func=mdp.reset_joints_by_scale,
            mode="reset",
            params={
                "position_range": (1.0, 1.0),
                "velocity_range": (0.0, 0.0),
            },
        )

        # Short-walk command: straight ahead, no lateral or yaw command.
        self.commands.base_velocity.ranges.lin_vel_x = (0.65, 0.65)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)

        # Short-walk objective replaces box-pushing rewards.
        self.rewards.track_lin_vel_xy_exp.weight = 3.0
        self.rewards.track_ang_vel_z_exp.weight = 1.5
        self.rewards.upward.weight = 1.0
        self.rewards.box_x_progress.weight = 0
        self.rewards.stage_progress.weight = 0
        self.rewards.alignment_with_box.weight = 0
        self.rewards.distance_to_box.weight = 0
        self.rewards.push_ready.weight = 0
        self.rewards.robot_x_progress = RewTerm(
            func=_short_walk_robot_x_progress,
            weight=8.0,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "target_distance": self.target_distance,
            },
        )
        self.rewards.short_walk_success = RewTerm(
            func=_short_walk_distance_reward,
            weight=self.short_walk_reward,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "target_distance": self.target_distance,
            },
        )
        self.rewards.short_walk_timeout = RewTerm(
            func=_short_walk_timeout_penalty,
            weight=-10.0,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "target_distance": self.target_distance,
            },
        )

        # Slightly stronger symmetry regularization to discourage three-leg gait.
        self.rewards.joint_mirror.weight = -0.1

        # Reset as soon as the short walk is complete; do not require box success.
        self.terminations.taskd_success = None
        self.terminations.short_walk_success = DoneTerm(
            func=_short_walk_distance_reached,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "target_distance": self.target_distance,
            },
            time_out=False,
        )


@configclass
class UnitreeB2WTaskFShortOmniEnvCfg(UnitreeB2WTaskFShortWalkEnvCfg):
    """Small-range omni walking pretraining from the stable short-walk policy."""

    def __post_init__(self):
        super().__post_init__()

        # Phase-1 omni warmup: keep commands close to the known stable forward gait.
        self.commands.base_velocity.mode_probs = (0.55, 0.10, 0.25, 0.05, 0.05)
        self.commands.base_velocity.ranges.lin_vel_x = (-0.15, 0.65)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.10, 0.10)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.25, 0.25)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)

        # Do not let the short-walk x-distance objective dominate omni learning.
        self.rewards.robot_x_progress.weight = 0
        self.rewards.short_walk_success.weight = 0
        self.rewards.short_walk_timeout.weight = 0


@configclass
class UnitreeB2WTaskFShortOmniFastEnvCfg(UnitreeB2WTaskFShortOmniEnvCfg):
    """Forward-biased fast omni fine-tuning from the stable short-omni policy."""

    def __post_init__(self):
        super().__post_init__()

        # Keep omni commands conservative while biasing sampling toward faster forward motion.
        self.commands.base_velocity.mode_probs = (0.65, 0.05, 0.20, 0.05, 0.05)
        self.commands.base_velocity.ranges.lin_vel_x = (-0.10, 0.80)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.08, 0.08)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.20, 0.20)


@configclass
class UnitreeB2WTaskFShortOmniRobustEnvCfg(UnitreeB2WTaskFShortOmniEnvCfg):
    """Forward-biased short-omni fine-tuning with stronger contact stability."""

    def __post_init__(self):
        super().__post_init__()

        # Keep the speed increase below the v3 fast branch while favoring forward samples.
        self.commands.base_velocity.mode_probs = (0.65, 0.05, 0.20, 0.05, 0.05)
        self.commands.base_velocity.ranges.lin_vel_x = (-0.10, 0.75)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.08, 0.08)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.20, 0.20)

        # Bias the policy away from faster-but-messy contacts without increasing speed reward.
        self.rewards.undesired_contacts.weight = -1.5
        self.rewards.contact_forces.weight = -2.5e-4
        self.rewards.feet_height_body.weight = -3.0
        self.rewards.joint_pos_penalty.weight = -1.25
        self.rewards.action_rate_l2.weight = -0.015
