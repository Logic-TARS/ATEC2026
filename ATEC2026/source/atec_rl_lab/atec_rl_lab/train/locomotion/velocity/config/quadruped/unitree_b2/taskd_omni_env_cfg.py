"""Task D fine-tuning env configs for B2W omni (pit-and-platform terrain, box pushing).

Extends ``UnitreeB2WPiperRoughOmniEnvCfg`` with:
- Pit-and-platform terrain (parameterized width/height)
- Pushable box in the scene
- 8 extra observation terms (61D total)
- Task D dense rewards (replacing velocity-tracking)
"""

from copy import deepcopy

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

import atec_rl_lab.train.locomotion.velocity.mdp as mdp
import atec_rl_lab.tasks.task_d.mdp as taskd_mdp
from atec_rl_lab.tasks.task_d.terrain import (
    PitAndPlatformTerrainCfg,
    TASK_D_TERRAIN_CFG,
)

from .b2w_omni_env_cfg import UnitreeB2WPiperRoughOmniEnvCfg


@configclass
class TaskDOmniEnvCfg(UnitreeB2WPiperRoughOmniEnvCfg):
    """Base Task D fine-tuning env config — pit-and-platform terrain, box in scene, 61D observations.

    Subclasses override ``pit_width_range``, ``platform_height_range``, and ``box_mass``
    for Easy / Medium / Official difficulty levels.
    """

    pit_width_range: tuple[float, float] = (1.3, 1.4)
    platform_height_range: tuple[float, float] = (1.0, 1.2)
    box_mass: float = 8.0
    task_episode_length_s: float = 180.0

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = self.task_episode_length_s

        # ------------------------------------------------------------------ #
        # Terrain: rough → pit-and-platform
        # ------------------------------------------------------------------ #
        terrain_cfg = deepcopy(TASK_D_TERRAIN_CFG)
        pit_cfg = terrain_cfg.terrain_generator.sub_terrains.get("pit_and_platform")
        if isinstance(pit_cfg, PitAndPlatformTerrainCfg):
            pit_cfg.pit_width_range = self.pit_width_range
            pit_cfg.platform_height_range = self.platform_height_range
        self.scene.terrain = terrain_cfg

        # ------------------------------------------------------------------ #
        # Pushable box
        # ------------------------------------------------------------------ #
        self.scene.box = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Box",
            spawn=sim_utils.CuboidCfg(
                size=(0.8, 1.0, 0.6),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                mass_props=sim_utils.MassPropertiesCfg(mass=self.box_mass),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=0.9,
                    dynamic_friction=0.8,
                    restitution=0.0,
                ),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(-3, 1.6, 0.5)),
        )

        self.events.taskd_reset_buffers = EventTerm(
            func=mdp.taskd_reset_buffers,
            mode="reset",
        )

        # ------------------------------------------------------------------ #
        # Observations: 53 inherited + 8 task-specific = 61D
        # ------------------------------------------------------------------ #
        self.observations.policy.taskd_score_norm = ObsTerm(
            func=mdp.taskd_score_norm,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "box_cfg": SceneEntityCfg("box"),
                "max_score": 36.0,
            },
        )
        self.observations.policy.taskd_stage_onehot = ObsTerm(
            func=mdp.taskd_stage_onehot,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "box_cfg": SceneEntityCfg("box"),
                "thresholds": [1.9, 15.0, 21.0],
            },
        )
        self.observations.policy.taskd_box_detected = ObsTerm(
            func=mdp.taskd_box_detected,
            params={
                "sensor_cfg": SceneEntityCfg("height_scanner"),
                "box_cfg": SceneEntityCfg("box"),
            },
        )
        self.observations.policy.taskd_box_bearing = ObsTerm(
            func=mdp.taskd_box_bearing,
            params={
                "sensor_cfg": SceneEntityCfg("height_scanner"),
                "box_cfg": SceneEntityCfg("box"),
            },
        )
        self.observations.policy.taskd_box_distance_norm = ObsTerm(
            func=mdp.taskd_box_distance_norm,
            params={
                "sensor_cfg": SceneEntityCfg("height_scanner"),
                "box_cfg": SceneEntityCfg("box"),
                "max_dist": 5.0,
            },
        )

        # ------------------------------------------------------------------ #
        # Rewards: replace velocity-tracking with Task D dense rewards
        # ------------------------------------------------------------------ #
        # Disable parent velocity-tracking (handled by stage/alignment rewards)
        self.rewards.track_lin_vel_xy_exp.weight = 0
        self.rewards.track_ang_vel_z_exp.weight = 0

        # Task D fine-tuning dense rewards
        self.rewards.official_cross_x = RewTerm(
            func=taskd_mdp.RewardCrossX,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "threshold": [-1.4, 2.0],
                "reward_value": [2.0, 20.0],
                "debug": False,
                "visual_assets": False,
            },
            weight=1.0,
        )
        self.rewards.official_box_in_target_x = RewTerm(
            func=taskd_mdp.RewardBoxXInRange,
            params={
                "asset_cfg": SceneEntityCfg("box"),
                "x_min": [-0.7, -1.4],
                "x_max": [0.7, -0.7],
                "reward_value": 14.0,
                "one_time": True,
                "debug": False,
            },
            weight=1.0,
        )
        self.rewards.box_x_progress = RewTerm(
            func=mdp.box_x_progress, weight=2.0
        )
        self.rewards.stage_progress = RewTerm(
            func=mdp.stage_progress, weight=1.5
        )
        self.rewards.alignment_with_box = RewTerm(
            func=mdp.alignment_with_box, weight=0.5
        )
        self.rewards.taskd_fall_penalty = RewTerm(
            func=mdp.taskd_fall_penalty, weight=-5.0
        )
        self.rewards.taskd_action_smoothness = RewTerm(
            func=mdp.taskd_action_smoothness, weight=-0.01
        )

        self.terminations.terrain_out_of_bounds = None
        self.terminations.x_reached = DoneTerm(
            func=taskd_mdp.robot_x_greater_than,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "x_threshold": 3.5,
            },
            time_out=False,
        )

        # ------------------------------------------------------------------ #
        # Curriculum: not needed for fine-tuning
        # ------------------------------------------------------------------ #
        self.curriculum.terrain_levels = None
        self.curriculum.command_levels_lin_vel = None
        self.curriculum.command_levels_ang_vel = None



@configclass
class TaskDOmniEnvEasyCfg(TaskDOmniEnvCfg):
    """Easy difficulty — narrow pit, low platform, light box."""
    pit_width_range: tuple[float, float] = (0.8, 0.9)
    platform_height_range: tuple[float, float] = (0.5, 0.6)
    box_mass: float = 5.0
    task_episode_length_s: float = 60.0


@configclass
class TaskDOmniEnvMediumCfg(TaskDOmniEnvCfg):
    """Medium difficulty — moderate pit, mid platform, medium box."""
    pit_width_range: tuple[float, float] = (1.0, 1.1)
    platform_height_range: tuple[float, float] = (0.7, 0.8)
    box_mass: float = 6.5
    task_episode_length_s: float = 90.0


@configclass
class TaskDOmniEnvOfficialCfg(TaskDOmniEnvCfg):
    """Official (hard) difficulty — wide pit, tall platform, heavy box."""
    pit_width_range: tuple[float, float] = (1.3, 1.4)
    platform_height_range: tuple[float, float] = (1.0, 1.2)
    box_mass: float = 8.0
    task_episode_length_s: float = 180.0
