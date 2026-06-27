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

import atec_rl_lab.train.locomotion.velocity.mdp as mdp

from .taskd_omni_env_cfg import TaskDOmniEnvOfficialCfg


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
        self.rewards.stage_progress = RewTerm(
            func=mdp.stage_progress_gated,
            weight=1.5,
            params={"box_cfg": SceneEntityCfg("box")},
        )
        self.rewards.distance_to_box = RewTerm(
            func=mdp.distance_to_box_exp,
            weight=3.0,
            params={
                "box_cfg": SceneEntityCfg("box"),
                "std": 1.5,
            },
        )
        self.rewards.alignment_with_box.weight = 2.0
        self.rewards.taskd_action_smoothness.weight = 0
        self.rewards.joint_pos_limits.weight = -2.0

        # ------------------------------------------------------------------ #
        # Task F success termination — end episode when task is achieved
        # ------------------------------------------------------------------ #
        self.terminations.taskd_success = DoneTerm(
            func=mdp.taskd_success,
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
