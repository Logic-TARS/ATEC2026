"""Flat-terrain version of official Task D (B2W omni, pit-and-platform replaced by plane, box pushing).

Extends ``TaskDOmniEnvOfficialCfg`` with:
- Flat plane terrain (no pit, no platform)
- All other official Task D settings unchanged (box, 61D obs, rewards, terminations, episode length)
"""

from isaaclab.utils import configclass

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

        # ------------------------------Curriculum------------------------------
        # No terrain levels curriculum on flat terrain
        self.curriculum.terrain_levels = None
