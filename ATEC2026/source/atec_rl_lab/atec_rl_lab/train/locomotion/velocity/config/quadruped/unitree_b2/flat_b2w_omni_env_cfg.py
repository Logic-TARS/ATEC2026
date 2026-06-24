# Reference: https://github.com/fan-ziqi/robot_lab

from isaaclab.utils import configclass

from .b2w_omni_env_cfg import UnitreeB2WPiperRoughOmniEnvCfg


@configclass
class UnitreeB2WPiperFlatOmniEnvCfg(UnitreeB2WPiperRoughOmniEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        self.scene.height_scanner = None
        if hasattr(self.observations.policy, "height_scan"):
            self.observations.policy.height_scan = None
        if hasattr(self.observations.critic, "height_scan"):
            self.observations.critic.height_scan = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None
