# Reference: https://github.com/fan-ziqi/robot_lab

from isaaclab.utils import configclass

from .b2w_omni_env_cfg import UnitreeB2WPiperRoughOmniEnvCfg


@configclass
class UnitreeB2WPiperFlatOmniEnvCfg(UnitreeB2WPiperRoughOmniEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Match Task A / Task D B2W root init height
        self.scene.robot = self.scene.robot.replace(
            init_state=self.scene.robot.init_state.replace(pos=(0.0, 0.0, 0.78))
        )

        # Upright spawn on flat ground: zero roll/pitch, keep yaw randomized
        self.events.randomize_reset_base.params = {
            "pose_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (-3.14, 3.14),
            },
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (-0.5, 0.5),
            },
        }

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
