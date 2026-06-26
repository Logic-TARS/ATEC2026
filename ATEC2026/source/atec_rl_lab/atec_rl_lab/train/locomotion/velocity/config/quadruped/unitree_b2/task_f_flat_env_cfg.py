import isaaclab.sim as sim_utils
from isaaclab.managers import ObservationTermCfg as ObsTerm, RewardTermCfg as RewTerm, TerminationTermCfg as DoneTerm, SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
import atec_rl_lab.train.locomotion.velocity.mdp as mdp

from .flat_b2w_omni_env_cfg import UnitreeB2WPiperFlatOmniEnvCfg


@configclass
class UnitreeB2WTaskFFlatEnvCfg(UnitreeB2WPiperFlatOmniEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # ------------------------------Scene------------------------------
        # Give the flat plane a light neutral color instead of the default black.
        self.scene.terrain.visual_material = sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.35, 0.42, 0.34),
            roughness=0.8,
        )

        # ------------------------------Observations------------------------------
        # Recreate base_lin_vel (parent set it to None)
        self.observations.policy.base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel,
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        # Set all policy observation scales to 1.0
        self.observations.policy.base_ang_vel.scale = 1.0
        self.observations.policy.joint_pos.scale = 1.0
        self.observations.policy.joint_vel.scale = 1.0

        # ------------------------------Rewards------------------------------
        # Recreate flat_orientation_l2 (parent zeroed it via disable_zero_weight_rewards)
        self.rewards.flat_orientation_l2 = RewTerm(
            func=mdp.flat_orientation_l2, weight=-0.5
        )

        # Adjust weights
        self.rewards.stand_still.weight = -1.0
        self.rewards.wheel_vel_penalty.weight = -1.0e-4
        self.rewards.feet_height_body.weight = -2.0
        self.rewards.feet_height_body.params["target_height"] = -0.45

        # Restrict asset_cfg to leg joints only
        self.rewards.joint_pos_limits.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.stand_still.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.joint_pos_penalty.params["asset_cfg"].joint_names = self.leg_joint_names

        # ------------------------------Terminations------------------------------
        # Recreate illegal_contact (parent set it to None)
        self.terminations.illegal_contact = DoneTerm(
            func=mdp.illegal_contact,
            params={
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces",
                    body_names=[self.base_link_name, ".*_hip", ".*_thigh"],
                ),
                "threshold": 1.0,
            },
        )
