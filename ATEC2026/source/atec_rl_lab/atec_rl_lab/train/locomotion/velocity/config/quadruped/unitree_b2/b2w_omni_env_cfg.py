from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from atec_rl_lab.train.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg
from atec_rl_lab.assets.robots import UNITREE_B2W_PIPER_CFG
import atec_rl_lab.train.locomotion.velocity.mdp as mdp


@configclass
class UnitreeB2WPiperRoughOmniEnvCfg(LocomotionVelocityRoughEnvCfg):
    base_link_name = "base_link"
    foot_link_name = ".*_foot"
    # fmt: off
    leg_joint_names = [
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    ]
    wheel_joint_names = [
        "FR_foot_joint", "FL_foot_joint",
        "RR_foot_joint", "RL_foot_joint",
    ]
    # fmt: on

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # ------------------------------Scene------------------------------
        self.scene.robot = UNITREE_B2W_PIPER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        self.scene.height_scanner_base.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name

        # ------------------------------Observations (53D)------------------------------
        # Layout: base_ang_vel*0.25(3) + proj_grav(3) + vel_cmds(3)
        #       + leg_joint_pos(12) + (leg+wheel)_joint_vel*0.05(16) + actions(16)
        self.observations.policy.base_lin_vel = None
        self.observations.policy.height_scan = None

        self.observations.policy.base_ang_vel.scale = 0.25
        self.observations.policy.joint_pos.scale = 1.0
        self.observations.policy.joint_vel.scale = 0.05
        # Leg joint positions only (12D)
        self.observations.policy.joint_pos.params["asset_cfg"].joint_names = self.leg_joint_names
        # Leg + wheel joint velocities (16D)
        self.observations.policy.joint_vel.params["asset_cfg"].joint_names = (
            self.leg_joint_names + self.wheel_joint_names
        )

        # ------------------------------Actions (16D)------------------------------
        # 12D leg positions (with per-joint scaling) + 4D wheel velocities
        self.actions.joint_pos = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=self.leg_joint_names,
            scale={".*_hip_joint": 0.125, "^(?!.*_hip_joint).*": 0.25},
            use_default_offset=True,
            preserve_order=True,
        )
        self.actions.wheel_vel = mdp.JointVelocityActionCfg(
            asset_name="robot",
            joint_names=self.wheel_joint_names,
            scale=5.0,   # matches Task D joint_vel_wheel scale for consistent action semantics
            preserve_order=True,
        )

        # ------------------------------Commands------------------------------
        self.commands.base_velocity = mdp.ModeBasedVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=(10.0, 10.0),
            rel_standing_envs=0.0,   # standing is handled via mode sampler
            rel_heading_envs=0.0,
            heading_command=False,
            mode_probs=(0.25, 0.25, 0.25, 0.20, 0.05),
            ranges=mdp.UniformThresholdVelocityCommandCfg.Ranges(
                lin_vel_x=(-1.2, 1.2),
                lin_vel_y=(-1.0, 1.0),
                ang_vel_z=(-2.0, 2.0),
                heading=(0.0, 0.0),
            ),
        )

        # ------------------------------Events------------------------------
        self.events.randomize_reset_base.params = {
            "pose_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (0.0, 0.2),
                "roll": (-3.14, 3.14),
                "pitch": (-3.14, 3.14),
                "yaw": (-3.14, 3.14),
            },
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        }
        self.events.randomize_rigid_body_mass_base.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_rigid_body_mass_others.params["asset_cfg"].body_names = [
            f"^(?!.*{self.base_link_name}).*"
        ]
        self.events.randomize_com_positions.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_apply_external_force_torque.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_apply_external_force_torque.params["force_range"] = (-30.0, 30.0)
        self.events.randomize_apply_external_force_torque.params["torque_range"] = (-10.0, 10.0)

        # Restrict actuator gain randomization to leg + wheel joints only.
        # Arm joints stay at default stiffness so the PD controller holds them
        # stable without policy-driven arm actions.
        self.events.randomize_actuator_gains.params["asset_cfg"].joint_names = (
            self.leg_joint_names + self.wheel_joint_names
        )

        # ------------------------------Rewards------------------------------
        # General
        self.rewards.is_terminated.weight = 0

        # Root penalties
        self.rewards.lin_vel_z_l2.weight = -2.0
        self.rewards.ang_vel_xy_l2.weight = -0.05
        self.rewards.flat_orientation_l2.weight = 0
        self.rewards.base_height_l2.weight = 0
        self.rewards.base_height_l2.params["target_height"] = 0.53
        self.rewards.base_height_l2.params["asset_cfg"].body_names = [self.base_link_name]
        self.rewards.body_lin_acc_l2.weight = 0
        self.rewards.body_lin_acc_l2.params["asset_cfg"].body_names = [self.base_link_name]

        # Joint penalties
        self.rewards.joint_torques_l2.weight = -1e-5
        self.rewards.joint_vel_l2.weight = 0
        self.rewards.joint_acc_l2.weight = -1e-7
        self.rewards.joint_pos_limits.weight = -5.0
        self.rewards.joint_vel_limits.weight = 0
        self.rewards.joint_power.weight = -1e-5
        self.rewards.stand_still.weight = -2.0
        self.rewards.joint_pos_penalty.weight = -1.0
        self.rewards.joint_mirror.weight = -0.05
        self.rewards.joint_mirror.params["mirror_joints"] = [
            ["FR_(hip|thigh|calf).*", "RL_(hip|thigh|calf).*"],
            ["FL_(hip|thigh|calf).*", "RR_(hip|thigh|calf).*"],
        ]

        # Action penalties
        self.rewards.action_rate_l2.weight = -0.01

        # Contact sensor
        self.rewards.undesired_contacts.weight = -1.0
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [f"^(?!.*{self.foot_link_name}).*"]
        self.rewards.contact_forces.weight = -1.5e-4
        self.rewards.contact_forces.params["sensor_cfg"].body_names = [self.foot_link_name]

        # Velocity-tracking rewards
        self.rewards.track_lin_vel_xy_exp.weight = 3.0
        self.rewards.track_ang_vel_z_exp.weight = 1.5

        # Wheel air-spin penalty (suppress wheels free-spinning when airborne)
        self.rewards.wheel_vel_penalty.weight = -0.01
        self.rewards.wheel_vel_penalty.params["asset_cfg"].joint_names = self.wheel_joint_names
        # Wheel joints sit on foot bodies: FR_foot_joint → FR_foot, etc.
        self.rewards.wheel_vel_penalty.params["sensor_cfg"].body_names = [
            name.replace("_joint", "") for name in self.wheel_joint_names
        ]

        # Others
        self.rewards.feet_air_time.weight = 0
        self.rewards.feet_contact.weight = 0
        self.rewards.feet_contact_without_cmd.weight = 0.1
        self.rewards.feet_contact_without_cmd.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_stumble.weight = 0
        self.rewards.feet_slide.weight = 0
        self.rewards.feet_height.weight = 0
        self.rewards.feet_height_body.weight = -5.0
        self.rewards.feet_height_body.params["target_height"] = -0.4
        self.rewards.feet_height_body.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_gait.weight = 0
        self.rewards.upward.weight = 3.0

        # Disable zero-weight reward terms to prevent empty-joint/body regex
        # resolution errors (this is the concrete leaf class).
        self.disable_zero_weight_rewards()

        # ------------------------------Terminations------------------------------
        self.terminations.illegal_contact = None

        # ------------------------------Curriculums------------------------------
        self.curriculum.command_levels_lin_vel = None
        self.curriculum.command_levels_ang_vel = None
