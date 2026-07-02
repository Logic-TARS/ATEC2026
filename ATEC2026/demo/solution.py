import os
import importlib
import numpy as np
import torch

class AlgSolution:

    ACTION_SCALE = 0.5
    LEG_ACTION_GAIN = 1.0
    WHEEL_ACTION_VALUE = float(os.getenv("ATEC_WHEEL_ACTION_VALUE", "3.0"))
    EE_BODY_NAME_CANDIDATES = ("gripper_base", "piper_gripper_base")
    ARM_JOINT_NAME_CANDIDATES = (
        ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"],
        ["arm_joint1", "arm_joint2", "arm_joint3", "arm_joint4", "arm_joint5", "arm_joint6"],
    )
    POLICY_MODE_B2W_LOCOMOTION_56D = "b2w_locomotion_56d"
    POLICY_MODE_B2W_LOCOMOTION_56D_LEGACY = "b2w_handover56"
    HIGH_LEVEL_TASK_TASK_F_PUSH = "task_f_push"
    HIGH_LEVEL_TASK_TASK_F_PUSH_LEGACY = "taskf_push"
    HIGH_LEVEL_TASK_TASK_D_AUTO = "task_d_auto"
    HIGH_LEVEL_TASK_TASK_D_AUTO_LEGACY = "taskd_auto"
    HIGH_LEVEL_TASK_TASK_D_SCRIPTED_PATH = "task_d_scripted_path"
    HIGH_LEVEL_TASK_TASK_D_SCRIPTED_SIDE_PUSH = "task_d_scripted_side_push"
    HIGH_LEVEL_TASK_TASK_D_SCRIPTED_PIT_PUSH = "task_d_scripted_pit_push"
    HIGH_LEVEL_TASK_TASK_D_TELEOP = "task_d_teleop"
    HIGH_LEVEL_TASK_TASK_D_WAYPOINT_ROUTE = "task_d_waypoint_route"
    TASK_F_STATE_SEARCH_BOX = 0
    TASK_F_STATE_APPROACH_BOX = 1
    TASK_F_STATE_ALIGN_PUSH = 2
    TASK_F_STATE_PUSH_BOX = 3
    TASK_F_STATE_RECOVERY = 4
    TASK_D_STATE_APPROACH_BOX = 0
    TASK_D_STATE_PUSH_BOX = 1
    TASK_D_STATE_NAV_PLATFORM = 2
    TASK_D_STATE_CLIMB_FINISH = 3
    TASK_D_SCRIPT_STATE_BACK_UP = 10
    TASK_D_SCRIPT_STATE_STRAFE_TO_BOX_LINE = 11
    TASK_D_SCRIPT_STATE_SETTLE = 12
    TASK_D_SCRIPT_STATE_PUSH_BOX = 13
    TASK_D_SCRIPT_STATE_NAV_FINISH = 14
    TASK_D_SIDE_STATE_BACK_UP_LONG = 20
    TASK_D_SIDE_STATE_STRAFE_TO_BOX_TOP = 21
    TASK_D_SIDE_STATE_FORWARD_TO_BOX_SIDE = 22
    TASK_D_SIDE_STATE_PUSH_BOX_TO_Y_NEG = 23
    TASK_D_SIDE_STATE_BACK_UP_BEHIND_BOX = 24
    TASK_D_SIDE_STATE_SETTLE = 25
    TASK_D_SIDE_STATE_FINAL_PUSH_X = 26
    TASK_D_SIDE_STATE_RECOVER_BOX = 27
    TASK_D_PIT_STATE_BACK_UP_TO_BOX_SIDE = 30
    TASK_D_PIT_STATE_STRAFE_TO_BOX_Y = 31
    TASK_D_PIT_STATE_PUSH_BOX_TO_PATH_Y = 32
    TASK_D_PIT_STATE_MOVE_BEHIND_BOX = 33
    TASK_D_PIT_STATE_SETTLE = 34
    TASK_D_PIT_STATE_PUSH_BOX_TO_PIT_X = 35
    TASK_D_PIT_STATE_CROSS_PIT = 36

    def __init__(self):
        # Policy mode — overrides robot-type-specific inference behaviours.
        self._policy_mode = self._normalize_policy_mode(os.getenv("ATEC_POLICY_MODE", ""))
        if os.getenv("ATEC_REGISTER_TRAIN_ENVS", "0") == "1":
            self._register_training_task_envs_if_needed()

        policy_path = os.getenv("ATEC_POLICY_PATH", self._default_policy_path())
        self.device = 'cuda'

        self.policy = torch.jit.load(policy_path, map_location=self.device)
        self.policy.eval()

        # Detect robot type
        self._robot_type = os.getenv("ATEC_ROBOT_TYPE", "")
        if self._robot_type == "tron2awheel":
            self.leg_action_dim = 8
            self.wheel_action_dim = 2
            self.arm_action_dim = 8
        else:
            self.leg_action_dim = 12
            self.wheel_action_dim = 4
            self.arm_action_dim = 8
        self.leg_joint_indices = list(range(self.leg_action_dim))

        # Omni-16 / Task D 61D mode: shared 16D action setup (12 legs + 4 wheels).
        if self._policy_mode in ("b2w_omni16", "b2w_taskd61", self.POLICY_MODE_B2W_LOCOMOTION_56D):
            self.leg_action_dim = 12
            self.wheel_action_dim = 4
            self.policy_out_dim = 16
            self.wheel_joint_indices = list(range(12, 16))
            self.arm_joint_indices = list(range(16, 24))

        if self._robot_type == "tron2awheel":
            _t2e = [0.125, 0.25, 0.25, 0.125, 0.125, 0.25, 0.25, 0.125]
            _e2t = [8.0, 4.0, 4.0, 8.0, 8.0, 4.0, 4.0, 8.0]
        else:
            _t2e = [0.125, 0.25, 0.25, 0.125, 0.25, 0.25, 0.125, 0.25, 0.25, 0.125, 0.25, 0.25]
            _e2t = [8.0, 4.0, 4.0, 8.0, 4.0, 4.0, 8.0, 4.0, 4.0, 8.0, 4.0, 4.0]
        self.train_to_env_action_scale = torch.tensor(
            _t2e, device=self.device, dtype=torch.float32,
        ).view(1, -1)
        self.env_to_train_action_scale = torch.tensor(
            _e2t, device=self.device, dtype=torch.float32,
        ).view(1, -1)

        # Task D state machine
        self._state = self.TASK_D_STATE_APPROACH_BOX
        self._state_enter_step = 0
        self._step_counter = 0
        self._prev_score = 0.0

        # Task D command mode for debugging policy omnidirectional capability.
        # Values: forward, backward, lateral_left, lateral_right, yaw_left,
        #         yaw_right, zero, auto (default state machine).
        self._taskd_command_mode = os.getenv("ATEC_TASKD_COMMAND_MODE", "auto")
        self._high_level_task = self._normalize_high_level_task(os.getenv(
            "ATEC_HIGH_LEVEL_TASK",
            self.HIGH_LEVEL_TASK_TASK_D_AUTO,
        ))
        self._fixed_cmds = {
            "forward":       ( 0.3,  0.0,  0.0),
            "backward":      (-0.3,  0.0,  0.0),
            "lateral_left":  ( 0.0,  0.3,  0.0),
            "lateral_right": ( 0.0, -0.3,  0.0),
            "yaw_left":      ( 0.0,  0.0,  0.5),
            "yaw_right":     ( 0.0,  0.0, -0.5),
            "zero":          ( 0.0,  0.0,  0.0),
        }

        # Stuck-recovery state (auto mode only)
        self._stuck_check_interval = 500   # steps (~10 s at 0.02 s/dt)
        self._stuck_score_threshold = 0.2  # minimum score increment
        self._stuck_check_score = 0.0
        self._stuck_check_step = 0
        self._recovery_phase = 0           # 0=idle, 1=backward, 2=yaw, 3=lateral
        self._recovery_phase_start = 0
        # Recovery schedule: (vx, vy, hd, duration_steps)
        self._recovery_sequence = [
            (-0.3,  0.0,  0.0, 150),   # backward 3s
            ( 0.0,  0.0,  0.8, 100),   # yaw sweep 2s
            ( 0.0,  0.3,  0.0, 100),   # lateral search 2s
        ]

        # State-dependent velocity commands [v_x, v_y, heading]
        self._state_vel_cmds = {
            self.TASK_D_STATE_APPROACH_BOX: (0.25, 0.0, 0.0),
            self.TASK_D_STATE_PUSH_BOX: (0.35, 0.0, 0.0),
            self.TASK_D_STATE_NAV_PLATFORM: (0.40, 0.0, 0.0),
            self.TASK_D_STATE_CLIMB_FINISH: (0.45, 0.0, 0.0),
        }
        self._script_backup_steps = int(os.getenv("ATEC_TASKD_SCRIPT_BACKUP_STEPS", "150"))
        self._script_strafe_steps = int(os.getenv("ATEC_TASKD_SCRIPT_STRAFE_STEPS", "300"))
        self._script_settle_steps = int(os.getenv("ATEC_TASKD_SCRIPT_SETTLE_STEPS", "40"))
        self._script_push_vx = float(os.getenv("ATEC_TASKD_SCRIPT_PUSH_VX", "0.35"))
        self._script_strafe_vy = float(os.getenv("ATEC_TASKD_SCRIPT_STRAFE_VY", "0.25"))
        self._script_nav_vx = float(os.getenv("ATEC_TASKD_SCRIPT_NAV_VX", "0.40"))
        self._script_state = self.TASK_D_SCRIPT_STATE_BACK_UP
        self._side_backup_steps = int(os.getenv("ATEC_TASKD_SIDE_BACKUP_STEPS", "430"))
        self._side_strafe_top_steps = int(os.getenv("ATEC_TASKD_SIDE_STRAFE_TOP_STEPS", "400"))
        self._side_forward_to_box_steps = int(os.getenv("ATEC_TASKD_SIDE_FORWARD_TO_BOX_STEPS", "600"))
        self._side_push_y_steps = int(os.getenv("ATEC_TASKD_SIDE_PUSH_Y_STEPS", "900"))
        self._side_backup_behind_steps = int(os.getenv("ATEC_TASKD_SIDE_BACKUP_BEHIND_STEPS", "400"))
        self._side_final_push_vx = float(os.getenv("ATEC_TASKD_SIDE_FINAL_PUSH_VX", "0.36"))
        self._side_lateral_vy = float(os.getenv("ATEC_TASKD_SIDE_LATERAL_VY", "0.25"))
        self._side_state = self.TASK_D_SIDE_STATE_BACK_UP_LONG
        self._side_state_enter_step = 0
        self._side_state_enter_score = 0.0
        self._side_score_check_interval = int(os.getenv("ATEC_TASKD_SCORE_CHECK_INTERVAL", "250"))
        self._side_score_progress_eps = float(os.getenv("ATEC_TASKD_SCORE_PROGRESS_EPS", "0.05"))
        self._side_score_check_step = 0
        self._side_score_check_score = 0.0
        self._side_recovery_start_step = 0
        self._side_recovery_attempt = 0
        self._side_recovery_resume_until_step = 0
        self._taskd_heading_lock_enabled = os.getenv("ATEC_TASKD_HEADING_LOCK", "1") != "0"
        self._taskd_heading_kd = float(os.getenv("ATEC_TASKD_HEADING_KD", "0.35"))
        self._taskd_heading_max_yaw = float(os.getenv("ATEC_TASKD_HEADING_MAX_YAW", "0.22"))
        self._taskd_speed_lock_enabled = os.getenv("ATEC_TASKD_SPEED_LOCK", "1") != "0"
        self._taskd_speed_kp = float(os.getenv("ATEC_TASKD_SPEED_KP", "0.35"))
        self._taskd_cross_vel_kp = float(os.getenv("ATEC_TASKD_CROSS_VEL_KP", "0.25"))
        self._taskd_speed_correction_max = float(os.getenv("ATEC_TASKD_SPEED_CORRECTION_MAX", "0.10"))
        self._taskd_lidar_contact_distance = float(os.getenv("ATEC_TASKD_LIDAR_CONTACT_DISTANCE", "1.25"))
        self._taskd_lidar_lost_distance = float(os.getenv("ATEC_TASKD_LIDAR_LOST_DISTANCE", "2.20"))
        self._taskd_lidar_approach_stop_distance = float(os.getenv("ATEC_TASKD_LIDAR_APPROACH_STOP_DISTANCE", "1.05"))
        self._pit_backup_steps = int(os.getenv("ATEC_TASKD_PIT_BACKUP_STEPS", "180"))
        self._pit_strafe_steps = int(os.getenv("ATEC_TASKD_PIT_STRAFE_STEPS", "400"))
        self._pit_push_y_steps = int(os.getenv("ATEC_TASKD_PIT_PUSH_Y_STEPS", "520"))
        self._pit_backup_behind_steps = int(os.getenv("ATEC_TASKD_PIT_BACKUP_BEHIND_STEPS", "180"))
        self._pit_push_x_steps = int(os.getenv("ATEC_TASKD_PIT_PUSH_X_STEPS", "950"))
        self._pit_push_vx = float(os.getenv("ATEC_TASKD_PIT_PUSH_VX", "0.34"))
        self._pit_lateral_vy = float(os.getenv("ATEC_TASKD_PIT_LATERAL_VY", "0.24"))
        self._pit_state = self.TASK_D_PIT_STATE_BACK_UP_TO_BOX_SIDE
        self._pit_backup_guard_end_step = 0
        self._pit_backup_behind_guard_end_step = 0
        self._taskd_backup_guard_enabled = os.getenv("ATEC_TASKD_BACKUP_GUARD", "1") != "0"
        self._taskd_backup_edge_distance = float(os.getenv("ATEC_TASKD_BACKUP_EDGE_DISTANCE", "3.5"))
        self._taskd_backup_max_tilt = float(os.getenv("ATEC_TASKD_BACKUP_MAX_TILT", "0.38"))
        self._taskd_backup_safe_vx = float(os.getenv("ATEC_TASKD_BACKUP_SAFE_VX", "-0.12"))
        self._teleop_max_vx = float(os.getenv("ATEC_TELEOP_MAX_VX", "0.45"))
        self._teleop_max_vy = float(os.getenv("ATEC_TELEOP_MAX_VY", "0.35"))
        self._teleop_max_yaw = float(os.getenv("ATEC_TELEOP_MAX_YAW", "0.60"))
        self._teleop_command = (0.0, 0.0, 0.0)
        self._waypoint_route = (
            ("BOX_BACK", -4.5, 1.6),
            ("PUSH_FORWARD", -2.0, 1.6),
            ("BOX_LEFT", -1.5, 2.7),
            ("PUSH_TO_PIT_SIDE", -1.5, 0.8),
        )
        self._waypoint_index = 0
        self._waypoint_enter_step = 0
        self._odom_x = float(os.getenv("ATEC_TASKD_ODOM_INIT_X", "-3.0"))
        self._odom_y = float(os.getenv("ATEC_TASKD_ODOM_INIT_Y", "0.0"))
        self._odom_yaw = float(os.getenv("ATEC_TASKD_ODOM_INIT_YAW", "0.0"))
        self._odom_dt = float(os.getenv("ATEC_TASKD_ODOM_DT", "0.02"))
        self._waypoint_kp = float(os.getenv("ATEC_TASKD_WAYPOINT_KP", "0.45"))
        self._waypoint_max_vx = float(os.getenv("ATEC_TASKD_WAYPOINT_MAX_VX", "0.32"))
        self._waypoint_max_vy = float(os.getenv("ATEC_TASKD_WAYPOINT_MAX_VY", "0.28"))
        self._waypoint_reached_dist = float(os.getenv("ATEC_TASKD_WAYPOINT_REACHED_DIST", "0.20"))
        self._waypoint_max_steps = int(os.getenv("ATEC_TASKD_WAYPOINT_MAX_STEPS", "600"))
        self._taskf_state = self.TASK_F_STATE_SEARCH_BOX
        self._taskf_state_enter_step = 0
        self._taskf_last_seen_step = -100000
        self._taskf_lost_box_grace = 80
        self._taskf_stuck_check_interval = 350
        self._taskf_stuck_score_threshold = 0.08
        self._taskf_training_seen_steps = 0
        self._taskf_training_push_ready_steps = 0
        self._taskf_training_last_distance_norm = None

        self.arm_default_action = torch.zeros(
            (1, self.arm_action_dim),
            device=self.device,
            dtype=torch.float32,
        )
        self.wheel_forward_action = torch.full(
            (1, self.wheel_action_dim),
            self.WHEEL_ACTION_VALUE,
            device=self.device,
            dtype=torch.float32,
        )

    @classmethod
    def _normalize_policy_mode(cls, policy_mode: str) -> str:
        if policy_mode == cls.POLICY_MODE_B2W_LOCOMOTION_56D_LEGACY:
            return cls.POLICY_MODE_B2W_LOCOMOTION_56D
        return policy_mode

    @classmethod
    def _normalize_high_level_task(cls, high_level_task: str) -> str:
        if high_level_task == cls.HIGH_LEVEL_TASK_TASK_F_PUSH_LEGACY:
            return cls.HIGH_LEVEL_TASK_TASK_F_PUSH
        if high_level_task == cls.HIGH_LEVEL_TASK_TASK_D_AUTO_LEGACY:
            return cls.HIGH_LEVEL_TASK_TASK_D_AUTO
        return high_level_task

    def _default_policy_path(self) -> str:
        demo_dir = os.path.dirname(os.path.abspath(__file__))
        if self._policy_mode != self.POLICY_MODE_B2W_LOCOMOTION_56D:
            return os.path.join(demo_dir, "policy.pt")

        preferred_path = os.path.join(demo_dir, "policy_b2w_locomotion_56d.pt")
        if os.path.exists(preferred_path):
            return preferred_path
        return os.path.join(demo_dir, "policy_handover56.pt")

    def set_teleop_command(self, vx: float, vy: float, yaw: float) -> None:
        self._teleop_command = (
            self._clip(float(vx), -self._teleop_max_vx, self._teleop_max_vx),
            self._clip(float(vy), -self._teleop_max_vy, self._teleop_max_vy),
            self._clip(float(yaw), -self._teleop_max_yaw, self._teleop_max_yaw),
        )

    def _register_training_task_envs_if_needed(self) -> None:
        if self._policy_mode != self.POLICY_MODE_B2W_LOCOMOTION_56D:
            return
        try:
            importlib.import_module("atec_rl_lab.train.locomotion.velocity.config.quadruped.unitree_b2")
        except Exception:
            pass


    def _resolve_joint_ids(self, candidates: tuple[list[str], ...]) -> list[int]:
        last_error = None
        for names in candidates:
            try:
                ids, found_names = self.robot.find_joints(names)
            except ValueError as err:
                last_error = err
                continue
            if len(ids) == len(names):
                if candidates is self.ARM_JOINT_NAME_CANDIDATES:
                    self.arm_joint_names = list(found_names)
                return list(ids)
        raise ValueError(
            f"Cannot resolve required joints from candidates: {candidates}. Last error: {last_error}"
        )

    def _resolve_ee_body_name(self) -> str:
        last_error = None
        for name in self.EE_BODY_NAME_CANDIDATES:
            try:
                body_ids, _ = self.robot.find_bodies(name)
            except ValueError as err:
                last_error = err
                continue
            if len(body_ids) == 1:
                return name
        raise ValueError(
            f"Cannot resolve EE body from candidates: {self.EE_BODY_NAME_CANDIDATES}. Last error: {last_error}"
        )

    def _ensure_cartesian_targets(self):
        self.cartesian_ctrl.reset()

    def _compute_arm_overlay_action(self) -> torch.Tensor:
        self._ensure_cartesian_targets()

        arm_jpos_des = self.cartesian_ctrl.compute_base(
            self.ee_pos_target_b,
            self.ee_quat_target_b,
        )

        full_target = self.robot.data.joint_pos.clone()
        full_target[:, self.arm_ids] = arm_jpos_des
        full_target[:, self.gripper_ids] = self.gripper_open_pos.repeat(full_target.shape[0], 1)

        return (full_target - self.default_joint_pos) / self.ACTION_SCALE

    def _get_velocity_commands(self, proprio: torch.Tensor) -> torch.Tensor:
        """Return fixed velocity commands for policy input."""
        num_envs = proprio.shape[0]

        cmd = self.fixed_velocity_commands.to(dtype=proprio.dtype, device=self.device)
        if num_envs > 1:
            cmd = cmd.repeat(num_envs, 1)
        return cmd

    def _extract_policy_obs(self, obs, action_dim) -> torch.Tensor:
        if self._policy_mode == self.POLICY_MODE_B2W_LOCOMOTION_56D:
            zero_cmd = torch.zeros((obs["proprio"].shape[0], 3), device=self.device, dtype=obs["proprio"].dtype)
            return self._build_b2w_locomotion_56d_obs(obs, action_dim, zero_cmd)

        if self._policy_mode == "b2w_omni16":
            return self._extract_policy_obs_omni16(obs, action_dim)

        proprio = obs["proprio"].to(self.device)

        idx = 0
        _base_lin_vel = proprio[:, idx:idx + 3]
        idx += 3

        base_ang_vel = proprio[:, idx:idx + 3]
        idx += 3

        velocity_commands_env = proprio[:, idx:idx + 3]
        idx += 3

        projected_gravity = proprio[:, idx:idx + 3]
        idx += 3

        joint_pos_all = proprio[:, idx:idx + action_dim]
        idx += action_dim

        joint_vel_all = proprio[:, idx:idx + action_dim]
        idx += action_dim

        actions_all = proprio[:, idx:idx + action_dim]

        joint_pos_leg = joint_pos_all[:, self.leg_joint_indices]
        joint_vel_leg = joint_vel_all[:, self.leg_joint_indices]
        actions_env_leg = actions_all[:, self.leg_joint_indices]

        actions_train_leg = actions_env_leg * self.env_to_train_action_scale.to(dtype=proprio.dtype)

        policy_obs = torch.cat(
            [
                base_ang_vel * 0.25,
                projected_gravity,
                velocity_commands_env,
                joint_pos_leg,
                joint_vel_leg * 0.05,
                actions_train_leg,
            ],
            dim=-1,
        )

        return policy_obs

    def _extract_policy_obs_omni16(self, obs, action_dim) -> torch.Tensor:
        """Build 53-D policy observation for the omni-16 B2W+Piper policy.

        Layout:
          base_ang_vel * 0.25   (3)
          projected_gravity      (3)
          velocity_commands      (3)   ← caller may override with task-state commands
          leg_joint_pos          (12)
          leg_joint_vel * 0.05   (12)
          wheel_vel * 0.05       (4)
          prev_leg_actions_train (12)  ← env-scale → train-scale
          prev_wheel_actions     (4)   ← pre-scaling value (both training and Task D use scale=5.0)
        """
        proprio = obs["proprio"].to(self.device)

        idx = 0
        _base_lin_vel = proprio[:, idx:idx + 3]
        idx += 3

        base_ang_vel = proprio[:, idx:idx + 3]
        idx += 3

        velocity_commands_env = proprio[:, idx:idx + 3]
        idx += 3

        projected_gravity = proprio[:, idx:idx + 3]
        idx += 3

        joint_pos_all = proprio[:, idx:idx + action_dim]
        idx += action_dim

        joint_vel_all = proprio[:, idx:idx + action_dim]
        idx += action_dim

        actions_all = proprio[:, idx:idx + action_dim]

        # Leg subset (indices 0–11) and wheel subset (indices 12–15).
        leg_joint_pos = joint_pos_all[:, :12]
        leg_joint_vel = joint_vel_all[:, :12]
        wheel_vel = joint_vel_all[:, 12:16]

        prev_leg_actions_env = actions_all[:, :12]
        prev_wheel_actions = actions_all[:, 12:16]

        prev_leg_actions_train = prev_leg_actions_env * self.env_to_train_action_scale.to(
            dtype=proprio.dtype
        )

        policy_obs = torch.cat(
            [
                base_ang_vel * 0.25,
                projected_gravity,
                velocity_commands_env,
                leg_joint_pos,
                leg_joint_vel * 0.05,
                wheel_vel * 0.05,
                prev_leg_actions_train,
                prev_wheel_actions,
            ],
            dim=-1,
        )

        return policy_obs

    def _build_b2w_locomotion_56d_obs(self, obs, action_dim, vel_cmd: torch.Tensor) -> torch.Tensor:
        """Build the 56-D observation expected by the B2W locomotion policy.

        Layout:
          base_lin_vel (3), base_ang_vel (3), projected_gravity (3), vel_cmd (3),
          leg_joint_pos (12), leg+wheel_joint_vel (16), previous leg+wheel actions (16).
        """
        proprio = obs["proprio"].to(self.device)
        if proprio.ndim == 1:
            proprio = proprio.unsqueeze(0)
        if action_dim < 16:
            raise ValueError(f"b2w_locomotion_56d requires at least 16 robot actions, got action_dim={action_dim}")

        if vel_cmd.shape[0] != proprio.shape[0]:
            vel_cmd = vel_cmd.repeat(proprio.shape[0], 1)
        vel_cmd = vel_cmd.to(device=self.device, dtype=proprio.dtype)

        idx = 0
        base_lin_vel = proprio[:, idx:idx + 3]
        idx += 3
        base_ang_vel = proprio[:, idx:idx + 3]
        idx += 3
        idx += 3  # Skip env velocity_commands; the high-level controller supplies vel_cmd.
        projected_gravity = proprio[:, idx:idx + 3]
        idx += 3
        joint_pos_all = proprio[:, idx:idx + action_dim]
        idx += action_dim
        joint_vel_all = proprio[:, idx:idx + action_dim]
        idx += action_dim
        actions_all = proprio[:, idx:idx + action_dim]

        return torch.cat(
            [
                base_lin_vel,
                base_ang_vel,
                projected_gravity,
                vel_cmd,
                joint_pos_all[:, :12],
                joint_vel_all[:, :16],
                actions_all[:, :16],
            ],
            dim=-1,
        )

    def _build_b2w_locomotion_56d_obs_from_training_groups(
        self,
        obs,
        vel_cmd: torch.Tensor,
    ) -> torch.Tensor:
        """Build the same 56-D policy input from ManagerBasedRLEnv policy/critic groups."""
        policy_obs = obs["policy"].to(self.device)
        critic_obs = obs["critic"].to(self.device)
        if policy_obs.ndim == 1:
            policy_obs = policy_obs.unsqueeze(0)
        if critic_obs.ndim == 1:
            critic_obs = critic_obs.unsqueeze(0)
        if policy_obs.shape[-1] < 61 or critic_obs.shape[-1] < 76:
            raise ValueError(
                "Training Task F obs requires policy>=61 and critic>=76, "
                f"got policy={tuple(policy_obs.shape)}, critic={tuple(critic_obs.shape)}"
            )
        if vel_cmd.shape[0] != critic_obs.shape[0]:
            vel_cmd = vel_cmd.repeat(critic_obs.shape[0], 1)
        vel_cmd = vel_cmd.to(device=self.device, dtype=critic_obs.dtype)

        return torch.cat(
            [
                critic_obs[:, 0:3],    # base_lin_vel
                critic_obs[:, 3:6],    # base_ang_vel
                critic_obs[:, 6:9],    # projected_gravity
                vel_cmd,
                critic_obs[:, 12:24],  # leg_joint_pos
                critic_obs[:, 36:52],  # leg+wheel_joint_vel
                policy_obs[:, 37:53],  # previous leg+wheel actions
            ],
            dim=-1,
        )

    def _map_policy_action_to_env_action(self, action_train: torch.Tensor, action_dim: int) -> torch.Tensor:
        """Map policy leg action to env full-body action."""
        num_envs = action_train.shape[0]

        if self._policy_mode == self.POLICY_MODE_B2W_LOCOMOTION_56D:
            action_env = torch.zeros(
                (num_envs, action_dim), device=self.device, dtype=torch.float32,
            )
            n = min(action_train.shape[-1], 16, action_dim)
            action_env[:, :n] = torch.nan_to_num(
                action_train[:, :n].to(device=self.device, dtype=torch.float32),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            return action_env

        if self._policy_mode in ("b2w_omni16", "b2w_taskd61"):
            # Policy outputs 16D: [leg_pos_train(12), wheel_vel(4)].
            # Map to 24D env action: legs[0:12], wheels[12:16], arms[16:24]=0.
            leg_action_env = (
                action_train[:, :12] * self.train_to_env_action_scale * self.LEG_ACTION_GAIN
            )
            wheel_action_env = action_train[:, 12:16]  # pre-scaling; env applies ×5.0 → rad/s

            action_env = torch.zeros(
                (num_envs, action_dim), device=self.device, dtype=torch.float32,
            )
            action_env[:, :12] = leg_action_env
            action_env[:, 12:16] = wheel_action_env
            # arms[16:24] remain zero — held by PD controller
            return action_env

        if self._robot_type == "tron2awheel":
            # Tron2AWheel: legs[0:8], wheels[8:10], arms[10:18]
            action_env = torch.zeros((num_envs, action_dim), device=self.device, dtype=torch.float32)
            leg_action_env = action_train * self.train_to_env_action_scale * self.LEG_ACTION_GAIN
            action_env[:, 0:8] = leg_action_env
            # Wheels at [8:10] — forward at WHEEL_ACTION_VALUE
            action_env[:, 8:10] = self.WHEEL_ACTION_VALUE
            return action_env

        if action_train.shape[-1] != self.leg_action_dim:
            raise ValueError(
                f"Policy output dim mismatch: got {action_train.shape[-1]}, expected {self.leg_action_dim}"
            )

        leg_action_env = action_train * self.train_to_env_action_scale * self.LEG_ACTION_GAIN

        action_env = torch.zeros(
            (num_envs, action_dim),
            device=self.device,
            dtype=torch.float32,
        )

        action_env[:, self.leg_joint_indices] = leg_action_env

        if action_dim >= self.leg_action_dim + self.wheel_action_dim + self.arm_action_dim:
            wheel_joint_indices = list(range(12, 16))
            arm_joint_indices = list(range(16, 24))
            action_env[:, wheel_joint_indices] = self.wheel_forward_action.repeat(num_envs, 1)
        else:
            arm_joint_indices = list(range(12, 20))

        action_env[:, arm_joint_indices] = self.arm_default_action.repeat(num_envs, 1)

        return action_env

    def _extract_policy_obs_taskd61(self, obs, action_dim, current_score) -> torch.Tensor:
        """Build 61-D policy observation for Task D fine-tuned policy.

        Layout:
          [0:53]  omni16 base (identical to _extract_policy_obs_omni16)
          [53:54] score_norm              = current_score / 36.0, clamped [0, 1]
          [54:58] stage_one_hot           = 4D one-hot based on score thresholds [1.9, 15.0, 21.0]
          [58:59] box_detected            = 1.0 if box in LiDAR view, else 0.0
          [59:60] box_bearing             = normalized bearing [-1, 1]
          [60:61] box_distance_norm       = distance / 5.0, clamped [0, 1]
        """
        policy_obs_53 = self._extract_policy_obs_omni16(obs, action_dim)

        score_norm = (current_score / 36.0)
        score_norm = float(max(0.0, min(1.0, score_norm)))
        score = current_score

        thresholds = [1.9, 15.0, 21.0]
        stage = 0
        if score >= thresholds[2]:
            stage = 3
        elif score >= thresholds[1]:
            stage = 2
        elif score >= thresholds[0]:
            stage = 1
        stage_one_hot = [0.0, 0.0, 0.0, 0.0]
        stage_one_hot[stage] = 1.0

        box_detected = 0.0
        box_bearing = 0.0
        box_distance_norm = 0.0
        extero = obs.get("extero")
        if extero is not None:
            lidar_info = self._parse_lidar(extero)
            if lidar_info.get("box_detected"):
                box_detected = 1.0
                bearing = lidar_info.get("box_bearing", 0.0)
                box_bearing = float(np.clip(-bearing * 1.0, -1.0, 1.0))
                dist = lidar_info.get("box_distance", 999.0)
                box_distance_norm = float(min(dist / 5.0, 1.0))

        suffix = torch.tensor(
            [[score_norm] + stage_one_hot + [box_detected, box_bearing, box_distance_norm]],
            device=self.device,
            dtype=policy_obs_53.dtype,
        )

        return torch.cat([policy_obs_53, suffix], dim=-1)

    def _run_policy_with_command(self, obs, action_dim: int, vel_cmd: torch.Tensor) -> torch.Tensor:
        if self._policy_mode == self.POLICY_MODE_B2W_LOCOMOTION_56D:
            policy_obs = self._build_b2w_locomotion_56d_obs(obs, action_dim, vel_cmd)
        elif self._policy_mode == "b2w_taskd61":
            policy_obs = self._extract_policy_obs_taskd61(obs, action_dim, self._prev_score)
            policy_obs = torch.cat([policy_obs[:, :6], vel_cmd, policy_obs[:, 9:]], dim=-1)
        else:
            policy_obs = self._extract_policy_obs(obs, action_dim)
            policy_obs = torch.cat([policy_obs[:, :6], vel_cmd, policy_obs[:, 9:]], dim=-1)

        with torch.inference_mode():
            action_train = self.policy(policy_obs)
        if not isinstance(action_train, torch.Tensor):
            action_train = torch.as_tensor(action_train, device=self.device, dtype=torch.float32)
        action_train = action_train.to(device=self.device, dtype=torch.float32)
        if action_train.ndim == 1:
            action_train = action_train.unsqueeze(0)
        return self._map_policy_action_to_env_action(action_train, action_dim)

    @staticmethod
    def _clip(value: float, low: float, high: float) -> float:
        return float(np.clip(value, low, high))

    @staticmethod
    def _wrap_pi(value: float) -> float:
        return float((value + np.pi) % (2.0 * np.pi) - np.pi)

    def _update_taskd_odometry(self, obs) -> None:
        try:
            proprio = obs["proprio"].to(self.device)
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            if proprio.shape[-1] < 6:
                return
            base_vx = float(proprio[0, 0].item())
            base_vy = float(proprio[0, 1].item())
            base_yaw_rate = float(proprio[0, 5].item())
        except Exception:
            return

        dt = self._odom_dt
        cos_yaw = float(np.cos(self._odom_yaw))
        sin_yaw = float(np.sin(self._odom_yaw))
        self._odom_x += (cos_yaw * base_vx - sin_yaw * base_vy) * dt
        self._odom_y += (sin_yaw * base_vx + cos_yaw * base_vy) * dt
        self._odom_yaw = self._wrap_pi(self._odom_yaw + base_yaw_rate * dt)

    def _compute_task_d_waypoint_route_command(self, obs, current_score: float):
        if self._waypoint_index >= len(self._waypoint_route):
            vx = 0.45 if current_score >= 21.0 else 0.38
            return vx, 0.0, 0.0

        _, target_x, target_y = self._waypoint_route[self._waypoint_index]
        dx = target_x - self._odom_x
        dy = target_y - self._odom_y
        distance = float(np.hypot(dx, dy))
        elapsed = self._step_counter - self._waypoint_enter_step

        if distance <= self._waypoint_reached_dist or elapsed >= self._waypoint_max_steps:
            self._waypoint_index += 1
            self._waypoint_enter_step = self._step_counter
            if self._waypoint_index >= len(self._waypoint_route):
                return 0.0, 0.0, 0.0
            _, target_x, target_y = self._waypoint_route[self._waypoint_index]
            dx = target_x - self._odom_x
            dy = target_y - self._odom_y

        vx = self._clip(self._waypoint_kp * dx, -self._waypoint_max_vx, self._waypoint_max_vx)
        vy = self._clip(self._waypoint_kp * dy, -self._waypoint_max_vy, self._waypoint_max_vy)
        yaw_cmd = 0.0

        lidar_info = self._parse_lidar(obs.get("extero"))
        if lidar_info.get("box_detected") and lidar_info.get("box_distance", 999.0) < 2.4:
            bearing = float(lidar_info.get("box_bearing", 0.0))
            if abs(bearing) > 0.12:
                yaw_cmd = self._clip(-0.25 * bearing, -0.12, 0.12)

        return vx, vy, yaw_cmd

    def _apply_taskd_heading_lock(self, obs, vx: float, vy: float, yaw_cmd: float):
        if (
            not self._taskd_heading_lock_enabled
            or abs(yaw_cmd) > 1e-4
            or (abs(vx) < 1e-4 and abs(vy) < 1e-4)
        ):
            return vx, vy, yaw_cmd

        try:
            proprio = obs["proprio"].to(self.device)
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            if proprio.shape[-1] < 6:
                return vx, vy, yaw_cmd
            base_ang_vel_z = float(proprio[0, 5].item())
        except Exception:
            return vx, vy, yaw_cmd

        yaw_correction = self._clip(
            -self._taskd_heading_kd * base_ang_vel_z,
            -self._taskd_heading_max_yaw,
            self._taskd_heading_max_yaw,
        )
        return vx, vy, yaw_correction

    def _apply_taskd_speed_lock(self, obs, vx: float, vy: float, yaw_cmd: float):
        if (
            not self._taskd_speed_lock_enabled
            or (abs(vx) < 1e-4 and abs(vy) < 1e-4)
        ):
            return vx, vy, yaw_cmd

        try:
            proprio = obs["proprio"].to(self.device)
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            if proprio.shape[-1] < 3:
                return vx, vy, yaw_cmd
            actual_vx = float(proprio[0, 0].item())
            actual_vy = float(proprio[0, 1].item())
        except Exception:
            return vx, vy, yaw_cmd

        correction_max = max(0.0, self._taskd_speed_correction_max)
        if abs(vx) >= abs(vy):
            vx += self._clip(self._taskd_speed_kp * (vx - actual_vx), -correction_max, correction_max)
            vy += self._clip(-self._taskd_cross_vel_kp * actual_vy, -correction_max, correction_max)
        else:
            vy += self._clip(self._taskd_speed_kp * (vy - actual_vy), -correction_max, correction_max)
            vx += self._clip(-self._taskd_cross_vel_kp * actual_vx, -correction_max, correction_max)

        return self._clip(vx, -0.45, 0.45), self._clip(vy, -0.35, 0.35), yaw_cmd

    @staticmethod
    def _taskd_lidar_sector_distance(mid_scan, start: int, end: int) -> float:
        sector = mid_scan[:, start:end]
        valid = sector[(np.isfinite(sector)) & (sector > 0.20) & (sector < 5.0)]
        if valid.size < 8:
            return 999.0
        return float(np.percentile(valid, 10))

    def _taskd_lidar_wrapped_sector_distance(self, mid_scan, start: int, end: int) -> float:
        if start <= end:
            return self._taskd_lidar_sector_distance(mid_scan, start, end)
        first = self._taskd_lidar_sector_distance(mid_scan, start, 360)
        second = self._taskd_lidar_sector_distance(mid_scan, 0, end)
        return min(first, second)

    def _parse_taskd_lidar_contacts(self, extero):
        info = self._parse_lidar(extero)
        info.update({
            "front_distance": 999.0,
            "right_distance": 999.0,
            "left_distance": 999.0,
            "rear_distance": 999.0,
            "rear_left_distance": 999.0,
            "rear_right_distance": 999.0,
        })
        if extero is None:
            return info
        try:
            scan = extero.squeeze(0).cpu().numpy()
            if scan.size != 5760:
                return info
            mid = scan.reshape(16, 360)[4:12, :]
            info["right_distance"] = self._taskd_lidar_sector_distance(mid, 90, 171)
            info["front_distance"] = self._taskd_lidar_sector_distance(mid, 165, 196)
            info["left_distance"] = self._taskd_lidar_sector_distance(mid, 190, 271)
            info["rear_distance"] = self._taskd_lidar_wrapped_sector_distance(mid, 345, 16)
            info["rear_left_distance"] = self._taskd_lidar_sector_distance(mid, 15, 61)
            info["rear_right_distance"] = self._taskd_lidar_sector_distance(mid, 300, 346)
        except Exception:
            pass
        return info

    def _taskd_pit_phase_bounds(self):
        backup_end = self._pit_backup_steps
        if self._pit_backup_guard_end_step > 0:
            backup_end = min(backup_end, self._pit_backup_guard_end_step)
        strafe_end = backup_end + self._pit_strafe_steps
        push_y_end = strafe_end + self._pit_push_y_steps
        backup_behind_end = push_y_end + self._pit_backup_behind_steps
        if self._pit_backup_behind_guard_end_step > 0:
            backup_behind_end = min(backup_behind_end, self._pit_backup_behind_guard_end_step)
        settle_end = backup_behind_end + self._script_settle_steps
        push_x_end = settle_end + self._pit_push_x_steps
        return backup_end, strafe_end, push_y_end, backup_behind_end, settle_end, push_x_end

    def _apply_taskd_backup_guard(self, obs, vx: float, vy: float, yaw_cmd: float):
        is_pit_backup = (
            self._high_level_task == self.HIGH_LEVEL_TASK_TASK_D_SCRIPTED_PIT_PUSH
            and self._pit_state in (
                self.TASK_D_PIT_STATE_BACK_UP_TO_BOX_SIDE,
                self.TASK_D_PIT_STATE_MOVE_BEHIND_BOX,
            )
        )
        is_teleop_backup = self._high_level_task == self.HIGH_LEVEL_TASK_TASK_D_TELEOP
        is_waypoint_backup = self._high_level_task == self.HIGH_LEVEL_TASK_TASK_D_WAYPOINT_ROUTE
        if (
            not self._taskd_backup_guard_enabled
            or not (is_pit_backup or is_teleop_backup or is_waypoint_backup)
            or vx >= 0.0
        ):
            return vx, vy, yaw_cmd

        contacts = self._parse_taskd_lidar_contacts(obs.get("extero"))
        rear_distance = float(contacts.get("rear_distance", 999.0))
        rear_left_distance = float(contacts.get("rear_left_distance", 999.0))
        rear_right_distance = float(contacts.get("rear_right_distance", 999.0))
        rear_min = min(rear_distance, rear_left_distance, rear_right_distance)
        rear_invalid_or_far = rear_min >= self._taskd_backup_edge_distance

        tilt = 0.0
        actual_vx = 0.0
        base_ang_vel_z = 0.0
        try:
            proprio = obs["proprio"].to(self.device)
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            if proprio.shape[-1] >= 12:
                gravity_xy = proprio[0, 9:11]
                tilt = float(torch.linalg.norm(gravity_xy).item())
            if proprio.shape[-1] >= 6:
                actual_vx = float(proprio[0, 0].item())
                base_ang_vel_z = float(proprio[0, 5].item())
        except Exception:
            pass

        too_tilted = tilt > self._taskd_backup_max_tilt
        backing_too_fast = actual_vx < self._taskd_backup_safe_vx * 2.5
        if rear_invalid_or_far or too_tilted:
            if is_teleop_backup or is_waypoint_backup:
                return 0.0, 0.0, 0.0
            if self._pit_state == self.TASK_D_PIT_STATE_BACK_UP_TO_BOX_SIDE:
                self._pit_backup_guard_end_step = self._step_counter
                self._pit_state = self.TASK_D_PIT_STATE_STRAFE_TO_BOX_Y
            elif self._pit_state == self.TASK_D_PIT_STATE_MOVE_BEHIND_BOX:
                self._pit_backup_behind_guard_end_step = self._step_counter
                self._pit_state = self.TASK_D_PIT_STATE_SETTLE
            return 0.0, 0.0, 0.0

        if backing_too_fast:
            vx = max(vx, self._taskd_backup_safe_vx)
        yaw_cmd = self._clip(yaw_cmd - 0.20 * base_ang_vel_z, -0.15, 0.15)
        return vx, vy, yaw_cmd

    def _apply_taskd_lidar_contact_adjustment(self, obs, current_score: float, vx: float, vy: float, yaw_cmd: float):
        contacts = self._parse_taskd_lidar_contacts(obs.get("extero"))
        front_distance = float(contacts.get("front_distance", 999.0))
        right_distance = float(contacts.get("right_distance", 999.0))
        box_distance = float(contacts.get("box_distance", 999.0))
        box_seen = bool(contacts.get("box_detected"))
        box_close = box_seen and box_distance < self._taskd_lidar_lost_distance
        contact_close = min(front_distance, right_distance, box_distance) < self._taskd_lidar_contact_distance

        if self._high_level_task == self.HIGH_LEVEL_TASK_TASK_D_SCRIPTED_PATH:
            if self._script_state == self.TASK_D_SCRIPT_STATE_PUSH_BOX and box_close:
                bearing = float(contacts.get("box_bearing", 0.0))
                if abs(bearing) > 0.08:
                    yaw_cmd = self._clip(-0.45 * bearing, -0.20, 0.20)
            return vx, vy, yaw_cmd

        if self._high_level_task == self.HIGH_LEVEL_TASK_TASK_D_SCRIPTED_PIT_PUSH:
            if self._pit_state == self.TASK_D_PIT_STATE_PUSH_BOX_TO_PATH_Y:
                if not contact_close:
                    vy = -min(abs(vy), 0.10)
                    if box_close:
                        bearing = float(contacts.get("box_bearing", 0.0))
                        yaw_cmd = self._clip(-0.30 * bearing, -0.14, 0.14)
                return vx, vy, yaw_cmd

            if self._pit_state == self.TASK_D_PIT_STATE_PUSH_BOX_TO_PIT_X:
                if front_distance >= self._taskd_lidar_lost_distance and not box_close:
                    vx = min(vx, 0.12)
                if box_close:
                    bearing = float(contacts.get("box_bearing", 0.0))
                    if abs(bearing) > 0.08:
                        yaw_cmd = self._clip(-0.45 * bearing, -0.18, 0.18)
                return vx, vy, yaw_cmd

            return vx, vy, yaw_cmd

        if self._high_level_task != self.HIGH_LEVEL_TASK_TASK_D_SCRIPTED_SIDE_PUSH:
            return vx, vy, yaw_cmd

        if self._side_state == self.TASK_D_SIDE_STATE_FORWARD_TO_BOX_SIDE:
            if min(front_distance, right_distance, box_distance) < self._taskd_lidar_approach_stop_distance:
                self._enter_taskd_side_state(self.TASK_D_SIDE_STATE_PUSH_BOX_TO_Y_NEG, current_score)
                return 0.0, -self._clip(self._side_lateral_vy, 0.05, 0.35), yaw_cmd

        if self._side_state == self.TASK_D_SIDE_STATE_PUSH_BOX_TO_Y_NEG:
            if not contact_close:
                if current_score < 15.0 and not box_close:
                    self._start_taskd_side_recovery(current_score)
                    return self._taskd_side_recovery_command(current_score)
                vy = -min(abs(vy), 0.10)
                if box_close:
                    bearing = float(contacts.get("box_bearing", 0.0))
                    yaw_cmd = self._clip(-0.35 * bearing, -0.16, 0.16)
            return vx, vy, yaw_cmd

        if self._side_state == self.TASK_D_SIDE_STATE_FINAL_PUSH_X:
            if current_score < 15.0 and front_distance >= self._taskd_lidar_lost_distance and not box_close:
                vx = min(vx, 0.12)
            if box_close:
                bearing = float(contacts.get("box_bearing", 0.0))
                if abs(bearing) > 0.08:
                    yaw_cmd = self._clip(-0.45 * bearing, -0.18, 0.18)

        return vx, vy, yaw_cmd

    def _enter_taskd_side_state(self, state: int, current_score: float):
        if self._side_state == state:
            return
        self._side_state = state
        self._side_state_enter_step = self._step_counter
        self._side_state_enter_score = current_score
        if state in (
            self.TASK_D_SIDE_STATE_PUSH_BOX_TO_Y_NEG,
            self.TASK_D_SIDE_STATE_FINAL_PUSH_X,
        ):
            self._side_score_check_step = self._step_counter
            self._side_score_check_score = current_score

    def _taskd_side_progress_stalled(self, current_score: float) -> bool:
        if current_score >= 15.0:
            return False
        if self._side_state not in (
            self.TASK_D_SIDE_STATE_PUSH_BOX_TO_Y_NEG,
            self.TASK_D_SIDE_STATE_FINAL_PUSH_X,
        ):
            return False
        if self._step_counter - self._side_score_check_step < self._side_score_check_interval:
            return False

        if current_score - self._side_score_check_score >= self._side_score_progress_eps:
            self._side_score_check_step = self._step_counter
            self._side_score_check_score = current_score
            return False
        return True

    def _start_taskd_side_recovery(self, current_score: float):
        self._side_recovery_attempt += 1
        self._side_recovery_start_step = self._step_counter
        self._enter_taskd_side_state(self.TASK_D_SIDE_STATE_RECOVER_BOX, current_score)

    def _taskd_side_recovery_command(self, current_score: float):
        elapsed = self._step_counter - self._side_recovery_start_step
        strafe_dir = -1.0 if self._side_recovery_attempt % 2 == 0 else 1.0

        if elapsed < 40:
            return 0.0, 0.0, 0.0
        if elapsed < 130:
            return -0.15, 0.0, 0.0
        if elapsed < 210:
            return 0.0, 0.10 * strafe_dir, self._clip(0.15 * strafe_dir, -0.15, 0.15)

        self._side_recovery_resume_until_step = self._step_counter + 300
        self._enter_taskd_side_state(self.TASK_D_SIDE_STATE_FORWARD_TO_BOX_SIDE, current_score)
        self._side_score_check_step = self._step_counter
        self._side_score_check_score = current_score
        return 0.18, 0.0, 0.0

    def _enter_taskf_state(self, state: int):
        if self._taskf_state != state:
            self._taskf_state = state
            self._taskf_state_enter_step = self._step_counter

    def _taskf_recovery_command(self):
        phase_idx = self._recovery_phase - 1
        r_vx, r_vy, r_hd, r_dur = self._recovery_sequence[phase_idx]
        if self._step_counter - self._recovery_phase_start >= r_dur:
            self._recovery_phase += 1
            self._recovery_phase_start = self._step_counter
            if self._recovery_phase > len(self._recovery_sequence):
                self._recovery_phase = 0
                self._stuck_check_score = self._prev_score
                self._stuck_check_step = self._step_counter
                self._enter_taskf_state(self.TASK_F_STATE_SEARCH_BOX)
        return r_vx, r_vy, r_hd

    def _compute_task_f_push_command(self, obs, current_score: float):
        if self._recovery_phase != 0:
            self._enter_taskf_state(self.TASK_F_STATE_RECOVERY)
            return self._taskf_recovery_command()

        if self._step_counter - self._stuck_check_step >= self._taskf_stuck_check_interval:
            d_score = current_score - self._stuck_check_score
            if (
                self._taskf_state in (self.TASK_F_STATE_ALIGN_PUSH, self.TASK_F_STATE_PUSH_BOX)
                and d_score < self._taskf_stuck_score_threshold
            ):
                self._recovery_phase = 1
                self._recovery_phase_start = self._step_counter
                self._enter_taskf_state(self.TASK_F_STATE_RECOVERY)
                return self._taskf_recovery_command()
            self._stuck_check_score = current_score
            self._stuck_check_step = self._step_counter

        lidar_info = self._parse_lidar(obs.get("extero"))
        seen = bool(lidar_info.get("box_detected"))
        if seen:
            self._taskf_last_seen_step = self._step_counter
        recently_seen = self._step_counter - self._taskf_last_seen_step <= self._taskf_lost_box_grace
        if not seen and not recently_seen:
            self._enter_taskf_state(self.TASK_F_STATE_SEARCH_BOX)
            # Rotate slowly while creeping forward, which keeps the box in LiDAR range when it appears.
            return 0.08, 0.0, 0.45

        bearing = float(lidar_info.get("box_bearing", 0.0))
        distance = float(lidar_info.get("box_distance", 999.0))
        abs_bearing = abs(bearing)

        if not seen:
            self._enter_taskf_state(self.TASK_F_STATE_SEARCH_BOX)
            return 0.05, 0.0, 0.35

        heading_cmd = self._clip(-1.25 * bearing, -0.75, 0.75)

        if distance > 1.55:
            self._enter_taskf_state(self.TASK_F_STATE_APPROACH_BOX)
            vx = self._clip(0.20 + 0.20 * (distance - 1.55), 0.25, 0.48)
            vy = self._clip(-0.18 * bearing, -0.12, 0.12)
            return vx, vy, heading_cmd

        if abs_bearing > 0.18 or distance < 0.65:
            self._enter_taskf_state(self.TASK_F_STATE_ALIGN_PUSH)
            vx = 0.12 if distance >= 0.65 else -0.10
            vy = self._clip(-0.35 * bearing, -0.18, 0.18)
            return vx, vy, heading_cmd

        self._enter_taskf_state(self.TASK_F_STATE_PUSH_BOX)
        vx = 0.65 if distance < 1.20 else 0.50
        vy = self._clip(-0.12 * bearing, -0.08, 0.08)
        hd = self._clip(-0.75 * bearing, -0.35, 0.35)
        return vx, vy, hd

    def _compute_task_d_auto_command(self, obs, current_score: float):
        state_steps = self._step_counter - self._state_enter_step
        if self._state == self.TASK_D_STATE_APPROACH_BOX and current_score >= 1.9:
            self._state = self.TASK_D_STATE_PUSH_BOX
            self._state_enter_step = self._step_counter
            state_steps = 0
        elif self._state == self.TASK_D_STATE_PUSH_BOX and (
            current_score >= 15.0 or (current_score >= 1.9 and state_steps >= 600)
        ):
            self._state = self.TASK_D_STATE_NAV_PLATFORM
            self._state_enter_step = self._step_counter
            state_steps = 0
        elif self._state == self.TASK_D_STATE_NAV_PLATFORM and (
            current_score >= 21.0 or (current_score >= 2.0 and state_steps >= 900)
        ):
            self._state = self.TASK_D_STATE_CLIMB_FINISH
            self._state_enter_step = self._step_counter

        vx, vy, hd = self._state_vel_cmds[self._state]

        if self._recovery_phase == 0:
            if self._step_counter - self._stuck_check_step >= self._stuck_check_interval:
                d_score = current_score - self._stuck_check_score
                if (
                    self._state in (self.TASK_D_STATE_PUSH_BOX, self.TASK_D_STATE_NAV_PLATFORM)
                    and d_score < self._stuck_score_threshold
                ):
                    self._recovery_phase = 1
                    self._recovery_phase_start = self._step_counter
                self._stuck_check_score = current_score
                self._stuck_check_step = self._step_counter
        else:
            phase_idx = self._recovery_phase - 1
            r_vx, r_vy, r_hd, r_dur = self._recovery_sequence[phase_idx]
            vx, vy, hd = r_vx, r_vy, r_hd
            if self._step_counter - self._recovery_phase_start >= r_dur:
                self._recovery_phase += 1
                self._recovery_phase_start = self._step_counter
                if self._recovery_phase > len(self._recovery_sequence):
                    self._recovery_phase = 0
                    self._stuck_check_score = current_score
                    self._stuck_check_step = self._step_counter
            return vx, vy, hd

        lidar_info = self._parse_lidar(obs.get("extero"))
        if (
            self._state in (self.TASK_D_STATE_APPROACH_BOX, self.TASK_D_STATE_PUSH_BOX)
            and lidar_info.get("box_detected")
            and lidar_info.get("box_distance", 999) < 3.0
        ):
            hd = self._clip(-float(lidar_info.get("box_bearing", 0.0)), -1.0, 1.0)
        return vx, vy, hd

    def _compute_task_d_scripted_path_command(self, obs, current_score: float):
        if current_score >= 21.0:
            self._script_state = self.TASK_D_SCRIPT_STATE_NAV_FINISH
            return 0.45, 0.0, 0.0
        if current_score >= 15.0:
            self._script_state = self.TASK_D_SCRIPT_STATE_NAV_FINISH
            return self._clip(self._script_nav_vx, 0.20, 0.45), 0.0, 0.0

        backup_end = self._script_backup_steps
        strafe_end = backup_end + self._script_strafe_steps
        settle_end = strafe_end + self._script_settle_steps

        if self._step_counter <= backup_end:
            self._script_state = self.TASK_D_SCRIPT_STATE_BACK_UP
            return -0.22, 0.0, 0.0

        if self._step_counter <= strafe_end:
            self._script_state = self.TASK_D_SCRIPT_STATE_STRAFE_TO_BOX_LINE
            return 0.0, self._clip(self._script_strafe_vy, 0.05, 0.35), 0.0

        if self._step_counter <= settle_end:
            self._script_state = self.TASK_D_SCRIPT_STATE_SETTLE
            return 0.0, 0.0, 0.0

        self._script_state = self.TASK_D_SCRIPT_STATE_PUSH_BOX
        vx = self._clip(self._script_push_vx, 0.20, 0.45)
        hd = 0.0
        lidar_info = self._parse_lidar(obs.get("extero"))
        if lidar_info.get("box_detected") and lidar_info.get("box_distance", 999.0) < 2.2:
            bearing = float(lidar_info.get("box_bearing", 0.0))
            if abs(bearing) > 0.08:
                hd = self._clip(-0.60 * bearing, -0.25, 0.25)
        return vx, 0.0, hd

    def _compute_task_d_scripted_side_push_command(self, obs, current_score: float):
        if current_score >= 21.0:
            self._enter_taskd_side_state(self.TASK_D_SIDE_STATE_FINAL_PUSH_X, current_score)
            return 0.45, 0.0, 0.0

        if current_score >= 15.0:
            self._enter_taskd_side_state(self.TASK_D_SIDE_STATE_FINAL_PUSH_X, current_score)
            vx = max(
                self._clip(self._side_final_push_vx, 0.20, 0.45),
                self._clip(self._script_nav_vx, 0.20, 0.45),
            )
            return vx, 0.0, 0.0

        if self._side_state == self.TASK_D_SIDE_STATE_RECOVER_BOX:
            return self._taskd_side_recovery_command(current_score)

        if self._taskd_side_progress_stalled(current_score):
            self._start_taskd_side_recovery(current_score)
            return self._taskd_side_recovery_command(current_score)

        if self._step_counter <= self._side_recovery_resume_until_step:
            if self._side_state == self.TASK_D_SIDE_STATE_PUSH_BOX_TO_Y_NEG:
                return 0.0, -self._clip(self._side_lateral_vy, 0.05, 0.35), 0.0
            self._enter_taskd_side_state(self.TASK_D_SIDE_STATE_FORWARD_TO_BOX_SIDE, current_score)
            return 0.18, 0.0, 0.0

        backup_end = self._side_backup_steps
        strafe_top_end = backup_end + self._side_strafe_top_steps
        forward_side_end = strafe_top_end + self._side_forward_to_box_steps
        push_y_end = forward_side_end + self._side_push_y_steps
        backup_behind_end = push_y_end + self._side_backup_behind_steps
        settle_end = backup_behind_end + self._script_settle_steps

        if self._step_counter <= backup_end:
            self._enter_taskd_side_state(self.TASK_D_SIDE_STATE_BACK_UP_LONG, current_score)
            return -0.25, 0.0, 0.0

        if self._step_counter <= strafe_top_end:
            self._enter_taskd_side_state(self.TASK_D_SIDE_STATE_STRAFE_TO_BOX_TOP, current_score)
            return 0.0, self._clip(self._side_lateral_vy, 0.05, 0.35), 0.0

        if self._step_counter <= forward_side_end:
            self._enter_taskd_side_state(self.TASK_D_SIDE_STATE_FORWARD_TO_BOX_SIDE, current_score)
            return 0.25, 0.0, 0.0

        if self._step_counter <= push_y_end and current_score < 15.0:
            self._enter_taskd_side_state(self.TASK_D_SIDE_STATE_PUSH_BOX_TO_Y_NEG, current_score)
            return 0.0, -self._clip(self._side_lateral_vy, 0.05, 0.35), 0.0

        if self._step_counter <= backup_behind_end and current_score < 15.0:
            self._enter_taskd_side_state(self.TASK_D_SIDE_STATE_BACK_UP_BEHIND_BOX, current_score)
            return -0.25, 0.0, 0.0

        if self._step_counter <= settle_end and current_score < 15.0:
            self._enter_taskd_side_state(self.TASK_D_SIDE_STATE_SETTLE, current_score)
            return 0.0, 0.0, 0.0

        self._enter_taskd_side_state(self.TASK_D_SIDE_STATE_FINAL_PUSH_X, current_score)
        vx = self._clip(self._side_final_push_vx, 0.20, 0.45)
        if current_score >= 15.0:
            vx = max(vx, self._clip(self._script_nav_vx, 0.20, 0.45))

        hd = 0.0
        lidar_info = self._parse_lidar(obs.get("extero"))
        if lidar_info.get("box_detected") and lidar_info.get("box_distance", 999.0) < 2.2:
            bearing = float(lidar_info.get("box_bearing", 0.0))
            if abs(bearing) > 0.08:
                hd = self._clip(-0.60 * bearing, -0.25, 0.25)
        return vx, 0.0, hd

    def _compute_task_d_scripted_pit_push_command(self, obs, current_score: float):
        backup_end, strafe_end, push_y_end, backup_behind_end, settle_end, push_x_end = self._taskd_pit_phase_bounds()

        if self._step_counter <= backup_end:
            self._pit_state = self.TASK_D_PIT_STATE_BACK_UP_TO_BOX_SIDE
            return -0.18, 0.0, 0.0

        if self._step_counter <= strafe_end:
            self._pit_state = self.TASK_D_PIT_STATE_STRAFE_TO_BOX_Y
            return 0.0, self._clip(self._pit_lateral_vy, 0.05, 0.35), 0.0

        if self._step_counter <= push_y_end:
            self._pit_state = self.TASK_D_PIT_STATE_PUSH_BOX_TO_PATH_Y
            return 0.0, -self._clip(self._pit_lateral_vy, 0.05, 0.35), 0.0

        if self._step_counter <= backup_behind_end:
            self._pit_state = self.TASK_D_PIT_STATE_MOVE_BEHIND_BOX
            return -0.18, 0.0, 0.0

        if self._step_counter <= settle_end:
            self._pit_state = self.TASK_D_PIT_STATE_SETTLE
            return 0.0, 0.0, 0.0

        if self._step_counter <= push_x_end:
            self._pit_state = self.TASK_D_PIT_STATE_PUSH_BOX_TO_PIT_X
            vx = self._clip(self._pit_push_vx, 0.20, 0.45)
            hd = 0.0
            lidar_info = self._parse_lidar(obs.get("extero"))
            if lidar_info.get("box_detected") and lidar_info.get("box_distance", 999.0) < 2.4:
                bearing = float(lidar_info.get("box_bearing", 0.0))
                if abs(bearing) > 0.08:
                    hd = self._clip(-0.45 * bearing, -0.18, 0.18)
            return vx, 0.0, hd

        self._pit_state = self.TASK_D_PIT_STATE_CROSS_PIT
        vx = 0.45 if current_score >= 21.0 else 0.38
        return vx, 0.0, 0.0

    def _compute_task_f_training_command(self, obs, current_score: float):
        policy_obs = obs["policy"].to(self.device)
        critic_obs = obs["critic"].to(self.device)
        if policy_obs.ndim == 1:
            policy_obs = policy_obs.unsqueeze(0)
        if critic_obs.ndim == 1:
            critic_obs = critic_obs.unsqueeze(0)

        detected = float(policy_obs[0, 58].item()) > 0.5
        bearing = self._clip(float(policy_obs[0, 59].item()), -1.0, 1.0)
        distance_norm = self._clip(float(policy_obs[0, 60].item()), 0.0, 1.0)
        gravity_xy = critic_obs[0, 6:8]
        tilt = float(torch.linalg.norm(gravity_xy).item())
        stable = self._step_counter > 40 and tilt < 0.65

        if detected:
            self._taskf_training_seen_steps += 1
        else:
            self._taskf_training_seen_steps = 0
            self._taskf_training_push_ready_steps = max(0, self._taskf_training_push_ready_steps - 1)

        distance_improved = (
            self._taskf_training_last_distance_norm is not None
            and distance_norm < self._taskf_training_last_distance_norm - 0.01
        )
        score_improved = current_score > self._prev_score + 0.02
        if detected and stable and (distance_improved or score_improved):
            self._taskf_training_push_ready_steps += 1
        elif not stable:
            self._taskf_training_push_ready_steps = 0

        self._taskf_training_last_distance_norm = distance_norm if detected else None

        if self._step_counter <= 80:
            self._enter_taskf_state(self.TASK_F_STATE_SEARCH_BOX)
            return 0.0, 0.0, 0.0

        if not stable:
            self._enter_taskf_state(self.TASK_F_STATE_RECOVERY)
            return 0.0, 0.0, 0.0

        if not detected:
            self._enter_taskf_state(self.TASK_F_STATE_SEARCH_BOX)
            return 0.02, 0.0, 0.0

        yaw_cmd = self._clip(-0.10 * bearing, -0.08, 0.08)
        if abs(bearing) > 0.25:
            self._enter_taskf_state(self.TASK_F_STATE_ALIGN_PUSH)
            return 0.01, 0.0, yaw_cmd

        if distance_norm > 0.35 or self._taskf_training_seen_steps < 20:
            self._enter_taskf_state(self.TASK_F_STATE_APPROACH_BOX)
            return 0.03, 0.0, yaw_cmd

        if self._taskf_training_push_ready_steps >= 30:
            self._enter_taskf_state(self.TASK_F_STATE_PUSH_BOX)
            return 0.08, 0.0, yaw_cmd

        self._enter_taskf_state(self.TASK_F_STATE_ALIGN_PUSH)
        return 0.04, 0.0, yaw_cmd

    def _predict_b2w_locomotion_56d_from_training_groups(self, obs, current_score: float):
        vx, vy, hd = self._compute_task_f_training_command(obs, current_score)
        policy_obs = obs["policy"].to(self.device)
        if policy_obs.ndim == 1:
            policy_obs = policy_obs.unsqueeze(0)
        vel_cmd = torch.tensor([[vx, vy, hd]], device=self.device, dtype=policy_obs.dtype)
        locomotion_obs = self._build_b2w_locomotion_56d_obs_from_training_groups(obs, vel_cmd)
        with torch.inference_mode():
            action_train = self.policy(locomotion_obs)
        if not isinstance(action_train, torch.Tensor):
            action_train = torch.as_tensor(action_train, device=self.device, dtype=torch.float32)
        action_train = torch.nan_to_num(
            action_train.to(device=self.device, dtype=torch.float32),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        if action_train.ndim == 1:
            action_train = action_train.unsqueeze(0)
        self._prev_score = current_score
        return {"action": action_train[:, :16].cpu().numpy().tolist(), "giveup": False}

    @staticmethod
    def _parse_lidar(extero):
        """Simple LiDAR box detection."""
        info = {"box_detected": False, "box_bearing": 0.0, "box_distance": 999}
        if extero is None:
            return info
        try:
            scan = extero.squeeze(0).cpu().numpy()
            if scan.size != 5760:  # 16*360
                return info
            scan_2d = scan.reshape(16, 360)
            mid = scan_2d[4:12, :]  # mid-elevation channels
            # Search right sector (rays 90-170) for box
            sector = mid[:, 90:171]
            valid = sector[(np.isfinite(sector)) & (sector > 0.2) & (sector < 5.0)]
            if valid.size >= 15:
                p10 = float(np.percentile(valid, 10))
                p30 = float(np.percentile(valid, 30))
                if p10 < 2.5 and (p30 - p10) < 0.6:
                    mask = (sector > 0.2) & (sector < 5.0)
                    w_rays = np.average(np.arange(90, 171), weights=mask.sum(axis=0).astype(float))
                    bearing = (float(w_rays) - 180.0) * np.pi / 180.0
                    info["box_detected"] = True
                    info["box_bearing"] = float(bearing)
                    info["box_distance"] = p10
                    return info

            # Fallback: scan wider horizontal sectors and pick the nearest compact cluster.
            best = None
            for start in range(35, 296, 10):
                end = start + 45
                sector = mid[:, start:end]
                valid = sector[(np.isfinite(sector)) & (sector > 0.25) & (sector < 4.0)]
                if valid.size < 18:
                    continue
                p10 = float(np.percentile(valid, 10))
                p35 = float(np.percentile(valid, 35))
                if p10 > 3.2 or (p35 - p10) > 0.75:
                    continue
                mask = (sector > 0.25) & (sector < min(4.0, p35 + 0.25))
                counts = mask.sum(axis=0).astype(float)
                if counts.sum() <= 0:
                    continue
                ray = float(np.average(np.arange(start, end), weights=counts))
                score = p10 - 0.003 * float(valid.size)
                if best is None or score < best[0]:
                    best = (score, ray, p10)
            if best is not None:
                _, ray, dist = best
                info["box_detected"] = True
                info["box_bearing"] = float((ray - 180.0) * np.pi / 180.0)
                info["box_distance"] = float(dist)
        except Exception:
            pass
        return info

    def predicts(self, obs, current_score):
        """Task D state machine + policy inference."""
        self._step_counter += 1
        if (
            self._policy_mode == self.POLICY_MODE_B2W_LOCOMOTION_56D
            and "policy" in obs
            and "critic" in obs
        ):
            return self._predict_b2w_locomotion_56d_from_training_groups(obs, current_score)

        proprio = obs["proprio"].to(self.device)
        action_dim = (int(proprio.shape[-1]) - 12) // 3

        # --- Fixed command mode: skip state machine, giveup, and recovery ---
        if self._taskd_command_mode != "auto":
            cmd_spec = self._fixed_cmds.get(self._taskd_command_mode)
            if cmd_spec is None:
                raise ValueError(
                    f"Unknown ATEC_TASKD_COMMAND_MODE={self._taskd_command_mode}. "
                    f"Valid: {list(self._fixed_cmds.keys())} + auto"
                )
            vx, vy, hd = cmd_spec
            vel_cmd = torch.tensor([[vx, vy, hd]], device=self.device, dtype=proprio.dtype)

            if self._policy_mode == self.POLICY_MODE_B2W_LOCOMOTION_56D:
                action_env = self._run_policy_with_command(obs, action_dim, vel_cmd)
                return {"action": action_env.cpu().numpy().tolist(), "giveup": False}

            # Run policy with the fixed velocity command
            if self._robot_type == "tron2awheel":
                policy_obs_8 = self._extract_policy_obs(obs, action_dim)
                n_head = 9
                head = policy_obs_8[:, :n_head]
                seg_len = self.leg_action_dim
                seg1 = policy_obs_8[:, n_head:n_head+seg_len]
                seg2 = policy_obs_8[:, n_head+seg_len:n_head+2*seg_len]
                seg3 = policy_obs_8[:, n_head+2*seg_len:]
                pad = torch.zeros((proprio.shape[0], 4), device=self.device, dtype=policy_obs_8.dtype)
                policy_obs = torch.cat([head, seg1, pad, seg2, pad, seg3, pad], dim=-1)
                with torch.inference_mode():
                    action_train_12 = self.policy(policy_obs)
                if not isinstance(action_train_12, torch.Tensor):
                    action_train_12 = torch.as_tensor(action_train_12, device=self.device, dtype=torch.float32)
                action_train_12 = action_train_12.to(device=self.device, dtype=torch.float32)
                if action_train_12.ndim == 1:
                    action_train_12 = action_train_12.unsqueeze(0)
                action_train = action_train_12[:, :self.leg_action_dim]
            else:
                if self._policy_mode == "b2w_taskd61":
                    policy_obs = self._extract_policy_obs_taskd61(obs, action_dim, current_score)
                else:
                    policy_obs = self._extract_policy_obs(obs, action_dim)
                policy_obs = torch.cat([policy_obs[:, :6], vel_cmd, policy_obs[:, 9:]], dim=-1)
                with torch.inference_mode():
                    action_train = self.policy(policy_obs)
                if not isinstance(action_train, torch.Tensor):
                    action_train = torch.as_tensor(action_train, device=self.device, dtype=torch.float32)
                action_train = action_train.to(device=self.device, dtype=torch.float32)
                if action_train.ndim == 1:
                    action_train = action_train.unsqueeze(0)

            action_env = self._map_policy_action_to_env_action(action_train, action_dim)
            action_env = action_env.cpu().numpy().tolist()
            return {"action": action_env, "giveup": False}

        if self._policy_mode == self.POLICY_MODE_B2W_LOCOMOTION_56D:
            total_time = self._step_counter * 0.02
            if total_time >= 1200 and current_score < 1.0:
                return {"action": [], "giveup": True}
            if self._high_level_task == self.HIGH_LEVEL_TASK_TASK_D_WAYPOINT_ROUTE:
                self._update_taskd_odometry(obs)

            if self._high_level_task == self.HIGH_LEVEL_TASK_TASK_D_AUTO:
                vx, vy, hd = self._compute_task_d_auto_command(obs, current_score)
            elif self._high_level_task == self.HIGH_LEVEL_TASK_TASK_D_SCRIPTED_PATH:
                vx, vy, hd = self._compute_task_d_scripted_path_command(obs, current_score)
            elif self._high_level_task == self.HIGH_LEVEL_TASK_TASK_D_SCRIPTED_SIDE_PUSH:
                vx, vy, hd = self._compute_task_d_scripted_side_push_command(obs, current_score)
            elif self._high_level_task == self.HIGH_LEVEL_TASK_TASK_D_SCRIPTED_PIT_PUSH:
                vx, vy, hd = self._compute_task_d_scripted_pit_push_command(obs, current_score)
            elif self._high_level_task == self.HIGH_LEVEL_TASK_TASK_D_TELEOP:
                vx, vy, hd = self._teleop_command
            elif self._high_level_task == self.HIGH_LEVEL_TASK_TASK_D_WAYPOINT_ROUTE:
                vx, vy, hd = self._compute_task_d_waypoint_route_command(obs, current_score)
            else:
                vx, vy, hd = self._compute_task_f_push_command(obs, current_score)
            if self._high_level_task in (
                self.HIGH_LEVEL_TASK_TASK_D_SCRIPTED_PATH,
                self.HIGH_LEVEL_TASK_TASK_D_SCRIPTED_SIDE_PUSH,
                self.HIGH_LEVEL_TASK_TASK_D_SCRIPTED_PIT_PUSH,
                self.HIGH_LEVEL_TASK_TASK_D_TELEOP,
                self.HIGH_LEVEL_TASK_TASK_D_WAYPOINT_ROUTE,
            ):
                vx, vy, hd = self._apply_taskd_lidar_contact_adjustment(obs, current_score, vx, vy, hd)
                vx, vy, hd = self._apply_taskd_backup_guard(obs, vx, vy, hd)
                vx, vy, hd = self._apply_taskd_speed_lock(obs, vx, vy, hd)
                vx, vy, hd = self._apply_taskd_heading_lock(obs, vx, vy, hd)
            vel_cmd = torch.tensor([[vx, vy, hd]], device=self.device, dtype=proprio.dtype)
            action_env = self._run_policy_with_command(obs, action_dim, vel_cmd)
            self._prev_score = current_score
            return {"action": action_env.cpu().numpy().tolist(), "giveup": False}

        # --- Auto mode: state machine + giveup + stuck recovery ---

        # State transitions (score-based)
        if self._state == 0 and current_score >= 1.9:
            self._state = 1; self._state_enter_step = self._step_counter
        elif self._state == 1 and current_score >= 15.0:
            self._state = 2; self._state_enter_step = self._step_counter
        elif self._state == 2 and current_score >= 21.0:
            self._state = 3; self._state_enter_step = self._step_counter
        self._prev_score = current_score

        # Giveup check
        total_time = self._step_counter * 0.02
        if (self._state <= 1 and total_time >= 240 and current_score < 15.0) or \
           (total_time >= 600 and current_score < 21.0):
            return {"action": [], "giveup": True}

        # State-dependent velocity command with LiDAR guidance
        vx, vy, hd = self._state_vel_cmds[self._state]

        # LiDAR box tracking: steer toward the box if visible
        extero = obs.get("extero")
        if extero is not None and self._robot_type != "tron2awheel":
            _li = self._parse_lidar(extero)
            if _li.get("box_detected") and _li.get("box_distance", 999) < 3.0:
                box_b = _li.get("box_bearing", 0.0)
                hd = float(np.clip(-box_b * 1.0, -1.0, 1.0))

        # --- Stuck recovery (score-based) ---
        # If the robot hasn't made meaningful score progress in the last ~10 s,
        # execute a short recovery sequence (backward → yaw → lateral).
        if self._recovery_phase == 0:
            if self._step_counter - self._stuck_check_step >= self._stuck_check_interval:
                d_score = current_score - self._stuck_check_score
                if d_score < self._stuck_score_threshold:
                    self._recovery_phase = 1
                    self._recovery_phase_start = self._step_counter
                self._stuck_check_score = current_score
                self._stuck_check_step = self._step_counter
        else:
            # In recovery: follow the sequence
            phase_idx = self._recovery_phase - 1
            r_vx, r_vy, r_hd, r_dur = self._recovery_sequence[phase_idx]
            vx, vy, hd = r_vx, r_vy, r_hd
            if self._step_counter - self._recovery_phase_start >= r_dur:
                self._recovery_phase += 1
                self._recovery_phase_start = self._step_counter
                if self._recovery_phase > len(self._recovery_sequence):
                    # Recovery complete — reset baseline and resume auto
                    self._recovery_phase = 0
                    self._stuck_check_score = current_score
                    self._stuck_check_step = self._step_counter

        vel_cmd = torch.tensor([[vx, vy, hd]], device=self.device, dtype=proprio.dtype)

        if self._robot_type == "tron2awheel":
            # Tron2AWheel: pad leg obs 8→12D, run policy, trim output 12→8D
            policy_obs_8 = self._extract_policy_obs(obs, action_dim)
            n_head = 9
            head = policy_obs_8[:, :n_head]
            seg_len = self.leg_action_dim
            seg1 = policy_obs_8[:, n_head:n_head+seg_len]
            seg2 = policy_obs_8[:, n_head+seg_len:n_head+2*seg_len]
            seg3 = policy_obs_8[:, n_head+2*seg_len:]
            pad = torch.zeros((proprio.shape[0], 4), device=self.device, dtype=policy_obs_8.dtype)
            policy_obs = torch.cat([head, seg1, pad, seg2, pad, seg3, pad], dim=-1)
            with torch.inference_mode():
                action_train_12 = self.policy(policy_obs)
            if not isinstance(action_train_12, torch.Tensor):
                action_train_12 = torch.as_tensor(action_train_12, device=self.device, dtype=torch.float32)
            action_train_12 = action_train_12.to(device=self.device, dtype=torch.float32)
            if action_train_12.ndim == 1:
                action_train_12 = action_train_12.unsqueeze(0)
            action_train = action_train_12[:, :self.leg_action_dim]
        else:
            # Replace env velocity_commands with our state-dependent one
            if self._policy_mode == "b2w_taskd61":
                policy_obs = self._extract_policy_obs_taskd61(obs, action_dim, current_score)
            else:
                policy_obs = self._extract_policy_obs(obs, action_dim)
            # Override velocity_commands segment (indices 6:9)
            policy_obs = torch.cat([policy_obs[:, :6], vel_cmd, policy_obs[:, 9:]], dim=-1)
            with torch.inference_mode():
                action_train = self.policy(policy_obs)
            if not isinstance(action_train, torch.Tensor):
                action_train = torch.as_tensor(action_train, device=self.device, dtype=torch.float32)
            action_train = action_train.to(device=self.device, dtype=torch.float32)
            if action_train.ndim == 1:
                action_train = action_train.unsqueeze(0)

        action_env = self._map_policy_action_to_env_action(action_train, action_dim)
        action_env = action_env.cpu().numpy().tolist()
        return {'action': action_env, 'giveup': False}
