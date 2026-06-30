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

    def _parse_taskd_lidar_contacts(self, extero):
        info = self._parse_lidar(extero)
        info.update({
            "front_distance": 999.0,
            "right_distance": 999.0,
            "left_distance": 999.0,
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
        except Exception:
            pass
        return info

    def _apply_taskd_lidar_contact_adjustment(self, obs, vx: float, vy: float, yaw_cmd: float):
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

        if self._high_level_task != self.HIGH_LEVEL_TASK_TASK_D_SCRIPTED_SIDE_PUSH:
            return vx, vy, yaw_cmd

        if self._side_state == self.TASK_D_SIDE_STATE_FORWARD_TO_BOX_SIDE:
            if min(front_distance, right_distance, box_distance) < self._taskd_lidar_approach_stop_distance:
                self._side_state = self.TASK_D_SIDE_STATE_PUSH_BOX_TO_Y_NEG
                return 0.0, -self._clip(self._side_lateral_vy, 0.05, 0.35), yaw_cmd

        if self._side_state == self.TASK_D_SIDE_STATE_PUSH_BOX_TO_Y_NEG:
            if not contact_close:
                vy = -min(abs(vy), 0.10)
                if box_close:
                    bearing = float(contacts.get("box_bearing", 0.0))
                    yaw_cmd = self._clip(-0.35 * bearing, -0.16, 0.16)
            return vx, vy, yaw_cmd

        if self._side_state == self.TASK_D_SIDE_STATE_FINAL_PUSH_X:
            if front_distance >= self._taskd_lidar_lost_distance and not box_close:
                vx = min(vx, 0.12)
            if box_close:
                bearing = float(contacts.get("box_bearing", 0.0))
                if abs(bearing) > 0.08:
                    yaw_cmd = self._clip(-0.45 * bearing, -0.18, 0.18)

        return vx, vy, yaw_cmd

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
            self._side_state = self.TASK_D_SIDE_STATE_FINAL_PUSH_X
            return 0.45, 0.0, 0.0

        backup_end = self._side_backup_steps
        strafe_top_end = backup_end + self._side_strafe_top_steps
        forward_side_end = strafe_top_end + self._side_forward_to_box_steps
        push_y_end = forward_side_end + self._side_push_y_steps
        backup_behind_end = push_y_end + self._side_backup_behind_steps
        settle_end = backup_behind_end + self._script_settle_steps

        if self._step_counter <= backup_end:
            self._side_state = self.TASK_D_SIDE_STATE_BACK_UP_LONG
            return -0.25, 0.0, 0.0

        if self._step_counter <= strafe_top_end:
            self._side_state = self.TASK_D_SIDE_STATE_STRAFE_TO_BOX_TOP
            return 0.0, self._clip(self._side_lateral_vy, 0.05, 0.35), 0.0

        if self._step_counter <= forward_side_end:
            self._side_state = self.TASK_D_SIDE_STATE_FORWARD_TO_BOX_SIDE
            return 0.25, 0.0, 0.0

        if self._step_counter <= push_y_end and current_score < 15.0:
            self._side_state = self.TASK_D_SIDE_STATE_PUSH_BOX_TO_Y_NEG
            return 0.0, -self._clip(self._side_lateral_vy, 0.05, 0.35), 0.0

        if self._step_counter <= backup_behind_end and current_score < 15.0:
            self._side_state = self.TASK_D_SIDE_STATE_BACK_UP_BEHIND_BOX
            return -0.25, 0.0, 0.0

        if self._step_counter <= settle_end and current_score < 15.0:
            self._side_state = self.TASK_D_SIDE_STATE_SETTLE
            return 0.0, 0.0, 0.0

        self._side_state = self.TASK_D_SIDE_STATE_FINAL_PUSH_X
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

            if self._high_level_task == self.HIGH_LEVEL_TASK_TASK_D_AUTO:
                vx, vy, hd = self._compute_task_d_auto_command(obs, current_score)
            elif self._high_level_task == self.HIGH_LEVEL_TASK_TASK_D_SCRIPTED_PATH:
                vx, vy, hd = self._compute_task_d_scripted_path_command(obs, current_score)
            elif self._high_level_task == self.HIGH_LEVEL_TASK_TASK_D_SCRIPTED_SIDE_PUSH:
                vx, vy, hd = self._compute_task_d_scripted_side_push_command(obs, current_score)
            else:
                vx, vy, hd = self._compute_task_f_push_command(obs, current_score)
            if self._high_level_task in (
                self.HIGH_LEVEL_TASK_TASK_D_SCRIPTED_PATH,
                self.HIGH_LEVEL_TASK_TASK_D_SCRIPTED_SIDE_PUSH,
            ):
                vx, vy, hd = self._apply_taskd_lidar_contact_adjustment(obs, vx, vy, hd)
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
