import os
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

    def __init__(self):
        default_policy_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "policy.pt")
        policy_path = os.getenv("ATEC_POLICY_PATH", default_policy_path)
        self.device = 'cuda'

        self.policy = torch.jit.load(policy_path, map_location=self.device)
        self.policy.eval()

        # Policy mode — overrides robot-type-specific inference behaviours.
        self._policy_mode = os.getenv("ATEC_POLICY_MODE", "")

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
        if self._policy_mode in ("b2w_omni16", "b2w_taskd61"):
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
        self._state = 0  # 0=APPROACH, 1=PUSH_BOX, 2=NAV_PLATFORM, 3=CLIMB_FINISH
        self._state_enter_step = 0
        self._step_counter = 0
        self._prev_score = 0.0

        # Task D command mode for debugging policy omnidirectional capability.
        # Values: forward, backward, lateral_left, lateral_right, yaw_left,
        #         yaw_right, zero, auto (default state machine).
        self._taskd_command_mode = os.getenv("ATEC_TASKD_COMMAND_MODE", "auto")
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
            0: (0.25, 0.0, 0.0),   # APPROACH: slow forward
            1: (0.35, 0.0, 0.0),   # PUSH_BOX: push forward
            2: (0.45, 0.0, 0.0),   # NAV_PLATFORM
            3: (0.55, 0.0, 0.0),   # CLIMB_FINISH
        }

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

    def _map_policy_action_to_env_action(self, action_train: torch.Tensor, action_dim: int) -> torch.Tensor:
        """Map policy leg action to env full-body action."""
        num_envs = action_train.shape[0]

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
        except Exception:
            pass
        return info

    def predicts(self, obs, current_score):
        """Task D state machine + policy inference."""
        self._step_counter += 1
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
