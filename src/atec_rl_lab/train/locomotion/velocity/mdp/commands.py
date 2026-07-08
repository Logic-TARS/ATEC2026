# Reference: https://github.com/fan-ziqi/robot_lab

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass

import atec_rl_lab.train.locomotion.velocity.mdp as mdp

from .utils import is_robot_on_terrain

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class UniformThresholdVelocityCommand(mdp.UniformVelocityCommand):
    """Command generator that generates a velocity command in SE(2) from uniform distribution with threshold.

    This command generator automatically detects "pits" terrain and applies restrictions:
    - For pit terrains: only allow forward movement (no lateral or rotational movement)
    """

    cfg: mdp.UniformThresholdVelocityCommandCfg  # type: ignore
    """The configuration of the command generator."""

    def __init__(self, cfg: mdp.UniformThresholdVelocityCommandCfg, env: ManagerBasedEnv):
        """Initialize the command generator.

        Args:
            cfg: The configuration of the command generator.
            env: The environment.
        """
        super().__init__(cfg, env)
        # Track which robots were on pit terrain in the previous step
        self.was_on_pit = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def _resample_command(self, env_ids: Sequence[int]):
        """Resample velocity commands with threshold."""
        super()._resample_command(env_ids)
        # set small commands to zero
        self.vel_command_b[env_ids, :2] *= (torch.norm(self.vel_command_b[env_ids, :2], dim=1) > 0.2).unsqueeze(1)

    def _update_command(self):
        """Update commands and apply terrain-aware restrictions in real-time.

        This function:
        1. Calls parent's update to handle heading and standing envs
        2. Checks which robots are currently on pit terrain
        3. For robots leaving pits: resamples their commands
        4. For robots on pits: restricts to forward-only movement and sets heading to 0
        """
        # First, call parent's update command
        super()._update_command()

        # Check which robots are currently on pit terrain (real-time check every step)
        on_pits = is_robot_on_terrain(self._env, "pits")

        # Find robots that just left pit terrain (need to resample)
        left_pit_mask = self.was_on_pit & ~on_pits
        if left_pit_mask.any():
            left_pit_env_ids = torch.where(left_pit_mask)[0]
            # Resample commands for robots that left pits
            self._resample_command(left_pit_env_ids)

        # For robots currently on pits: restrict to forward-only movement with min/max speed
        if on_pits.any():
            pit_env_ids = torch.where(on_pits)[0]
            # Force forward-only movement with min and max speed limits
            self.vel_command_b[pit_env_ids, 0] = torch.clamp(
                torch.abs(self.vel_command_b[pit_env_ids, 0]), min=0.3, max=0.6
            )
            self.vel_command_b[pit_env_ids, 1] = 0.0  # no lateral movement
            self.vel_command_b[pit_env_ids, 2] = 0.0  # no yaw rotation
            # Set heading to 0 for pit robots
            if self.cfg.heading_command:
                self.heading_target[pit_env_ids] = 0.0

        # Update tracking state
        self.was_on_pit = on_pits


@configclass
class UniformThresholdVelocityCommandCfg(mdp.UniformVelocityCommandCfg):
    """Configuration for the uniform threshold velocity command generator."""

    class_type: type = UniformThresholdVelocityCommand


class DiscreteCommandController(CommandTerm):
    """
    Command generator that assigns discrete commands to environments.

    Commands are stored as a list of predefined integers.
    The controller maps these commands by their indices (e.g., index 0 -> 10, index 1 -> 20).
    """

    cfg: DiscreteCommandControllerCfg
    """Configuration for the command controller."""

    def __init__(self, cfg: DiscreteCommandControllerCfg, env: ManagerBasedEnv):
        """
        Initialize the command controller.

        Args:
            cfg: The configuration of the command controller.
            env: The environment object.
        """
        # Initialize the base class
        super().__init__(cfg, env)

        # Validate that available_commands is non-empty
        if not self.cfg.available_commands:
            raise ValueError("The available_commands list cannot be empty.")

        # Ensure all elements are integers
        if not all(isinstance(cmd, int) for cmd in self.cfg.available_commands):
            raise ValueError("All elements in available_commands must be integers.")

        # Store the available commands
        self.available_commands = self.cfg.available_commands

        # Create buffers to store the command
        # -- command buffer: stores discrete action indices for each environment
        self.command_buffer = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)

        # -- current_commands: stores a snapshot of the current commands (as integers)
        self.current_commands = [self.available_commands[0]] * self.num_envs  # Default to the first command

    def __str__(self) -> str:
        """Return a string representation of the command controller."""
        return (
            "DiscreteCommandController:\n"
            f"\tNumber of environments: {self.num_envs}\n"
            f"\tAvailable commands: {self.available_commands}\n"
        )

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """Return the current command buffer. Shape is (num_envs, 1)."""
        return self.command_buffer

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        """Update metrics for the command controller."""
        pass

    def _resample_command(self, env_ids: Sequence[int]):
        """Resample commands for the given environments."""
        sampled_indices = torch.randint(
            len(self.available_commands), (len(env_ids),), dtype=torch.int32, device=self.device
        )
        sampled_commands = torch.tensor(
            [self.available_commands[idx.item()] for idx in sampled_indices], dtype=torch.int32, device=self.device
        )
        self.command_buffer[env_ids] = sampled_commands

    def _update_command(self):
        """Update and store the current commands."""
        self.current_commands = self.command_buffer.tolist()


class ModeBasedVelocityCommand(UniformThresholdVelocityCommand):
    """Command generator that samples velocity commands from discrete locomotion modes.

    Five modes are sampled according to configurable probabilities:
      0: forward/backward  — vx ~ U[-1.2, 1.2], vy=0, wz=0
      1: lateral           — vx=0, vy ~ U[-1.0, 1.0], wz=0
      2: in-place yaw      — vx=0, vy=0, wz ~ U[-2.0, 2.0]
      3: mixed omni        — all three sampled from their full ranges
      4: standing          — all zero

    This generator forces ``heading_command=False`` so that the third command
    component is a yaw-rate target, not a heading-error target.
    """

    cfg: "ModeBasedVelocityCommandCfg"

    def __init__(self, cfg: "ModeBasedVelocityCommandCfg", env: "ManagerBasedEnv"):
        # Force heading_command off — yaw is direct angular velocity
        cfg.heading_command = False

        # Validate mode_probs
        probs = cfg.mode_probs
        if len(probs) != 5:
            raise ValueError(
                f"mode_probs must have exactly 5 elements, got {len(probs)}: {probs}"
            )
        if any(p < 0.0 for p in probs):
            raise ValueError(f"mode_probs contains negative values: {probs}")
        if sum(probs) <= 0.0:
            raise ValueError(f"mode_probs sum must be positive, got {sum(probs)}: {probs}")

        super().__init__(cfg, env)
        # Build normalized mode probability tensor once
        self._mode_probs = torch.tensor(probs, dtype=torch.float, device=self.device)
        self._mode_probs = self._mode_probs / self._mode_probs.sum()

    def _resample_command(self, env_ids: "Sequence[int]"):
        n = len(env_ids)
        # 1. Sample mode for each env
        mode = torch.distributions.Categorical(self._mode_probs).sample((n,))

        # 2. Sample velocity components for each env (batch over env_ids)
        vx = torch.zeros(n, device=self.device)
        vy = torch.zeros(n, device=self.device)
        wz = torch.zeros(n, device=self.device)

        ranges = self.cfg.ranges

        # Mode 0: forward/backward
        mask0 = mode == 0
        if mask0.any():
            k = mask0.sum().item()
            vx[mask0] = (ranges.lin_vel_x[0] + (ranges.lin_vel_x[1] - ranges.lin_vel_x[0]) * torch.rand(k, device=self.device))
            vy[mask0] = 0.0
            wz[mask0] = 0.0

        # Mode 1: lateral
        mask1 = mode == 1
        if mask1.any():
            k = mask1.sum().item()
            vx[mask1] = 0.0
            vy[mask1] = (ranges.lin_vel_y[0] + (ranges.lin_vel_y[1] - ranges.lin_vel_y[0]) * torch.rand(k, device=self.device))
            wz[mask1] = 0.0

        # Mode 2: in-place yaw
        mask2 = mode == 2
        if mask2.any():
            k = mask2.sum().item()
            vx[mask2] = 0.0
            vy[mask2] = 0.0
            wz[mask2] = (ranges.ang_vel_z[0] + (ranges.ang_vel_z[1] - ranges.ang_vel_z[0]) * torch.rand(k, device=self.device))

        # Mode 3: mixed omni (sample all three)
        mask3 = mode == 3
        if mask3.any():
            k = mask3.sum().item()
            vx[mask3] = (ranges.lin_vel_x[0] + (ranges.lin_vel_x[1] - ranges.lin_vel_x[0]) * torch.rand(k, device=self.device))
            vy[mask3] = (ranges.lin_vel_y[0] + (ranges.lin_vel_y[1] - ranges.lin_vel_y[0]) * torch.rand(k, device=self.device))
            wz[mask3] = (ranges.ang_vel_z[0] + (ranges.ang_vel_z[1] - ranges.ang_vel_z[0]) * torch.rand(k, device=self.device))

        # Mode 4: standing (explicitly zeroed by torch.zeros init above)

        # 3. Write into command buffer
        self.vel_command_b[env_ids, 0] = vx
        self.vel_command_b[env_ids, 1] = vy
        self.vel_command_b[env_ids, 2] = wz

        # 4. Threshold small commands (inherited behaviour)
        self.vel_command_b[env_ids, :2] *= (
            torch.norm(self.vel_command_b[env_ids, :2], dim=1) > 0.2
        ).unsqueeze(1)


@configclass
class ModeBasedVelocityCommandCfg(UniformThresholdVelocityCommandCfg):
    """Configuration for the mode-based velocity command generator."""

    class_type: type = ModeBasedVelocityCommand

    mode_probs: tuple[float, float, float, float, float] = (0.25, 0.25, 0.25, 0.20, 0.05)
    """Probabilities for the five modes: (fwd_back, lateral, yaw, mixed_omni, standing)."""


@configclass
class DiscreteCommandControllerCfg(CommandTermCfg):
    """Configuration for the discrete command controller."""

    class_type: type = DiscreteCommandController

    available_commands: list[int] = []
    """
    List of available discrete commands, where each element is an integer.
    Example: [10, 20, 30, 40, 50]
    """
