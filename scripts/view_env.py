#!/usr/bin/env python3
"""
Generic visualization script for any ManagerBasedRLEnvCfg.

Usage:
    python scripts/view_env.py --env_cfg <module_path>:<ClassName> [--num_envs N]

Example:
    python scripts/view_env.py \\
        --env_cfg atec_rl_lab.train.locomotion.velocity.config.quadruped.unitree_b2.task_f_flat_env_cfg:UnitreeB2WTaskFFlatEnvCfg \\
        --num_envs 1
"""

import argparse
from datetime import datetime
import importlib
import itertools
from pathlib import Path

import torch

from isaaclab.app import AppLauncher

# ------------------------------------------------------------------
# Parse arguments
# ------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Visualize any ManagerBasedRLEnvCfg.")
parser.add_argument(
    "--env_cfg",
    type=str,
    required=True,
    help="Fully qualified env config class, e.g. "
    "atec_rl_lab.train.locomotion.velocity.config.quadruped.unitree_b2."
    "task_f_flat_env_cfg:UnitreeB2WTaskFFlatEnvCfg",
)
parser.add_argument(
    "--num_envs",
    type=int,
    default=1,
    help="Number of environments to spawn (default: 1).",
)
parser.add_argument(
    "--video",
    action="store_true",
    default=False,
    help="Record a headless MP4 instead of opening the GUI.",
)
parser.add_argument(
    "--video_length",
    type=int,
    default=300,
    help="Number of simulation steps to record when --video is set.",
)
parser.add_argument(
    "--video_output_dir",
    type=str,
    default=None,
    help="Directory for recorded MP4 files (default: artifacts/view_env_videos).",
)
parser.add_argument(
    "--video_name",
    type=str,
    default=None,
    help="Optional MP4 filename. Defaults to env class name plus timestamp.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.video:
    # The GUI experience is version-sensitive in this workspace. Video mode uses
    # the stable headless rendering path and still produces inspectable output.
    args_cli.headless = True
    args_cli.enable_cameras = True

# Launch Omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ------------------------------------------------------------------
# Imports that must come after AppLauncher
# ------------------------------------------------------------------
import imageio.v2 as imageio  # noqa: E402

from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnvCfg  # noqa: E402

# Import the atec_rl_lab.train package so custom envs are registered
import atec_rl_lab.train  # noqa: F401, E402


def import_env_cfg(env_cfg_str: str) -> ManagerBasedRLEnvCfg:
    """Dynamically import and instantiate an env config from 'module.path:ClassName'."""
    module_path, class_name = env_cfg_str.split(":", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls()


def default_video_path(env_cfg_str: str) -> Path:
    """Build a stable default output path under the workspace artifacts dir."""
    repo_root = Path(__file__).resolve().parents[1]
    output_dir = Path(args_cli.video_output_dir) if args_cli.video_output_dir else repo_root / "artifacts" / "view_env_videos"
    if not output_dir.is_absolute():
        output_dir = Path.cwd() / output_dir

    class_name = env_cfg_str.rsplit(":", 1)[-1]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = args_cli.video_name or f"{class_name}_{timestamp}.mp4"
    if not filename.endswith(".mp4"):
        filename = f"{filename}.mp4"
    return output_dir / filename


def main():
    env = None
    video_writer = None
    video_frames = 0
    video_path = default_video_path(args_cli.env_cfg) if args_cli.video else None
    try:
        # Instantiate env config
        env_cfg = import_env_cfg(args_cli.env_cfg)
        env_cfg.scene.num_envs = args_cli.num_envs

        print(f"Env cfg: {args_cli.env_cfg}")
        print(f"Num envs: {env_cfg.scene.num_envs}")
        if args_cli.video:
            print(f"Video length: {args_cli.video_length} steps")
            print(f"Video output: {video_path}")

        # Create environment
        env = ManagerBasedRLEnv(env_cfg, render_mode="rgb_array" if args_cli.video else None)

        # Print robot info for each articulation
        for name, articulation in env.scene.articulations.items():
            print("-" * 100)
            print("Robot name:", name)
            print("Bodies:", articulation.num_bodies, "->", articulation.body_names)
            print("Joints:", articulation.num_joints, "->", articulation.joint_names)
            articulation.set_joint_position_target(articulation.data.default_joint_pos)

        action_space = env.action_space
        obs, info = env.reset()

        if args_cli.video:
            video_path.parent.mkdir(parents=True, exist_ok=True)
            if video_path.exists():
                video_path.unlink()
            fps = int(round(1.0 / env.step_dt)) if env.step_dt > 0 else 50
            video_writer = imageio.get_writer(video_path, fps=fps, quality=7, macro_block_size=1)

        # Zero-action loop
        for step in itertools.count():
            if not simulation_app.is_running():
                break
            action = torch.zeros(action_space.shape, device=env.device)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated | truncated

            if args_cli.video:
                env.sim.render()
                frame = env.render(recompute=True)
                if frame is not None:
                    video_writer.append_data(frame)
                    video_frames += 1
                if step + 1 >= args_cli.video_length:
                    break

            if done.any():
                env_ids = done.nonzero(as_tuple=False).squeeze(-1)
                env.reset(env_ids=env_ids)

        if args_cli.video:
            print(f"Recorded frames: {video_frames}")
            if video_frames == 0:
                raise RuntimeError("No video frames were captured. Confirm --enable_cameras reached AppLauncher.")
            print(f"Saved video: {video_path}")
    finally:
        if video_writer is not None:
            video_writer.close()
        if env is not None:
            env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
