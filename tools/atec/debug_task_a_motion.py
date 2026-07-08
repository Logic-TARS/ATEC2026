import argparse
import os
import sys

# Ensure repo root is on sys.path so we can import submission/.
# The activation scripts add src/ to PYTHONPATH for atec_rl_lab.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Debug ATEC robot root/action motion state.")
parser.add_argument("--task", type=str, default="ATEC-TaskA-B2wPiper")
parser.add_argument("--num_steps", type=int, default=500)
parser.add_argument("--print_every", type=int, default=50)
parser.add_argument("--disable_fabric", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab_tasks.utils import parse_env_cfg

import atec_rl_lab.tasks  # noqa: F401
from submission.solution import AlgSolution


def main():
    print(f"[motion-debug] creating env task={args_cli.task}", flush=True)
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=1,
        use_fabric=not args_cli.disable_fabric,
    )
    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    print("[motion-debug] loading solution", flush=True)
    solution = AlgSolution()
    print("[motion-debug] resetting env", flush=True)
    obs, _ = env.reset()
    print("[motion-debug] reset complete", flush=True)
    robot = env.unwrapped.scene["robot"]

    action_dim = (int(obs["proprio"].shape[-1]) - 12) // 3
    start_x = robot.data.root_pos_w[0, 0].item()
    total_reward = 0.0

    print(f"action_dim={action_dim}", flush=True)
    print(f"start_root_x={start_x:.4f}", flush=True)
    print("step, elapsed_s, root_x, delta_x, base_lin_vel_x, obs_wheel_vel, action_wheel", flush=True)

    for step in range(args_cli.num_steps + 1):
        proprio = obs["proprio"]
        base_lin_vel_x = proprio[0, 0].item()
        joint_vel = proprio[0, 12 + action_dim:12 + 2 * action_dim]
        obs_wheel_vel = joint_vel[12:16].detach().cpu().tolist() if action_dim >= 24 else []

        resp = solution.predicts(obs, total_reward)
        action = torch.tensor(resp["action"], dtype=torch.float32, device=args_cli.device).view(1, -1)
        action_wheel = action[0, 12:16].detach().cpu().tolist() if action.shape[-1] >= 24 else []

        root_x = robot.data.root_pos_w[0, 0].item()
        elapsed = step * env.unwrapped.step_dt
        if step % args_cli.print_every == 0:
            print(
                f"{step}, {elapsed:.2f}, {root_x:.4f}, {root_x - start_x:.4f}, "
                f"{base_lin_vel_x:.4f}, {obs_wheel_vel}, {action_wheel}",
                flush=True,
            )

        obs, reward, terminated, truncated, info = env.step(action)
        sim_dt = info["Step_dt"]
        total_reward += reward.mean().item() / sim_dt

        if terminated.item() or truncated.item():
            print(f"done at step={step}, total_reward={total_reward:.4f}", flush=True)
            break

    final_x = robot.data.root_pos_w[0, 0].item()
    print(f"final_root_x={final_x:.4f}", flush=True)
    print(f"final_delta_x={final_x - start_x:.4f}", flush=True)
    print(f"total_reward={total_reward:.4f}", flush=True)
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
