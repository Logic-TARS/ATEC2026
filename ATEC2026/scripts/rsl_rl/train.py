
"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--actor_checkpoint",
    type=str,
    default=None,
    help="Optional checkpoint path used to initialize only compatible actor weights.",
)
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
parser.add_argument(
    "--ray-proc-id", "-rid", type=int, default=None, help="Automatically configured by Ray integration, otherwise None."
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for minimum supported RSL-RL version."""

import importlib.metadata as metadata
import platform

from packaging import version

# check minimum supported rsl-rl version
RSL_RL_VERSION = "3.0.1"
installed_version = metadata.version("rsl-rl-lib")
if version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    if platform.system() == "Windows":
        cmd = [r".\isaaclab.bat", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    else:
        cmd = ["./isaaclab.sh", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    print(
        f"Please install the correct version of RSL-RL.\nExisting version is: '{installed_version}'"
        f" and required version is: '{RSL_RL_VERSION}'.\nTo install the correct version, run:"
        f"\n\n\t{' '.join(cmd)}\n"
    )
    exit(1)

"""Rest everything follows."""

import logging
import os
import time
from datetime import datetime

import gymnasium as gym
import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import atec_rl_lab.train  # noqa: F401  # isort: skip

# import logger
logger = logging.getLogger(__name__)

# PLACEHOLDER: Extension template (do not remove this comment)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def _segmented_col_transfer_45_53(source: "torch.Tensor", target: "torch.Tensor") -> "torch.Tensor":
    """Map B2 Rough-Straight 45-D observation columns to B2W Omni 53-D columns.

    Source (45)                    Target (53)
    [0:3]   base_ang_vel*0.25  →  [0:3]   base_ang_vel*0.25     copy
    [3:6]   projected_gravity  →  [3:6]   projected_gravity     copy
    [6:9]   velocity_commands  →  [6:9]   velocity_commands     copy
    [9:21]  leg_joint_pos      →  [9:21]  leg_joint_pos         copy
    [21:33] leg_joint_vel      →  [21:33] leg_joint_vel         copy
                                 [33:37] wheel_vel*0.05          init random
    [33:45] prev_leg_actions   →  [37:49] prev_leg_actions      copy
                                 [49:53] prev_wheel_actions      init random
    """
    merged = target.clone()
    # Block 1: head through leg_joint_vel (cols 0-33 are identical)
    merged[:, :33] = source[:, :33]
    # Block 2: wheel_vel (cols 33-37) — small random init
    merged[:, 33:37] = torch.randn(
        source.shape[0], 4, device=merged.device, dtype=merged.dtype
    ) * 0.01
    # Block 3: prev_leg_actions (source cols 33-45 → target cols 37-49)
    merged[:, 37:49] = source[:, 33:45]
    # Block 4: prev_wheel_actions (cols 49-53) — small random init
    merged[:, 49:53] = torch.randn(
        source.shape[0], 4, device=merged.device, dtype=merged.dtype
    ) * 0.01
    return merged


def _segmented_col_transfer_53_61(source: "torch.Tensor", target: "torch.Tensor") -> "torch.Tensor":
    """Map B2W Omni 53-D observation columns to Task D 61-D columns.

    Source (53)                    Target (61)
    [0:53]  omni obs (identical)  →  [0:53]  copy all
                                 →  [53:54] score_norm          init random
                                 →  [54:58] stage_one_hot       init random
                                 →  [58:59] box_detected        init random
                                 →  [59:60] box_bearing         init random
                                 →  [60:61] box_distance_norm   init random
    """
    merged = target.clone()
    merged[:, :53] = source[:, :53]
    merged[:, 53:61] = torch.randn(
        source.shape[0], 8, device=merged.device, dtype=merged.dtype
    ) * 0.01
    return merged


def _load_actor_weights_only(runner, checkpoint_path: str):
    """Initialize the runner policy actor from a checkpoint while leaving critic/optimizer fresh.

    Handles three categories of tensors:
    - exact match: same shape → loaded directly
    - output-expanded: more output rows (12D→16D) → copy old rows, init new with N(0, 1e-2)
    - input-expanded: more input columns (45D→53D) → segmented column transfer for the
      known B2→B2W observation layout change
    - skipped: any other shape mismatch
    """
    checkpoint_path = os.path.abspath(checkpoint_path)
    print(f"[INFO]: Loading actor weights from checkpoint: {checkpoint_path}")
    loaded_dict = torch.load(checkpoint_path, map_location=runner.device, weights_only=False)
    source_state = loaded_dict["model_state_dict"]

    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        policy_nn = runner.alg.actor_critic

    target_state = policy_nn.state_dict()
    actor_state = {}
    exact_matches = []
    output_expanded = []
    input_expanded = []
    skipped = []

    for name, source_tensor in source_state.items():
        if not name.startswith("actor."):
            continue
        if name not in target_state:
            skipped.append(name)
            continue

        target_shape = target_state[name].shape
        source_shape = source_tensor.shape

        # --- Exact match ---
        if target_shape == source_shape:
            actor_state[name] = source_tensor
            exact_matches.append(name)
            continue

        # --- Output expansion: same hidden dim, more output rows (e.g. 12→16) ---
        if (
            source_tensor.ndim == 2
            and len(target_shape) == 2
            and target_shape[1] == source_shape[1]
            and target_shape[0] > source_shape[0]
        ):
            merged = target_state[name].clone()
            merged[: source_shape[0]] = source_tensor
            new_rows = target_shape[0] - source_shape[0]
            merged[source_shape[0] :] = torch.randn(
                new_rows, target_shape[1], device=merged.device, dtype=merged.dtype
            ) * 0.01
            actor_state[name] = merged
            output_expanded.append(f"{name}  [{source_shape} → {target_shape}]")
            continue

        # --- Output expansion (1-D bias / log_std): same logic ---
        if (
            source_tensor.ndim == 1
            and len(target_shape) == 1
            and target_shape[0] > source_shape[0]
        ):
            merged = target_state[name].clone()
            merged[: source_shape[0]] = source_tensor
            new_elems = target_shape[0] - source_shape[0]
            merged[source_shape[0] :] = torch.randn(
                new_elems, device=merged.device, dtype=merged.dtype
            ) * 0.01
            actor_state[name] = merged
            output_expanded.append(f"{name}  [{source_shape} → {target_shape}]")
            continue

        # --- Input expansion: same output dim, more input columns ---
        # This handles the B2 45D → B2W 53D observation dimension change for
        # the first-layer weight tensor (actor.0.weight).
        if (
            source_tensor.ndim == 2
            and len(target_shape) == 2
            and target_shape[0] == source_shape[0]
            and target_shape[1] > source_shape[1]
        ):
            # Check for the known 45→53 mapping.
            if source_shape[1] == 45 and target_shape[1] == 53:
                actor_state[name] = _segmented_col_transfer_45_53(
                    source_tensor, target_state[name]
                )
                input_expanded.append(
                    f"{name}  [{source_shape} → {target_shape}]  (45D→53D segmented)"
                )
            # Check for the known 53→61 mapping (Task D fine-tuning).
            elif source_shape[1] == 53 and target_shape[1] == 61:
                actor_state[name] = _segmented_col_transfer_53_61(
                    source_tensor, target_state[name]
                )
                input_expanded.append(
                    f"{name}  [{source_shape} → {target_shape}]  (53D→61D segmented)"
                )
            else:
                # Generic fallback: copy old columns, random-init the new suffix.
                merged = target_state[name].clone()
                merged[:, : source_shape[1]] = source_tensor
                merged[:, source_shape[1] :] = torch.randn(
                    target_shape[0],
                    target_shape[1] - source_shape[1],
                    device=merged.device,
                    dtype=merged.dtype,
                ) * 0.01
                actor_state[name] = merged
                input_expanded.append(
                    f"{name}  [{source_shape} → {target_shape}]  (generic suffix)"
                )
            continue

        # --- Unhandled mismatch ---
        skipped.append(name)

    # Logging
    if exact_matches:
        print(f"[INFO]: Exact match: {len(exact_matches)} tensor(s)")
    if output_expanded:
        print(f"[INFO]: Output-expanded: {len(output_expanded)} tensor(s) —")
        for msg in output_expanded:
            print(f"        {msg}")
    if input_expanded:
        print(f"[INFO]: Input-expanded: {len(input_expanded)} tensor(s) —")
        for msg in input_expanded:
            print(f"        {msg}")
    if skipped:
        print(f"[INFO]: Skipped: {len(skipped)} tensor(s) — {skipped}")

    if not actor_state:
        raise ValueError(f"No compatible actor weights found in checkpoint: {checkpoint_path}")

    target_state.update(actor_state)
    policy_nn.load_state_dict(target_state)
    print(
        f"[INFO]: Loaded {len(actor_state)} actor tensors "
        f"(exact: {len(exact_matches)}, output-expanded: {len(output_expanded)}, "
        f"input-expanded: {len(input_expanded)}, skipped: {len(skipped)})."
    )


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.sim.use_fabric = not args_cli.disable_fabric
    # check for invalid combination of CPU device with distributed training
    if args_cli.distributed and args_cli.device is not None and "cpu" in args_cli.device:
        raise ValueError(
            "Distributed training is not supported when using CPU device. "
            "Please use GPU device (e.g., --device cuda) for distributed training."
        )

    # multi-gpu training configuration
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # set seed to have diversity in different threads
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # The Ray Tune workflow extracts experiment name using the logging line below, hence, do not
    # change it (see PR #2346, comment-2819298849)
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # set the IO descriptors export flag if requested
    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
    else:
        logger.warning(
            "IO descriptors are only supported for manager based RL environments. No IO descriptors will be exported."
        )

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # save resume path before creating a new log_dir
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    start_time = time.time()

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # create runner from rsl-rl
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    if args_cli.actor_checkpoint is not None:
        _load_actor_weights_only(runner, args_cli.actor_checkpoint)
    # load the checkpoint
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    print(f"Training time: {round(time.time() - start_time, 2)} seconds")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
