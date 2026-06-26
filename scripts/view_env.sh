#!/usr/bin/env bash
set -eo pipefail

# ------------------------------------------------------------------
# Generic visualization/video wrapper for any ManagerBasedRLEnvCfg.
# Usage: ./scripts/view_env.sh --env_cfg <module_path>:<ClassName> [--video] [--num_envs N]
#
# Examples:
#   ./scripts/view_env.sh \\
#       --env_cfg atec_rl_lab.train.locomotion.velocity.config.quadruped.unitree_b2.task_f_flat_env_cfg:UnitreeB2WTaskFFlatEnvCfg \\
#       --video --video_length 300 --num_envs 1
#
#   ./scripts/view_env.sh \\
#       --env_cfg atec_rl_lab.tasks.task_a.env_cfg:TaskAEnvB2WCfg \\
#       --video --video_length 300 --num_envs 1
# ------------------------------------------------------------------

show_usage() {
    cat <<EOF
Usage: $(basename "$0") --env_cfg <module_path>:<ClassName> [--video] [--num_envs N]

Required:
  --env_cfg <module_path>:<ClassName>
                    Fully qualified env config class, e.g.:
                    atec_rl_lab.train.locomotion.velocity.config.quadruped.unitree_b2.task_f_flat_env_cfg:UnitreeB2WTaskFFlatEnvCfg

Optional:
  --num_envs <N>    Number of environments (default: 1)
  --video           Record a headless MP4 instead of opening the GUI
  --video_length N  Number of steps to record (default: 300)
  --video_output_dir <path>
                    Output directory (default: artifacts/view_env_videos)
  --video_name <name>
                    Optional MP4 filename

Recommended video command:
  $(basename "$0") \\
      --env_cfg atec_rl_lab.train.locomotion.velocity.config.quadruped.unitree_b2.task_f_flat_env_cfg:UnitreeB2WTaskFFlatEnvCfg \\
      --video --video_length 300 --num_envs 1

Note:
  In this workspace, the non-headless Isaac Sim GUI experience may fail because
  IsaacLab's GUI .kit dependencies do not match the installed /opt/IsaacSim
  extensions. Use --video for the stable headless rendering path.
EOF
    exit 1
}

if [[ $# -eq 0 ]]; then
    show_usage
fi

# Source environment and change to challenge root
source /home/1ctnltug/atec2026/scripts/env/activate_atec2026_sim.sh

# Forward all args to the Python visualization script
python /home/1ctnltug/atec2026/scripts/view_env.py "$@"
