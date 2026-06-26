#!/usr/bin/env bash
set -eo pipefail

# ------------------------------------------------------------------
# Generic policy playback/video wrapper for any registered Isaac Lab environment.
# Usage: ./scripts/play_env.sh --task <env_id> [extra_args...]
#
# Examples:
#   ./scripts/play_env.sh --task ATEC-Isaac-Velocity-Flat-TaskF-Unitree-B2W-Piper-v0 \
#       --num_envs 1 --checkpoint logs/rsl_rl/unitree_b2w_flat_omni/<run>/model_1000.pt
#
#   ./scripts/play_env.sh --task ATEC-Isaac-Velocity-Flat-Omni-B2W-Piper-v0 \
#       --num_envs 1 --load_run <run_name>
#
#   ./scripts/play_env.sh --task ATEC-Isaac-Velocity-Flat-TaskF-Unitree-B2W-Piper-v0 \
#       --num_envs 1 --load_run <run_name> --video --video_length 300
# ------------------------------------------------------------------

show_usage() {
    cat <<EOF
Usage: $(basename "$0") --task <env_id> [extra_args...]

Required:
  --task <env_id>   ID of the Isaac Lab environment to play
                    (e.g. ATEC-Isaac-Velocity-Flat-TaskF-Unitree-B2W-Piper-v0)

Common extra args (forwarded to scripts/rsl_rl/play.py):
  --num_envs <N>               Number of parallel environments (default 64)
  --load_run <path>            Experiment sub-directory to load
  --checkpoint <path>          Checkpoint path. If set, it is resolved directly.
  --headless                   Run without GUI
  --enable_cameras             Enable camera sensors
  --disable_fabric             Disable Fabric
  --video                      Record playback video
  --video_length <N>           Number of playback steps to record
  --video_output_dir <path>    Copy recorded MP4 here (default: artifacts/play_env_videos)
  --video_name <name>          Optional output MP4 filename
  --real-time                  Run in real time (adds sleep)

Examples:
  $(basename "$0") --task ATEC-Isaac-Velocity-Flat-TaskF-Unitree-B2W-Piper-v0 \\
      --num_envs 1 --checkpoint logs/rsl_rl/unitree_b2w_flat_omni/2025-06-01_12-00-00/model_1000.pt

  $(basename "$0") --task ATEC-Isaac-Velocity-Flat-TaskF-Unitree-B2W-Piper-v0 \\
      --num_envs 1 --load_run 2025-06-01_12-00-00

  $(basename "$0") --task ATEC-Isaac-Velocity-Flat-TaskF-Unitree-B2W-Piper-v0 \\
      --num_envs 1 --load_run 2025-06-01_12-00-00 --video --video_length 300

Note:
  In video mode this wrapper adds --headless to avoid the local GUI .kit
  dependency mismatch. Do not pass --disable_fabric for video recording;
  Fabric must stay enabled for frames to refresh.
EOF
    exit 1
}

# Require at least --task
if [[ $# -eq 0 ]]; then
    show_usage
fi

has_task=false
for arg in "$@"; do
    if [[ "$arg" == "--task" ]] || [[ "$arg" == --task=* ]]; then
        has_task=true
        break
    fi
done
if ! $has_task; then
    echo "ERROR: --task is required."
    echo
    show_usage
fi

repo_root="/home/1ctnltug/atec2026"
challenge_root="${repo_root}/ATEC2026"
video=false
has_headless=false
has_disable_fabric=false
help_requested=false
task_id=""
video_output_dir="${repo_root}/artifacts/play_env_videos"
video_name=""
forwarded_args=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --task)
            forwarded_args+=("$1")
            shift
            if [[ $# -gt 0 ]]; then
                task_id="$1"
                forwarded_args+=("$1")
                shift
            fi
            ;;
        --task=*)
            task_id="${1#--task=}"
            forwarded_args+=("$1")
            shift
            ;;
        --video)
            video=true
            forwarded_args+=("$1")
            shift
            ;;
        --headless)
            has_headless=true
            forwarded_args+=("$1")
            shift
            ;;
        --disable_fabric)
            has_disable_fabric=true
            forwarded_args+=("$1")
            shift
            ;;
        --video_output_dir)
            shift
            if [[ $# -eq 0 ]]; then
                echo "ERROR: --video_output_dir requires a value."
                exit 1
            fi
            video_output_dir="$1"
            shift
            ;;
        --video_output_dir=*)
            video_output_dir="${1#--video_output_dir=}"
            shift
            ;;
        --video_name)
            shift
            if [[ $# -eq 0 ]]; then
                echo "ERROR: --video_name requires a value."
                exit 1
            fi
            video_name="$1"
            shift
            ;;
        --video_name=*)
            video_name="${1#--video_name=}"
            shift
            ;;
        -h|--help)
            help_requested=true
            forwarded_args+=("$1")
            shift
            ;;
        *)
            forwarded_args+=("$1")
            shift
            ;;
    esac
done

if [[ "${video}" == true && "${has_headless}" == false ]]; then
    forwarded_args+=(--headless)
fi

if [[ "${video}" == true && "${has_disable_fabric}" == true ]]; then
    echo "WARNING: --disable_fabric was passed with --video."
    echo "         Video frames may be stale/frozen unless Fabric stays enabled."
fi

if [[ -n "${video_output_dir}" && "${video_output_dir}" != /* ]]; then
    video_output_dir="${repo_root}/${video_output_dir}"
fi

marker="$(mktemp)"
cleanup() {
    rm -f "${marker}"
}
trap cleanup EXIT

# Source environment and change to challenge root
source "${repo_root}/scripts/env/activate_atec2026_sim.sh"

# Forward all args to the play script
python scripts/rsl_rl/play.py "${forwarded_args[@]}"

if [[ "${video}" == true && "${help_requested}" == false ]]; then
    recorded_path="$(
        find "${challenge_root}/logs/rsl_rl" \
            -path "*/videos/play/rl-video-step-0.mp4" \
            -newer "${marker}" \
            -printf "%T@ %p\n" 2>/dev/null \
            | sort -nr \
            | head -1 \
            | cut -d' ' -f2-
    )"

    if [[ -z "${recorded_path}" || ! -f "${recorded_path}" ]]; then
        echo "ERROR: recorded video was not found under ${challenge_root}/logs/rsl_rl."
        echo "       Check the play.py output above for the run log directory."
        exit 1
    fi

    mkdir -p "${video_output_dir}"
    if [[ -z "${video_name}" ]]; then
        safe_task="${task_id:-play_env}"
        safe_task="${safe_task//[^A-Za-z0-9._-]/_}"
        timestamp="$(date +%Y%m%d_%H%M%S)"
        video_name="${safe_task}_${timestamp}.mp4"
    elif [[ "${video_name}" != *.mp4 ]]; then
        video_name="${video_name}.mp4"
    fi

    output_path="${video_output_dir}/${video_name}"
    cp "${recorded_path}" "${output_path}"

    echo
    echo "Recorded video source:"
    echo "${recorded_path}"
    echo "Copied video to:"
    echo "${output_path}"
fi
