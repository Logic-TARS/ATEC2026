#!/usr/bin/env bash
set -eo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
challenge_root="${repo_root}/ATEC2026_Simulation_Challenge"

source "${repo_root}/scripts/env/activate_atec2026_sim.sh"
cd "${challenge_root}"

task_name="ATEC-TaskD-B2wPiper"
video_length="${ATEC_VIDEO_LENGTH:-1500}"
camera_mode="${ATEC_CAMERA_MODE:-follow}"
video_recorder="${ATEC_VIDEO_RECORDER:-manual}"
disable_fabric="${ATEC_DISABLE_FABRIC:-0}"
policy_path="${ATEC_POLICY_PATH:-${challenge_root}/demo/policy_taskd_omni.pt}"
policy_mode="${ATEC_POLICY_MODE:-b2w_omni16}"
video_output_dir="${ATEC_TASKD_VIDEO_OUTPUT_DIR:-${repo_root}/artifacts/task_d_videos}"
record_video_dir="${challenge_root}/logs/videos/${task_name}/play"
record_video_path="${record_video_dir}/rl-video-step-0.mp4"
timestamp="$(date +%Y%m%d_%H%M%S)"
video_name="task_d_b2w_${timestamp}.mp4"
latest_link="${repo_root}/artifacts/latest_task_d_video.mp4"

debug_args=()
if [[ "${ATEC_TASKD_DEBUG:-1}" != "0" ]]; then
  debug_args=(--debug)
fi

fabric_args=()
if [[ "${disable_fabric}" == "1" ]]; then
  fabric_args=(--disable_fabric)
fi

echo "Task D video config:"
echo "  task=${task_name}"
echo "  policy=${policy_path}"
echo "  policy_mode=${policy_mode}"
echo "  video_length=${video_length} frames"
echo "  camera_mode=${camera_mode}"
echo "  video_recorder=${video_recorder}"
echo "  disable_fabric=${disable_fabric}"
echo "  debug=${ATEC_TASKD_DEBUG:-1}"
echo "  output_dir=${video_output_dir}"
echo

if [[ ! -f "${policy_path}" ]]; then
  echo "ERROR: Omni policy not found at ${policy_path}"
  echo
  echo "The 16D omni policy has not been exported yet. Run:"
  echo "  ./scripts/training/export_latest_b2w_omni_policy_to_demo.sh"
  echo
  echo "If you haven't trained it yet, run the training first:"
  echo "  ./scripts/training/train_b2w_rough_omni_from_straight.sh"
  echo
  echo "To use a different policy, set ATEC_POLICY_PATH:"
  echo "  ATEC_POLICY_PATH=demo/policy.pt ATEC_POLICY_MODE=\"\" ./record_task_d_b2w_video.sh"
  exit 1
fi

mkdir -p "${video_output_dir}"
rm -f "${record_video_path}"

ATEC_POLICY_PATH="${policy_path}" \
ATEC_POLICY_MODE="${policy_mode}" \
python scripts/play_atec_task.py \
  --task "${task_name}" \
  --headless \
  --video \
  --video_length "${video_length}" \
  --video_recorder "${video_recorder}" \
  --camera_mode "${camera_mode}" \
  --enable_cameras \
  --num_envs 1 \
  "${fabric_args[@]}" \
  "${debug_args[@]}"

if [[ -f "${record_video_path}" ]]; then
  mv "${record_video_path}" "${video_output_dir}/${video_name}"
  ln -sfn "${video_output_dir}/${video_name}" "${latest_link}"
else
  echo "Warning: recorded video was not found at:"
  echo "${record_video_path}"
  echo
  echo "Troubleshooting:"
  echo "  - Confirm Isaac Sim completed without crashing."
  echo "  - Confirm --enable_cameras was accepted by AppLauncher."
  echo "  - Try a shorter run: ATEC_VIDEO_LENGTH=500 ./record_task_d_b2w_video.sh"
  exit 1
fi

echo
echo "Video output directory:"
echo "${video_output_dir}"
echo "Latest video:"
echo "${video_output_dir}/${video_name}"
echo "Latest video shortcut:"
echo "${latest_link}"
