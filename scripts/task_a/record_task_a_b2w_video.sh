#!/usr/bin/env bash
set -eo pipefail

cd /home/1ctnltug/atec2026/ATEC2026_Simulation_Challenge
source /home/1ctnltug/atec2026/scripts/env/activate_atec2026_sim.sh

config_file="/home/1ctnltug/atec2026/scripts/task_a/task_a_video_config.sh"
if [[ -f "${config_file}" ]]; then
  source "${config_file}"
fi

video_length="${ATEC_VIDEO_LENGTH:-600}"
camera_mode="${ATEC_CAMERA_MODE:-follow}"
record_video_dir="/home/1ctnltug/atec2026/ATEC2026_Simulation_Challenge/logs/videos/ATEC-TaskA-B2wPiper/play"
video_output_dir="${ATEC_VIDEO_OUTPUT_DIR:-/home/1ctnltug/atec2026/artifacts/task_a_videos}"
timestamp="$(date +%Y%m%d_%H%M%S)"
video_name="task_a_b2w_${timestamp}.mp4"
latest_link="/home/1ctnltug/atec2026/artifacts/latest_task_a_video.mp4"
fabric_args=()
if [[ "${ATEC_DISABLE_FABRIC:-0}" == "1" ]]; then
  fabric_args=(--disable_fabric)
fi

echo "Task A video config:"
echo "  video_length=${video_length} frames"
echo "  camera_mode=${camera_mode}"
echo "  disable_fabric=${ATEC_DISABLE_FABRIC:-0}"
echo "  output_dir=${video_output_dir}"
echo

mkdir -p "${video_output_dir}"

python scripts/play_atec_task.py \
  --task ATEC-TaskA-B2wPiper \
  --headless \
  --video \
  --video_length "${video_length}" \
  --camera_mode "${camera_mode}" \
  --enable_cameras \
  "${fabric_args[@]}" \
  --num_envs 1 \
  --debug

if [[ -f "${record_video_dir}/rl-video-step-0.mp4" ]]; then
  mv "${record_video_dir}/rl-video-step-0.mp4" "${video_output_dir}/${video_name}"
  ln -sfn "${video_output_dir}/${video_name}" "${latest_link}"
else
  echo "Warning: recorded video was not found at:"
  echo "${record_video_dir}/rl-video-step-0.mp4"
fi

echo
echo "Video output directory:"
echo "${video_output_dir}"
echo "Latest video:"
echo "${video_output_dir}/${video_name}"
echo "Latest video shortcut:"
echo "${latest_link}"
