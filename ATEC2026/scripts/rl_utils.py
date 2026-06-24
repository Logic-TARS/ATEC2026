import torch
import isaaclab.utils.math as math_utils

def camera_follow(env, robot_name: str = "robot", env_index: int = 0, alpha: float = 0.25):
    unwrapped = env.unwrapped

    if not hasattr(unwrapped, "viewport_camera_controller"):
        return

    try:
        robot = unwrapped.scene[robot_name]
    except KeyError as e:
        raise KeyError(
            f"Robot asset '{robot_name}' not found in env.unwrapped.scene."
        ) from e

    device = unwrapped.device

    robot_pos = robot.data.root_pos_w[env_index]
    robot_quat = robot.data.root_quat_w[env_index]

    camera_offset = torch.tensor([-6.0, 0.0, 2.2], dtype=torch.float32, device=device)
    lookat_offset = torch.tensor([1.8, 0.0, 0.7], dtype=torch.float32, device=device)

    target_camera_pos = math_utils.transform_points(
        camera_offset.unsqueeze(0),
        pos=robot_pos.unsqueeze(0),
        quat=robot_quat.unsqueeze(0),
    ).squeeze(0)
    target_lookat_pos = math_utils.transform_points(
        lookat_offset.unsqueeze(0),
        pos=robot_pos.unsqueeze(0),
        quat=robot_quat.unsqueeze(0),
    ).squeeze(0)

    target_camera_pos[2] = torch.clamp(target_camera_pos[2], min=1.0)
    target_lookat_pos[2] = torch.clamp(target_lookat_pos[2], min=0.7)

    if not hasattr(camera_follow, "_smooth_pos"):
        camera_follow._smooth_pos = {}
    if not hasattr(camera_follow, "_smooth_lookat"):
        camera_follow._smooth_lookat = {}

    if env_index not in camera_follow._smooth_pos:
        camera_follow._smooth_pos[env_index] = target_camera_pos.clone()
    if env_index not in camera_follow._smooth_lookat:
        camera_follow._smooth_lookat[env_index] = target_lookat_pos.clone()

    smooth_camera_pos = camera_follow._smooth_pos[env_index]
    smooth_camera_pos = (1.0 - alpha) * smooth_camera_pos + alpha * target_camera_pos
    camera_follow._smooth_pos[env_index] = smooth_camera_pos
    smooth_lookat_pos = camera_follow._smooth_lookat[env_index]
    smooth_lookat_pos = (1.0 - alpha) * smooth_lookat_pos + alpha * target_lookat_pos
    camera_follow._smooth_lookat[env_index] = smooth_lookat_pos

    unwrapped.viewport_camera_controller.set_view_env_index(env_index=env_index)
    unwrapped.viewport_camera_controller.update_view_location(
        eye=smooth_camera_pos.detach().cpu().numpy(),
        lookat=smooth_lookat_pos.detach().cpu().numpy(),
    )


def set_fixed_camera(env, env_index: int = 0):
    unwrapped = env.unwrapped

    if not hasattr(unwrapped, "viewport_camera_controller"):
        return

    unwrapped.viewport_camera_controller.set_view_env_index(env_index=env_index)
    unwrapped.viewport_camera_controller.update_view_location(
        eye=(-133.0, -26.0, 11.0),
        lookat=(-133.0, 0.0, 0.8),
    )
