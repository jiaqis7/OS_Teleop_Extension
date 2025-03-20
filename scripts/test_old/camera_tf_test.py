# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run an environment with zero action agent."""

"""Launch Isaac Sim Simulator first."""

import argparse

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Zero agent for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-Reach-Dual-PSM-v0", help="Name of the task.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils import parse_env_cfg

import orbit.surgical.tasks  # noqa: F401

import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.spatial.transform import Rotation as R

# Enable interactive mode
plt.ion()

# Initialize empty figure windows for left and right cameras
fig_l, ax_l = plt.subplots(figsize=(5, 5), num="Left")
fig_r, ax_r = plt.subplots(figsize=(5, 5), num="Right")

# Set window titles
fig_l.canvas.manager.set_window_title("Left")
fig_r.canvas.manager.set_window_title("Right")

# Remove axes, margins, and whitespace
for ax in [ax_l, ax_r]:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)

# Set tight layout to remove white spaces
fig_l.tight_layout(pad=0)
fig_r.tight_layout(pad=0)

# Set margins to zero
fig_l.subplots_adjust(left=0, right=1, top=1, bottom=0)
fig_r.subplots_adjust(left=0, right=1, top=1, bottom=0)

# Dummy initial images (black)
img_l = ax_l.imshow(np.zeros((480, 640, 3), dtype=np.uint8))
img_r = ax_r.imshow(np.zeros((480, 640, 3), dtype=np.uint8))


def update_images(cam_l_input, cam_r_input):
    img_l.set_data(cam_l_input)
    img_r.set_data(cam_r_input)
    fig_l.canvas.flush_events()
    fig_r.canvas.flush_events()
    # plt.pause(0.001)


def pose_to_transformation_matrix(position, quaternion):
    """
    Convert position and quaternion to a 4x4 transformation matrix.

    Args:
        position (np.ndarray): Position array (x, y, z).
        quaternion (np.ndarray): Quaternion array (w, x, y, z).

    Returns:
        np.ndarray: 4x4 transformation matrix.
    """
    # Create a 4x4 identity matrix
    transformation_matrix = np.eye(4)

    # Set the translation part
    transformation_matrix[:3, 3] = position

    # Convert quaternion to rotation matrix
    rotation_matrix = R.from_quat([quaternion[1], quaternion[2], quaternion[3], quaternion[0]]).as_matrix()

    # Set the rotation part
    transformation_matrix[:3, :3] = rotation_matrix

    return transformation_matrix


def transformation_matrix_to_pose(transformation_matrix):
    """
    Convert a 4x4 transformation matrix to position and quaternion.

    Args:
        transformation_matrix (np.ndarray): 4x4 transformation matrix.

    Returns:
        tuple: Position array (x, y, z) and quaternion array (w, x, y, z).
    """
    # Extract the translation part
    position = transformation_matrix[:3, 3]

    # Extract the rotation part and convert to quaternion
    rotation_matrix = transformation_matrix[:3, :3]
    quaternion = R.from_matrix(rotation_matrix).as_quat()

    # Reorder quaternion to (w, x, y, z)
    quaternion = np.array([quaternion[3], quaternion[0], quaternion[1], quaternion[2]])

    return position, quaternion


def main():
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    # reset environment
    env.reset()

    psm1 = env.unwrapped.scene["robot_1"]
    psm2 = env.unwrapped.scene["robot_2"]

    camera_l = env.unwrapped.scene["camera_left"]
    camera_r = env.unwrapped.scene["camera_right"]

    psm1_cur_eef_pos = psm1.data.body_link_pos_w[0][-1].cpu().numpy()
    psm1_cur_eef_quat = psm1.data.body_link_quat_w[0][-1].cpu().numpy()
    world_T_psm1_tip = pose_to_transformation_matrix(psm1_cur_eef_pos, psm1_cur_eef_quat)
    psm2_cur_eef_pos = psm2.data.body_link_pos_w[0][-1].cpu().numpy()
    psm2_cur_eef_quat = psm2.data.body_link_quat_w[0][-1].cpu().numpy()
    world_T_psm2_tip = pose_to_transformation_matrix(psm2_cur_eef_pos, psm2_cur_eef_quat)

    psm1_base_link_pos = psm1.data.body_link_pos_w[0][0].cpu().numpy()
    psm1_base_link_quat = psm1.data.body_link_quat_w[0][0].cpu().numpy()
    world_T_psm1_base = pose_to_transformation_matrix(psm1_base_link_pos, psm1_base_link_quat)
    psm2_base_link_pos = psm2.data.body_link_pos_w[0][0].cpu().numpy()
    psm2_base_link_quat = psm2.data.body_link_quat_w[0][0].cpu().numpy()
    world_T_psm2_base = pose_to_transformation_matrix(psm2_base_link_pos, psm2_base_link_quat)

    psm1_base_T_psm1_tip = np.linalg.inv(world_T_psm1_base) @ world_T_psm1_tip
    psm2_base_T_psm2_tip = np.linalg.inv(world_T_psm2_base) @ world_T_psm2_tip

    psm1_rel_pos, psm1_rel_quat = transformation_matrix_to_pose(psm1_base_T_psm1_tip)
    psm2_rel_pos, psm2_rel_quat = transformation_matrix_to_pose(psm2_base_T_psm2_tip)

    while simulation_app.is_running():
        camera_l_pos = camera_l.data.pos_w
        camera_r_pos = camera_r.data.pos_w
        # get center of both cameras
        str_camera_pos = (camera_l_pos + camera_r_pos) / 2
        camera_quat = camera_l.data.quat_w_world  # forward x, up z
        world_T_camera = pose_to_transformation_matrix(str_camera_pos.cpu().numpy()[0], camera_quat.cpu().numpy()[0])

        # calculate psm1 eef pose relative to psm1 base link frame

        # concat psm1_init_pos, psm1_init_rot, psm2_init_pos, psm2_init_rot
        actions = np.concatenate([psm1_rel_pos, psm1_rel_quat, [False], psm2_rel_pos, psm2_rel_quat, [False]])
        # convert to torch
        actions = torch.tensor(actions, device=env.unwrapped.device).repeat(env.unwrapped.num_envs, 1)
        # actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)

        # apply actions
        env.step(actions)
        cam_l_input = camera_l.data.output["rgb"][0].cpu().numpy()
        cam_r_input = camera_r.data.output["rgb"][0].cpu().numpy()

        # Update displayed images
        update_images(cam_l_input, cam_r_input)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
