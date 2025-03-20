# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run a keyboard teleoperation with Isaac Lab manipulation environments."""

"""Launch Isaac Sim Simulator first."""

"""python source/standalone/environments/teleoperation/teleop_phantomomni.py --task Isaac-Reach-Dual-PSM-IK-Abs-v0 --num_envs 1 --teleop_device po"""
import argparse

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Keyboard teleoperation for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-Reach-Dual-PSM-IK-Rel-v0", help="Name of the task.")
parser.add_argument("--sensitivity", type=float, default=1.0, help="Sensitivity factor.")
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

import carb

from omni.isaac.lab.devices import Se3Gamepad, Se3Keyboard, Se3SpaceMouse
import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils import parse_env_cfg

from orbit.surgical.ext.devices.MTM import MTMTeleop
import orbit.surgical.tasks  # noqa: F401
import matplotlib.pyplot as plt
import cv2
from scipy.spatial.transform import Rotation as R
import numpy as np
import time


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

# map mtm gripper joint angle to psm jaw gripper angles in simulation
def get_jaw_gripper_angles(gripper_command):
    # input: -1.72 (closed), 1.06 (opened)
    # output: 0,0 (closed), -0.52359, 0.52359 (opened)
    gripper2_angle = 0.52359 / (1.06 + 1.72) * (gripper_command + 1.72)
    return np.array([-gripper2_angle, gripper2_angle])

# process cam_T_psmtip to psmbase_T_psmtip and make usuable action input
def process_actions(cam_T_psm1, w_T_psm1base, cam_T_psm2, w_T_psm2base, w_T_cam, env, gripper1_command, gripper2_command) -> torch.Tensor:
    """Process actions for the environment."""
    psm1base_T_psm1 = np.linalg.inv(w_T_psm1base)@w_T_cam@cam_T_psm1
    psm2base_T_psm2 = np.linalg.inv(w_T_psm2base)@w_T_cam@cam_T_psm2
    psm1_rel_pos, psm1_rel_quat = transformation_matrix_to_pose(psm1base_T_psm1)
    psm2_rel_pos, psm2_rel_quat = transformation_matrix_to_pose(psm2base_T_psm2)
    actions = np.concatenate([psm1_rel_pos, psm1_rel_quat, get_jaw_gripper_angles(gripper1_command), psm2_rel_pos, psm2_rel_quat, get_jaw_gripper_angles(gripper2_command),])
    actions = torch.tensor(actions, device=env.unwrapped.device).repeat(env.unwrapped.num_envs, 1)
    return actions


def main():
    """Running keyboard teleoperation with Isaac Lab manipulation environment."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    # modify configuration
    env_cfg.terminations.time_out = None

    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()

    teleop_interface = MTMTeleop(pos_sensitivity=1.0, rot_sensitivity=1.0)
    teleop_interface.reset()

    camera_l = env.unwrapped.scene["camera_left"]
    camera_r = env.unwrapped.scene["camera_right"]

    psm1 = env.unwrapped.scene["robot_1"]
    psm2 = env.unwrapped.scene["robot_2"]

    was_in_clutch = True
    init_mtml_position = None
    init_psm1_tip_position = None
    init_mtmr_position = None
    init_psm2_tip_position = None

    print("Press the clutch button and release to start teleoperation.")
    # simulate environment
    while simulation_app.is_running():
        # process actions
        camera_l_pos = camera_l.data.pos_w
        camera_r_pos = camera_r.data.pos_w
        # get center of both cameras
        str_camera_pos = (camera_l_pos + camera_r_pos) / 2
        camera_quat = camera_l.data.quat_w_world  # forward x, up z
        world_T_cam = pose_to_transformation_matrix(str_camera_pos.cpu().numpy()[0], camera_quat.cpu().numpy()[0])
        cam_T_world = np.linalg.inv(world_T_cam)

        psm1_base_link_pos = psm1.data.body_link_pos_w[0][0].cpu().numpy()
        psm1_base_link_quat = psm1.data.body_link_quat_w[0][0].cpu().numpy()
        world_T_psm1_base = pose_to_transformation_matrix(psm1_base_link_pos, psm1_base_link_quat)

        psm2_base_link_pos = psm2.data.body_link_pos_w[0][0].cpu().numpy()
        psm2_base_link_quat = psm2.data.body_link_quat_w[0][0].cpu().numpy()
        world_T_psm2_base = pose_to_transformation_matrix(psm2_base_link_pos, psm2_base_link_quat)

        # get target pos, rot in camera view with joint and clutch commands
        mtml_pos, mtml_rot, l_gripper_joint, mtmr_pos, mtmr_rot, r_gripper_joint, clutch = teleop_interface.advance()
        if not l_gripper_joint:
            time.sleep(0.05)
            continue
        mtml_orientation = R.from_rotvec(mtml_rot).as_quat()
        mtml_orientation = np.concatenate([[mtml_orientation[3]], mtml_orientation[:3]])
        mtmr_orientation = R.from_rotvec(mtmr_rot).as_quat()
        mtmr_orientation = np.concatenate([[mtmr_orientation[3]], mtmr_orientation[:3]])

        if not clutch:
            if was_in_clutch:
                print("Released from clutch. Starting teleoperation again")
                init_mtml_position = mtml_pos
                psm1_tip_pose_w = psm1.data.body_link_pos_w[0][-1].cpu().numpy()
                psm1_tip_quat_w = psm1.data.body_link_quat_w[0][-1].cpu().numpy()
                world_T_psm1tip = pose_to_transformation_matrix(psm1_tip_pose_w, psm1_tip_quat_w)
                init_cam_T_psm1tip = cam_T_world @ world_T_psm1tip
                init_psm1_tip_position = init_cam_T_psm1tip[:3, 3]
                cam_T_psm1tip = pose_to_transformation_matrix(init_cam_T_psm1tip[:3, 3], mtml_orientation)

                init_mtmr_position = mtmr_pos
                psm2_tip_pose_w = psm2.data.body_link_pos_w[0][-1].cpu().numpy()
                psm2_tip_quat_w = psm2.data.body_link_quat_w[0][-1].cpu().numpy()
                world_T_psm2tip = pose_to_transformation_matrix(psm2_tip_pose_w, psm2_tip_quat_w)
                init_cam_T_psm2tip = cam_T_world @ world_T_psm2tip
                init_psm2_tip_position = init_cam_T_psm2tip[:3, 3]
                cam_T_psm2tip = pose_to_transformation_matrix(init_cam_T_psm2tip[:3, 3], mtmr_orientation)

                actions = process_actions(cam_T_psm1tip, world_T_psm1_base, cam_T_psm2tip, world_T_psm2_base, world_T_cam, env, l_gripper_joint, r_gripper_joint)
            else:
                # normal operation
                psm1_target_position = init_psm1_tip_position + (mtml_pos - init_mtml_position)
                cam_T_psm1tip = pose_to_transformation_matrix(psm1_target_position, mtml_orientation)

                psm2_target_position = init_psm2_tip_position + (mtmr_pos - init_mtmr_position)
                cam_T_psm2tip = pose_to_transformation_matrix(psm2_target_position, mtmr_orientation)
                
                actions = process_actions(cam_T_psm1tip, world_T_psm1_base, cam_T_psm2tip, world_T_psm2_base, world_T_cam, env, l_gripper_joint, r_gripper_joint)
            was_in_clutch = False
        
        else:  # clutch pressed: stop moving, set was_in_clutch to True
            was_in_clutch = True

            psm1_cur_eef_pos = psm1.data.body_link_pos_w[0][-1].cpu().numpy()
            psm1_cur_eef_quat = psm1.data.body_link_quat_w[0][-1].cpu().numpy()
            world_T_psm1_tip = pose_to_transformation_matrix(psm1_cur_eef_pos, psm1_cur_eef_quat)

            psm2_cur_eef_pos = psm2.data.body_link_pos_w[0][-1].cpu().numpy()
            psm2_cur_eef_quat = psm2.data.body_link_quat_w[0][-1].cpu().numpy()
            world_T_psm2_tip = pose_to_transformation_matrix(psm2_cur_eef_pos, psm2_cur_eef_quat)

            psm1_base_T_psm1_tip = np.linalg.inv(world_T_psm1_base) @ world_T_psm1_tip
            psm2_base_T_psm2_tip = np.linalg.inv(world_T_psm2_base) @ world_T_psm2_tip

            psm1_rel_pos, psm1_rel_quat = transformation_matrix_to_pose(psm1_base_T_psm1_tip)
            psm2_rel_pos, psm2_rel_quat = transformation_matrix_to_pose(psm2_base_T_psm2_tip)

            psm1_gripper1_joint_angle = psm1.data.joint_pos[0][-2].cpu().numpy()
            psm1_gripper2_joint_angle = psm1.data.joint_pos[0][-1].cpu().numpy()

            psm2_gripper1_joint_angle = psm2.data.joint_pos[0][-2].cpu().numpy()
            psm2_gripper2_joint_angle = psm2.data.joint_pos[0][-1].cpu().numpy()

            actions = np.concatenate([psm1_rel_pos, psm1_rel_quat, [psm1_gripper1_joint_angle, psm1_gripper2_joint_angle], 
                                      psm2_rel_pos, psm2_rel_quat, [psm2_gripper1_joint_angle, psm2_gripper2_joint_angle]])
            actions = torch.tensor(actions, device=env.unwrapped.device).repeat(env.unwrapped.num_envs, 1)

        env.step(actions)
        # cam_l_input = camera_l.data.output["rgb"][0].cpu().numpy()
        # cam_r_input = camera_r.data.output["rgb"][0].cpu().numpy()
        # # Update displayed images
        # update_images(cam_l_input, cam_r_input)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
