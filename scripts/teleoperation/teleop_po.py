import argparse

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="PhantomOmni teleoperation for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-PO-Teleop-v0", help="Name of the task.")
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

import orbit.surgical.tasks  # noqa: F401
import matplotlib.pyplot as plt
import cv2
from scipy.spatial.transform import Rotation as R
import numpy as np
import time

from utils.tf_utils import pose_to_transformation_matrix, transformation_matrix_to_pose

import sys
import os
sys.path.append(os.path.abspath("."))
from teleop_interface.phantomomni.se3_phantomomni import PhantomOmniTeleop
import custom_envs

# process cam_T_psmtip to psmbase_T_psmtip and make usuable action input
def process_actions(cam_T_psm1, w_T_psm1base, cam_T_psm2, w_T_psm2base, w_T_cam, env, gripper1_command, gripper2_command) -> torch.Tensor:
    """Process actions for the environment."""
    psm1base_T_psm1 = np.linalg.inv(w_T_psm1base)@w_T_cam@cam_T_psm1
    psm2base_T_psm2 = np.linalg.inv(w_T_psm2base)@w_T_cam@cam_T_psm2
    psm1_rel_pos, psm1_rel_quat = transformation_matrix_to_pose(psm1base_T_psm1)
    psm2_rel_pos, psm2_rel_quat = transformation_matrix_to_pose(psm2base_T_psm2)
    actions = np.concatenate([psm1_rel_pos, psm1_rel_quat, [1.0 if gripper1_command else -1.0], psm2_rel_pos, psm2_rel_quat, [1.0 if gripper2_command else -1.0]])
    actions = torch.tensor(actions, device=env.unwrapped.device).repeat(env.unwrapped.num_envs, 1)
    return actions


def main():
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    # modify configuration
    env_cfg.terminations.time_out = None

    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()

    teleop_interface = PhantomOmniTeleop(pos_sensitivity=0.4, rot_sensitivity=1.0)
    teleop_interface.reset()

    camera_l = env.unwrapped.scene["camera_left"]
    camera_r = env.unwrapped.scene["camera_right"]

    psm1 = env.unwrapped.scene["robot_1"]
    psm2 = env.unwrapped.scene["robot_2"]

    was_in_clutch = True
    init_stylus_position = None
    print("Press the grey button and release to start teleoperation.")
    # simulate environment
    while simulation_app.is_running():
        # get 6D transform vector, gripper command, clutch from teleop interface
        stylus_pose, gripper_command, clutch = teleop_interface.advance()
        if clutch is None:
            print("Waiting for the topic subscription...")
            time.sleep(0.05)
            continue
        stylus_orientation = R.from_rotvec(stylus_pose[3:]).as_quat()
        stylus_orientation = np.concatenate([[stylus_orientation[3]], stylus_orientation[:3]])

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

        if not clutch:
            if was_in_clutch:
                print("Released from clutch. Starting teleoperation again")
                init_stylus_position = stylus_pose[:3]

                psm1_tip_pose_w = psm1.data.body_link_pos_w[0][-1].cpu().numpy()
                psm1_tip_quat_w = psm1.data.body_link_quat_w[0][-1].cpu().numpy()
                world_T_psm1tip = pose_to_transformation_matrix(psm1_tip_pose_w, psm1_tip_quat_w)
                init_cam_T_psm1tip = cam_T_world @ world_T_psm1tip
                init_psm1_tip_position = init_cam_T_psm1tip[:3, 3]
                cam_T_psm1tip = pose_to_transformation_matrix(init_cam_T_psm1tip[:3, 3], stylus_orientation)

                psm2_cur_eef_pos = psm2.data.body_link_pos_w[0][-1].cpu().numpy()
                psm2_cur_eef_quat = psm2.data.body_link_quat_w[0][-1].cpu().numpy()
                world_T_psm2tip = pose_to_transformation_matrix(psm2_cur_eef_pos, psm2_cur_eef_quat)
                cam_T_psm2tip = cam_T_world @ world_T_psm2tip

                actions = process_actions(cam_T_psm1tip, world_T_psm1_base, cam_T_psm2tip, world_T_psm2_base, world_T_cam, env, gripper_command, False)
            else:
                # normal operation
                target_position = init_psm1_tip_position + (stylus_pose[:3] - init_stylus_position)
                cam_T_psm1tip = pose_to_transformation_matrix(target_position, stylus_orientation)

                psm2_cur_eef_pos = psm2.data.body_link_pos_w[0][-1].cpu().numpy()
                psm2_cur_eef_quat = psm2.data.body_link_quat_w[0][-1].cpu().numpy()
                world_T_psm2tip = pose_to_transformation_matrix(psm2_cur_eef_pos, psm2_cur_eef_quat)
                cam_T_psm2tip = cam_T_world @ world_T_psm2tip
                
                actions = process_actions(cam_T_psm1tip, world_T_psm1_base, cam_T_psm2tip, world_T_psm2_base, world_T_cam, env, gripper_command, False)
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

            actions = np.concatenate([psm1_rel_pos, psm1_rel_quat, [False], psm2_rel_pos, psm2_rel_quat, [False]])
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
