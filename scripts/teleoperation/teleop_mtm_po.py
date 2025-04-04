import argparse

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Simultaneous MTM + PO teleoperation for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-MTM-PO-Teleop-v0", help="Name of the task.")
parser.add_argument("--scale", type=float, default=0.4, help="Teleop scaling factor.")
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

from omni.isaac.lab_tasks.utils import parse_env_cfg

import matplotlib.pyplot as plt
import cv2
from scipy.spatial.transform import Rotation as R
import numpy as np
import time

import omni.kit.viewport.utility as vp_utils
from tf_utils import pose_to_transformation_matrix, transformation_matrix_to_pose

import sys
import os
sys.path.append(os.path.abspath("."))
from teleop_interface.phantomomni.se3_phantomomni import PhantomOmniTeleop
from teleop_interface.MTM.se3_mtm import MTMTeleop
from teleop_interface.MTM.mtm_manipulator import MTMManipulator
import custom_envs

# map mtm gripper joint angle to psm jaw gripper angles in simulation
def get_jaw_gripper_angles(gripper_command, env, robot_name="robot_1"):
    if gripper_command is None:
        gripper1_joint_angle = env.unwrapped[robot_name].data.joint_pos[0][-2].cpu().numpy()
        gripper2_joint_angle = env.unwrapped[robot_name].data.joint_pos[0][-1].cpu().numpy()
        return np.array([gripper1_joint_angle, gripper2_joint_angle])
    # input: -1.72 (closed), 1.06 (opened)
    # output: 0,0 (closed), -0.52359, 0.52359 (opened)
    gripper2_angle = 0.52359 / (1.06 + 1.72) * (gripper_command + 1.72)
    return np.array([-gripper2_angle, gripper2_angle])

# process cam_T_psmtip to psmbase_T_psmtip and make usuable action input
def process_actions(cam_T_psm1, w_T_psm1base, cam_T_psm2, w_T_psm2base, cam_T_psm3, w_T_psm3base, w_T_cam, env, gripper1_command, gripper2_command, gripper3_command) -> torch.Tensor:
    """Process actions for the environment."""
    psm1base_T_psm1 = np.linalg.inv(w_T_psm1base)@w_T_cam@cam_T_psm1
    psm2base_T_psm2 = np.linalg.inv(w_T_psm2base)@w_T_cam@cam_T_psm2
    psm1_rel_pos, psm1_rel_quat = transformation_matrix_to_pose(psm1base_T_psm1)
    psm2_rel_pos, psm2_rel_quat = transformation_matrix_to_pose(psm2base_T_psm2)

    psm3base_T_psm3 = np.linalg.inv(w_T_psm3base)@w_T_cam@cam_T_psm3
    psm3_rel_pos, psm3_rel_quat = transformation_matrix_to_pose(psm3base_T_psm3)

    actions = np.concatenate([psm1_rel_pos, psm1_rel_quat, get_jaw_gripper_angles(gripper1_command, env, 'robot_1'), 
                              psm2_rel_pos, psm2_rel_quat, get_jaw_gripper_angles(gripper2_command, env, 'robot_2'),
                              psm3_rel_pos, psm3_rel_quat, [1.0 if gripper3_command else -1.0]])
    actions = torch.tensor(actions, device=env.unwrapped.device).repeat(env.unwrapped.num_envs, 1)
    return actions

def main():
    scale=args_cli.scale

    # Setup the MTM in the real world
    mtm_manipulator = MTMManipulator()
    mtm_manipulator.home()

    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    # modify configuration
    env_cfg.terminations.time_out = None

    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()

    mtm_interface = MTMTeleop()
    mtm_interface.reset()

    po_interface = PhantomOmniTeleop()
    po_interface.reset()

    camera_l = env.unwrapped.scene["camera_left"]
    camera_r = env.unwrapped.scene["camera_right"]

    view_port_l = vp_utils.create_viewport_window("Left Camera", width = 800, height = 600)
    view_port_l.viewport_api.camera_path = '/World/envs/env_0/Robot_4/ecm_end_link/camera_left' #camera_l.cfg.prim_path

    view_port_r = vp_utils.create_viewport_window("Right Camera", width = 800, height = 600)
    view_port_r.viewport_api.camera_path = '/World/envs/env_0/Robot_4/ecm_end_link/camera_right' #camera_r.cfg.prim_path

    psm1 = env.unwrapped.scene["robot_1"]
    psm2 = env.unwrapped.scene["robot_2"]
    psm3 = env.unwrapped.scene["robot_3"]

    mtm_orientation_matched = False
    was_in_mtm_clutch = True
    was_in_po_clutch = True
    init_mtml_position = None
    init_psm1_tip_position = None
    init_mtmr_position = None
    init_psm2_tip_position = None
    init_stylus_position = None
    init_psm3_tip_position = None

    print("Press the clutch button and release to start teleoperation.")
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()

        # process actions
        camera_l_pos = camera_l.data.pos_w
        camera_r_pos = camera_r.data.pos_w
        # get center of both cameras
        str_camera_pos = (camera_l_pos + camera_r_pos) / 2
        camera_quat = camera_l.data.quat_w_world  # forward x, up z
        world_T_cam = pose_to_transformation_matrix(str_camera_pos.cpu().numpy()[0], camera_quat.cpu().numpy()[0])
        cam_T_world = np.linalg.inv(world_T_cam)

        if not mtm_orientation_matched:
            print("Start matching orientation of MTM with the PSMs in the simulation. May take a few seconds.")
            mtm_orientation_matched = True
            psm1_tip_pose_w = psm1.data.body_link_pos_w[0][-1].cpu().numpy()
            psm1_tip_quat_w = psm1.data.body_link_quat_w[0][-1].cpu().numpy()
            world_T_psm1tip = pose_to_transformation_matrix(psm1_tip_pose_w, psm1_tip_quat_w)
            hrsv_T_mtml = mtm_interface.simpose2hrsvpose(cam_T_world @ world_T_psm1tip)

            psm2_tip_pose_w = psm2.data.body_link_pos_w[0][-1].cpu().numpy()
            psm2_tip_quat_w = psm2.data.body_link_quat_w[0][-1].cpu().numpy()
            world_T_psm2tip = pose_to_transformation_matrix(psm2_tip_pose_w, psm2_tip_quat_w)
            hrsv_T_mtmr = mtm_interface.simpose2hrsvpose(cam_T_world @ world_T_psm2tip)

            mtm_manipulator.adjust_orientation(hrsv_T_mtml, hrsv_T_mtmr)
            # mtm_manipulator.release_force()
            print("Initial orientation matched. Start teleoperation by pressing and releasing the clutch button.")
            continue

        psm1_base_link_pos = psm1.data.body_link_pos_w[0][0].cpu().numpy()
        psm1_base_link_quat = psm1.data.body_link_quat_w[0][0].cpu().numpy()
        world_T_psm1_base = pose_to_transformation_matrix(psm1_base_link_pos, psm1_base_link_quat)

        psm2_base_link_pos = psm2.data.body_link_pos_w[0][0].cpu().numpy()
        psm2_base_link_quat = psm2.data.body_link_quat_w[0][0].cpu().numpy()
        world_T_psm2_base = pose_to_transformation_matrix(psm2_base_link_pos, psm2_base_link_quat)

        psm3_base_link_pos = psm3.data.body_link_pos_w[0][0].cpu().numpy()
        psm3_base_link_quat = psm3.data.body_link_quat_w[0][0].cpu().numpy()
        world_T_psm3_base = pose_to_transformation_matrix(psm3_base_link_pos, psm3_base_link_quat)

        # get target pos, rot in camera view with joint and clutch commands
        mtml_pos, mtml_rot, l_gripper_joint, mtmr_pos, mtmr_rot, r_gripper_joint, mtm_clutch = mtm_interface.advance()
        if not l_gripper_joint:
            print("Waiting for the MTM topic subscription...")
            time.sleep(0.05)
            continue

        mtml_orientation = R.from_rotvec(mtml_rot).as_quat()
        mtml_orientation = np.concatenate([[mtml_orientation[3]], mtml_orientation[:3]])

        mtmr_orientation = R.from_rotvec(mtmr_rot).as_quat()
        mtmr_orientation = np.concatenate([[mtmr_orientation[3]], mtmr_orientation[:3]])

        stylus_pose, po_gripper, po_clutch = po_interface.advance()
        if po_clutch is None:
            print("Waiting for the PO topic subscription...")
            time.sleep(0.05)
            continue
        stylus_orientation = R.from_rotvec(stylus_pose[3:]).as_quat()
        stylus_orientation = np.concatenate([[stylus_orientation[3]], stylus_orientation[:3]])

        # Process MTM teleoperation action
        if not mtm_clutch:
            if was_in_mtm_clutch:
                print("Released from clutch. Starting teleoperation again")
                init_mtml_position = mtml_pos
                psm1_tip_pose_w = psm1.data.body_link_pos_w[0][-1].cpu().numpy()
                psm1_tip_quat_w = psm1.data.body_link_quat_w[0][-1].cpu().numpy()
                world_T_psm1tip = pose_to_transformation_matrix(psm1_tip_pose_w, psm1_tip_quat_w)
                init_cam_T_psm1tip = cam_T_world @ world_T_psm1tip
                init_psm1_tip_position = init_cam_T_psm1tip[:3, 3]
                cam_T_psm1tip = pose_to_transformation_matrix(init_cam_T_psm1tip[:3, 3], mtml_orientation)
                # psm1_gripper_angle = l_gripper_joint

                init_mtmr_position = mtmr_pos
                psm2_tip_pose_w = psm2.data.body_link_pos_w[0][-1].cpu().numpy()
                psm2_tip_quat_w = psm2.data.body_link_quat_w[0][-1].cpu().numpy()
                world_T_psm2tip = pose_to_transformation_matrix(psm2_tip_pose_w, psm2_tip_quat_w)
                init_cam_T_psm2tip = cam_T_world @ world_T_psm2tip
                init_psm2_tip_position = init_cam_T_psm2tip[:3, 3]
                cam_T_psm2tip = pose_to_transformation_matrix(init_cam_T_psm2tip[:3, 3], mtmr_orientation)
                # psm2_gripper_angle = r_gripper_joint

            else:
                # normal operation
                psm1_target_position = init_psm1_tip_position + (mtml_pos - init_mtml_position) * scale
                cam_T_psm1tip = pose_to_transformation_matrix(psm1_target_position, mtml_orientation)

                psm2_target_position = init_psm2_tip_position + (mtmr_pos - init_mtmr_position) * scale
                cam_T_psm2tip = pose_to_transformation_matrix(psm2_target_position, mtmr_orientation)

            was_in_mtm_clutch = False

        else:  # clutch pressed: stop moving, set was_in_clutch to True
            was_in_mtm_clutch = True

            psm1_cur_eef_pos = psm1.data.body_link_pos_w[0][-1].cpu().numpy()
            psm1_cur_eef_quat = psm1.data.body_link_quat_w[0][-1].cpu().numpy()
            world_T_psm1tip = pose_to_transformation_matrix(psm1_cur_eef_pos, psm1_cur_eef_quat)
            cam_T_psm1tip = cam_T_world @ world_T_psm1tip
            # psm1_gripper_angle = None

            psm2_cur_eef_pos = psm2.data.body_link_pos_w[0][-1].cpu().numpy()
            psm2_cur_eef_quat = psm2.data.body_link_quat_w[0][-1].cpu().numpy()
            world_T_psm2tip = pose_to_transformation_matrix(psm2_cur_eef_pos, psm2_cur_eef_quat)
            cam_T_psm2tip = cam_T_world @ world_T_psm2tip
            # psm2_gripper_angle = None

        # Process PO teleoperation input
        if not po_clutch:
            if was_in_po_clutch:
                print("Released from clutch. Starting teleoperation again")
                init_stylus_position = stylus_pose[:3]

                psm3_tip_pose_w = psm3.data.body_link_pos_w[0][-1].cpu().numpy()
                psm3_tip_quat_w = psm3.data.body_link_quat_w[0][-1].cpu().numpy()
                world_T_psm3tip = pose_to_transformation_matrix(psm3_tip_pose_w, psm3_tip_quat_w)
                init_cam_T_psm3tip = cam_T_world @ world_T_psm3tip
                init_psm3_tip_position = init_cam_T_psm3tip[:3, 3]
                cam_T_psm3tip = pose_to_transformation_matrix(init_cam_T_psm3tip[:3, 3], stylus_orientation)
            else:
                # normal operation
                target_position = init_psm3_tip_position + (stylus_pose[:3] - init_stylus_position) * scale
                cam_T_psm3tip = pose_to_transformation_matrix(target_position, stylus_orientation)
            was_in_po_clutch = False
        else:  # clutch pressed: stop moving, set was_in_po_clutch to True
            was_in_po_clutch = True
            psm3_cur_eef_pos = psm3.data.body_link_pos_w[0][-1].cpu().numpy()
            psm3_cur_eef_quat = psm3.data.body_link_quat_w[0][-1].cpu().numpy()
            world_T_psm3tip = pose_to_transformation_matrix(psm3_cur_eef_pos, psm3_cur_eef_quat)
            cam_T_psm3tip = cam_T_world @ world_T_psm3tip

        actions = process_actions(cam_T_psm1tip, world_T_psm1_base, cam_T_psm2tip, world_T_psm2_base, cam_T_psm3tip, world_T_psm3_base, world_T_cam, env, l_gripper_joint, r_gripper_joint, po_gripper)
        env.step(actions)
        time.sleep(max(0.0, 1/30.0 - time.time() + start_time))

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
