import argparse

from omni.isaac.lab.app import AppLauncher
from scripts.teleoperation.teleop_logger_3_arm import TeleopLogger


# add argparse arguments
parser = argparse.ArgumentParser(description="MTM teleoperation for Custom MultiArm dVRK environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-MTM-Teleop-v0", help="Name of the task.")
parser.add_argument("--scale", type=float, default=0.4, help="Teleop scaling factor.")
parser.add_argument("--is_simulated", type=bool, default=False, help="Whether the MTM input is from the simulated model or not.")
# parser.add_argument("--enable_logging", type=bool, default=False, help="Whether to log the teleoperation output or not.")
parser.add_argument("--enable_logging", action="store_true", help="Enable logging from the start (default is off)")
# parser.add_argument("--sim_time", type=float, default=30.0, help="Duration (in seconds) for teleoperation before auto exit.")
parser.add_argument("--log_trigger_file", type=str, default="log_trigger.txt",
    help="Path to a file that enables logging when it exists.")

parser.add_argument("--disable_viewport", action="store_true", help="Disable extra viewport windows.")
parser.add_argument("--demo_name", type=str, default=None, help="Custom name for the logging folder (e.g., 'demo_1')")


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

from omni.isaac.lab_tasks.utils import parse_env_cfg

import cv2
from scipy.spatial.transform import Rotation as R
import numpy as np
import time
from datetime import datetime
import math

import omni.kit.viewport.utility as vp_utils
import omni.kit.commands
from tf_utils import pose_to_transformation_matrix, transformation_matrix_to_pose
from logger_utils import CSVLogger

import sys
import os
import random
sys.path.append(os.path.abspath("."))
from teleop_interface.MTM.se3_mtm import MTMTeleop
from teleop_interface.MTM.mtm_manipulator import MTMManipulator
import custom_envs
from scripts.teleoperation.teleop_logger_2_arm import reset_cube_pose, log_current_pose

# map mtm gripper joint angle to psm jaw gripper angles in simulation
# def get_jaw_gripper_angles(gripper_command, env, robot_name="robot_2"):
#     if gripper_command is None:
#         gripper1_joint_angle = env.unwrapped[robot_name].data.joint_pos[0][-2].cpu().numpy()
#         gripper2_joint_angle = env.unwrapped[robot_name].data.joint_pos[0][-1].cpu().numpy()
#         return np.array([gripper1_joint_angle, gripper2_joint_angle])
#         # return np.array([-0.52359, 0.52359])
#     # input: -1.72 (closed), 1.06 (opened)
#     # output: 0,0 (closed), -0.52359, 0.52359 (opened)
#     gripper2_angle = 0.52359 / (1.06 + 1.72) * (gripper_command + 1.72)
#     return np.array([-gripper2_angle, gripper2_angle])

def get_jaw_gripper_angles(gripper_command, robot_name):
    if gripper_command is None:
        return np.array([0.0, 0.0])
    norm_cmd = np.interp(gripper_command, [-1.0, 1.0], [-1.72, 1.06])
    g2_angle = 0.52359 / (1.06 + 1.72) * (norm_cmd + 1.72)
    return np.array([-g2_angle, g2_angle])


# process cam_T_psmtip to psmbase_T_psmtip and make usuable action input
def process_actions(cam_T_psm1, w_T_psm1base, cam_T_psm2, w_T_psm2base, w_T_cam, env, gripper1_command, gripper2_command) -> torch.Tensor:
    """Process actions for the environment."""
    psm1base_T_psm1 = np.linalg.inv(w_T_psm1base)@w_T_cam@cam_T_psm1
    psm2base_T_psm2 = np.linalg.inv(w_T_psm2base)@w_T_cam@cam_T_psm2
    psm1_rel_pos, psm1_rel_quat = transformation_matrix_to_pose(psm1base_T_psm1)
    psm2_rel_pos, psm2_rel_quat = transformation_matrix_to_pose(psm2base_T_psm2)
    actions = np.concatenate([psm1_rel_pos, psm1_rel_quat, get_jaw_gripper_angles(gripper1_command,  'robot_1'),
                              psm2_rel_pos, psm2_rel_quat, get_jaw_gripper_angles(gripper2_command,  'robot_2')])
    actions = torch.tensor(actions, device=env.unwrapped.device).repeat(env.unwrapped.num_envs, 1)
    return actions

def reset_psm_to_initial_pose(
    env,
    saved_psm1_tip_pos_w, saved_psm1_tip_quat_w,
    saved_psm2_tip_pos_w, saved_psm2_tip_quat_w,
    world_T_psm1_base, world_T_psm2_base,
    num_steps=30
):
    print("[RESET] Moving PSMs to saved initial pose...")

    world_T_psm1_tip = pose_to_transformation_matrix(saved_psm1_tip_pos_w, saved_psm1_tip_quat_w)
    world_T_psm2_tip = pose_to_transformation_matrix(saved_psm2_tip_pos_w, saved_psm2_tip_quat_w)

    psm1_base_T_tip = np.linalg.inv(world_T_psm1_base) @ world_T_psm1_tip
    psm2_base_T_tip = np.linalg.inv(world_T_psm2_base) @ world_T_psm2_tip

    psm1_rel_pos, psm1_rel_quat = transformation_matrix_to_pose(psm1_base_T_tip)
    psm2_rel_pos, psm2_rel_quat = transformation_matrix_to_pose(psm2_base_T_tip)

    action_vec = np.concatenate([
        psm1_rel_pos, psm1_rel_quat, [0.0, 0.0],
        psm2_rel_pos, psm2_rel_quat, [0.0, 0.0]
    ], dtype=np.float32)

    action_tensor = torch.tensor(action_vec, device=env.unwrapped.device).unsqueeze(0)

    for _ in range(num_steps):
        env.step(action_tensor)

    print("[RESET] PSMs moved to initial pose.")


def reorient_mtm_to_match_psm(
    mtm_manipulator, teleop_interface,
    saved_psm1_tip_pos_w, saved_psm1_tip_quat_w,
    saved_psm2_tip_pos_w, saved_psm2_tip_quat_w,
    cam_T_world
):
    print("[RESET] Reorienting MTMs to match PSM tip pose...")

    world_T_psm1tip = pose_to_transformation_matrix(saved_psm1_tip_pos_w, saved_psm1_tip_quat_w)
    world_T_psm2tip = pose_to_transformation_matrix(saved_psm2_tip_pos_w, saved_psm2_tip_quat_w)

    cam_T_psm1tip = cam_T_world @ world_T_psm1tip
    cam_T_psm2tip = cam_T_world @ world_T_psm2tip

    hrsv_T_mtml = teleop_interface.simpose2hrsvpose(cam_T_psm1tip)
    hrsv_T_mtmr = teleop_interface.simpose2hrsvpose(cam_T_psm2tip)

    mtm_manipulator.home()
    time.sleep(2.0)
    mtm_manipulator.adjust_orientation(hrsv_T_mtml, hrsv_T_mtmr)

    print("[RESET] MTMs reoriented.")


def _compute_base_relative_action(env, psm1, psm2, jaw1, jaw2):
    """Helper to compute and return action tensor given PSMs and target jaw angles."""
    # Get world-frame tip poses
    psm1_tip_pos = psm1.data.body_link_pos_w[0][-1].cpu().numpy()
    psm1_tip_quat = psm1.data.body_link_quat_w[0][-1].cpu().numpy()
    psm2_tip_pos = psm2.data.body_link_pos_w[0][-1].cpu().numpy()
    psm2_tip_quat = psm2.data.body_link_quat_w[0][-1].cpu().numpy()
    world_T_psm1tip = pose_to_transformation_matrix(psm1_tip_pos, psm1_tip_quat)
    world_T_psm2tip = pose_to_transformation_matrix(psm2_tip_pos, psm2_tip_quat)

    # Get base transforms
    psm1_base_link_pos = psm1.data.body_link_pos_w[0][0].cpu().numpy()
    psm1_base_link_quat = psm1.data.body_link_quat_w[0][0].cpu().numpy()
    psm2_base_link_pos = psm2.data.body_link_pos_w[0][0].cpu().numpy()
    psm2_base_link_quat = psm2.data.body_link_quat_w[0][0].cpu().numpy()
    world_T_psm1_base = pose_to_transformation_matrix(psm1_base_link_pos, psm1_base_link_quat)
    world_T_psm2_base = pose_to_transformation_matrix(psm2_base_link_pos, psm2_base_link_quat)

    # Compute base-relative poses
    psm1_rel_pos, psm1_rel_quat = transformation_matrix_to_pose(np.linalg.inv(world_T_psm1_base) @ world_T_psm1tip)
    psm2_rel_pos, psm2_rel_quat = transformation_matrix_to_pose(np.linalg.inv(world_T_psm2_base) @ world_T_psm2tip)

    # Construct action tensor
    action_tensor = torch.tensor(
        np.concatenate([psm1_rel_pos, psm1_rel_quat, jaw1, psm2_rel_pos, psm2_rel_quat, jaw2], dtype=np.float32),
        device=env.unwrapped.device
    ).unsqueeze(0)

    return action_tensor

def open_jaws(env, psm1, psm2, target=1.04):
    """Open PSM jaws to match a target angle."""
    jaw_angle = 0.52359 / (1.06 + 1.72) * (target + 1.72)
    jaw1 = [-jaw_angle, jaw_angle]
    jaw2 = [-jaw_angle, jaw_angle]

    print(f"[INIT] Opening jaws to approximate MTM input: {target} → [{jaw1[0]:.3f}, {jaw1[1]:.3f}]")
    action_tensor = _compute_base_relative_action(env, psm1, psm2, jaw1=jaw1, jaw2=jaw2)
    for _ in range(30):
        env.step(action_tensor)
    print("[INIT] PSM jaws opened.")


def set_jaws_closed(env, psm1, psm2):
    """Force-close jaws after reset to override any input artifacts."""
    print("[FIX] Forcing PSM jaws closed...")
    action_tensor = _compute_base_relative_action(env, psm1, psm2, jaw1=[0.0, 0.0], jaw2=[-0.0, 0.0])
    for _ in range(30):
        env.step(action_tensor)
    print("[FIX] PSM jaws closed.")




def main():

    saved_psm1_tip_pos_w = None
    saved_psm1_tip_quat_w = None
    saved_psm2_tip_pos_w = None
    saved_psm2_tip_quat_w = None

    is_simulated = args_cli.is_simulated
    scale=args_cli.scale
    enable_logging = args_cli.enable_logging

    psm_name_dict = {
        "PSM1": "robot_1",
        "PSM2": "robot_2"
    }

    

    # logger = None
    # frame_num = 0

    teleop_logger = TeleopLogger(
        trigger_file="log_trigger.txt",
        psm_name_dict=psm_name_dict,
        log_duration=30.0,
    )


        
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

    teleop_interface = MTMTeleop(is_simulated=is_simulated)
    teleop_interface.reset()

    # Add a new flag to track if we have reset the PSMs
    has_synced_psms = False
    teleop_started = False
    logging_start_time = None

    camera_l = env.unwrapped.scene["camera_left"]
    camera_r = env.unwrapped.scene["camera_right"]

    if not is_simulated and not args_cli.disable_viewport:
        view_port_l = vp_utils.create_viewport_window("Left Camera", width = 800, height = 600)
        view_port_l.viewport_api.camera_path = '/World/envs/env_0/Robot_4/ecm_end_link/camera_left' #camera_l.cfg.prim_path

        view_port_r = vp_utils.create_viewport_window("Right Camera", width = 800, height = 600)
        view_port_r.viewport_api.camera_path = '/World/envs/env_0/Robot_4/ecm_end_link/camera_right' #camera_r.cfg.prim_path


    psm1 = env.unwrapped.scene[psm_name_dict["PSM1"]]
    psm2 = env.unwrapped.scene[psm_name_dict["PSM2"]]

    mtm_orientation_matched = False
    was_in_clutch = True
    init_mtml_position = None
    init_psm1_tip_position = None
    init_mtmr_position = None
    init_psm2_tip_position = None


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
            hrsv_T_mtml = teleop_interface.simpose2hrsvpose(cam_T_world @ world_T_psm1tip)

            psm2_tip_pose_w = psm2.data.body_link_pos_w[0][-1].cpu().numpy()
            psm2_tip_quat_w = psm2.data.body_link_quat_w[0][-1].cpu().numpy()
            world_T_psm2tip = pose_to_transformation_matrix(psm2_tip_pose_w, psm2_tip_quat_w)
            hrsv_T_mtmr = teleop_interface.simpose2hrsvpose(cam_T_world @ world_T_psm2tip)

            mtm_manipulator.adjust_orientation(hrsv_T_mtml, hrsv_T_mtmr)
            print("Initial orientation matched. Start teleoperation by pressing and releasing the clutch button.")

            saved_psm1_tip_pos_w = psm1_tip_pose_w
            saved_psm1_tip_quat_w = psm1_tip_quat_w
            saved_psm2_tip_pos_w = psm2_tip_pose_w
            saved_psm2_tip_quat_w = psm2_tip_quat_w

            open_jaws(env, psm1, psm2)

            continue

        # get target pos, rot in camera view with joint and clutch commands
        (
                mtml_pos, mtml_rot, l_gripper_joint,
                mtmr_pos, mtmr_rot, r_gripper_joint,
                clutch, mono,
                trigger_reset
            ) = teleop_interface.advance()

        if not l_gripper_joint:
            time.sleep(0.05)
            continue
        
        psm1_base_link_pos = psm1.data.body_link_pos_w[0][0].cpu().numpy()
        psm1_base_link_quat = psm1.data.body_link_quat_w[0][0].cpu().numpy()
        world_T_psm1_base = pose_to_transformation_matrix(psm1_base_link_pos, psm1_base_link_quat)

        psm2_base_link_pos = psm2.data.body_link_pos_w[0][0].cpu().numpy()
        psm2_base_link_quat = psm2.data.body_link_quat_w[0][0].cpu().numpy()
        world_T_psm2_base = pose_to_transformation_matrix(psm2_base_link_pos, psm2_base_link_quat)

        

        mtml_orientation = R.from_rotvec(mtml_rot).as_quat()
        mtml_orientation = np.concatenate([[mtml_orientation[3]], mtml_orientation[:3]])
        mtmr_orientation = R.from_rotvec(mtmr_rot).as_quat()
        mtmr_orientation = np.concatenate([[mtmr_orientation[3]], mtmr_orientation[:3]])


        # stop teleoperation if mono button is pressed
        if mono:
            print("Mono button pressed. Stopping teleoperation.")
            mtm_manipulator.hold_position()
            break

        

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
                psm1_target_position = init_psm1_tip_position + (mtml_pos - init_mtml_position) * scale
                cam_T_psm1tip = pose_to_transformation_matrix(psm1_target_position, mtml_orientation)

                psm2_target_position = init_psm2_tip_position + (mtmr_pos - init_mtmr_position) * scale
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


        if os.path.exists("reset_trigger.txt"):
            print("[RESET-TRIGGER] reset_trigger.txt detected. Resetting environment...")
            env.reset()

            time.sleep(0.5)  # Give sim time to stabilize

            # Random position
            cube_pos_x = random.uniform(0.0, 0.1)
            cube_pos_y = random.uniform(-0.05, 0.05)
            cube_pos_z = 0.0  # slightly above table

            # Random yaw rotation (z-axis only)
            cube_yaw = random.uniform(-math.pi, math.pi)
            cube_quat = R.from_euler("z", cube_yaw).as_quat()  # (x, y, z, w)
            cube_orientation = [cube_quat[3], cube_quat[0], cube_quat[1], cube_quat[2]]  # (w, x, y, z)

            reset_cube_pose(
                env,
                log_dir="teleop_logs/cube_latest",
                position=[cube_pos_x, cube_pos_y, cube_pos_z],
                orientation=cube_orientation,
            )

            was_in_clutch = True
            has_synced_psms = False
            teleop_started = False

            # Recompute cam_T_world and base transforms
            camera_l_pos = camera_l.data.pos_w
            camera_r_pos = camera_r.data.pos_w
            str_camera_pos = (camera_l_pos + camera_r_pos) / 2
            camera_quat = camera_l.data.quat_w_world
            world_T_cam = pose_to_transformation_matrix(str_camera_pos.cpu().numpy()[0], camera_quat.cpu().numpy()[0])
            cam_T_world = np.linalg.inv(world_T_cam)

            psm1_base_link_pos = psm1.data.body_link_pos_w[0][0].cpu().numpy()
            psm1_base_link_quat = psm1.data.body_link_quat_w[0][0].cpu().numpy()
            world_T_psm1_base = pose_to_transformation_matrix(psm1_base_link_pos, psm1_base_link_quat)

            psm2_base_link_pos = psm2.data.body_link_pos_w[0][0].cpu().numpy()
            psm2_base_link_quat = psm2.data.body_link_quat_w[0][0].cpu().numpy()
            world_T_psm2_base = pose_to_transformation_matrix(psm2_base_link_pos, psm2_base_link_quat)

            # Step 1: reset PSMs
            reset_psm_to_initial_pose(
                env,
                saved_psm1_tip_pos_w, saved_psm1_tip_quat_w,
                saved_psm2_tip_pos_w, saved_psm2_tip_quat_w,
                world_T_psm1_base, world_T_psm2_base,
                num_steps=30
            )

            # Step 2: reorient MTMs
            reorient_mtm_to_match_psm(
                mtm_manipulator, teleop_interface,
                saved_psm1_tip_pos_w, saved_psm1_tip_quat_w,
                saved_psm2_tip_pos_w, saved_psm2_tip_quat_w,
                cam_T_world
            )

            open_jaws(env, psm1, psm2, target=0.8) 

            # Delete file and force reclutch
            os.remove("reset_trigger.txt")
            
            
            print("[RESET-TRIGGER] Reset complete. Please clutch to resume teleoperation.")

        teleop_logger.check_and_start_logging(env)

        if teleop_logger.enable_logging and teleop_logger.frame_num == 0:
            log_current_pose(env, teleop_logger.log_dir)

        teleop_logger.stop_logging()

        if teleop_logger.enable_logging:
            # PREPARE ALL DATA ON MAIN THREAD
            frame_num = teleop_logger.frame_num + 1
            timestamp = time.time()

            robot_states = {}
            for psm, robot_name in psm_name_dict.items():
                robot = env.unwrapped.scene[robot_name]
                joint_positions = robot.data.joint_pos[0][:6].cpu().numpy()
                jaw_angle = abs(robot.data.joint_pos[0][-2].cpu().numpy()) + abs(robot.data.joint_pos[0][-1].cpu().numpy())
                ee_position = robot.data.body_link_pos_w[0][-1].cpu().numpy()
                ee_quat = robot.data.body_link_quat_w[0][-1].cpu().numpy()
                orientation_matrix = R.from_quat(np.concatenate([ee_quat[1:], [ee_quat[0]]])).as_matrix()

                robot_states[psm] = {
                    "joint_positions": joint_positions,
                    "jaw_angle": jaw_angle,
                    "ee_position": ee_position,
                    "orientation_matrix": orientation_matrix,
                }

            cam_l_img = camera_l.data.output["rgb"][0].cpu().numpy()
            cam_r_img = camera_r.data.output["rgb"][0].cpu().numpy()

            teleop_logger.enqueue(frame_num, timestamp, robot_states, cam_l_img, cam_r_img)
            teleop_logger.frame_num = frame_num


        elapsed = time.time() - start_time
        sleep_time = max(0.0, (1/200.0) - elapsed)
        time.sleep(sleep_time)

    # close the simulator
    env.close()
    if os.path.exists(args_cli.log_trigger_file):
        os.remove(args_cli.log_trigger_file)



if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    teleop_logger.shutdown()

    simulation_app.close()