import argparse
from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Playback of teleoperated MTM-PSM motion.")
parser.add_argument("--csv_file", type=str, required=True, help="Path to the teleop log CSV file for playback.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-MTM-Teleop-v0", help="Name of the task.")
parser.add_argument("--is_simulated", action="store_true", default=False, help="Run in simulation mode.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
import pandas as pd
import numpy as np
import time
import sys
import os

sys.path.append(os.path.abspath("."))
from omni.isaac.lab_tasks.utils import parse_env_cfg
import omni.kit.viewport.utility as vp_utils
from teleop_interface.MTM.mtm_manipulator import MTMManipulator
from teleop_interface.MTM.se3_mtm import MTMTeleop
from tf_utils import pose_to_transformation_matrix, transformation_matrix_to_pose
import custom_envs
from scipy.spatial.transform import Rotation as R


def process_actions(cam_T_psm1, w_T_psm1base, cam_T_psm2, w_T_psm2base, w_T_cam, env, gripper1_command, gripper2_command):
    """Process actions for the environment."""
    psm1base_T_psm1 = np.linalg.inv(w_T_psm1base) @ w_T_cam @ cam_T_psm1
    psm2base_T_psm2 = np.linalg.inv(w_T_psm2base) @ w_T_cam @ cam_T_psm2
    psm1_rel_pos, psm1_rel_quat = transformation_matrix_to_pose(psm1base_T_psm1)
    psm2_rel_pos, psm2_rel_quat = transformation_matrix_to_pose(psm2base_T_psm2)
    
    # Jaw angle mapping logic (assuming you want 0.52359 scaling)
    gripper2_angle1 = 0.52359 / (1.06 + 1.72) * (gripper1_command + 1.72)
    gripper2_angle2 = 0.52359 / (1.06 + 1.72) * (gripper2_command + 1.72)
    
    jaw_angles1 = np.array([-gripper2_angle1, gripper2_angle1])
    jaw_angles2 = np.array([-gripper2_angle2, gripper2_angle2])

    actions = np.concatenate([
        psm1_rel_pos, psm1_rel_quat, jaw_angles1,
        psm2_rel_pos, psm2_rel_quat, jaw_angles2
    ])
    actions = torch.tensor(actions, device=env.unwrapped.device).unsqueeze(0)
    return actions

def main():
    # Setup hardware interface (safe for simulated too)
    mtm_manipulator = MTMManipulator()
    mtm_manipulator.home()
    teleop_interface = MTMTeleop(is_simulated=args_cli.is_simulated)
    teleop_interface.reset()
    teleop_interface.clutch = False

    # Parse env config
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    env_cfg.terminations.time_out = None

    # Create env
    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()

    # Load CSV
    df = pd.read_csv(args_cli.csv_file)
    total_frames = len(df)

    # Setup cameras (optional for visualization)
    camera_l = env.unwrapped.scene["camera_left"]
    camera_r = env.unwrapped.scene["camera_right"]
    view_port_l = vp_utils.create_viewport_window("Left Camera", width=800, height=600)
    view_port_l.viewport_api.camera_path = '/World/envs/env_0/Robot_4/ecm_end_link/camera_left'
    view_port_r = vp_utils.create_viewport_window("Right Camera", width=800, height=600)
    view_port_r.viewport_api.camera_path = '/World/envs/env_0/Robot_4/ecm_end_link/camera_right'

    # Setup robots
    psm1 = env.unwrapped.scene["robot_1"]
    psm2 = env.unwrapped.scene["robot_2"]

    # Wait for start trigger
    print("Waiting for external trigger (start_playback.txt) to start playback...")
    while simulation_app.is_running():
        simulation_app.update()
        if os.path.exists("start_playback.txt"):
            print("[START] External trigger detected. Resetting environment...")
            env.reset()
            os.remove("start_playback.txt")
            break
        time.sleep(0.05)

    # Calculate world_T_cam
    camera_l_pos = camera_l.data.pos_w
    camera_r_pos = camera_r.data.pos_w
    str_camera_pos = (camera_l_pos + camera_r_pos) / 2
    camera_quat = camera_l.data.quat_w_world
    world_T_cam = pose_to_transformation_matrix(str_camera_pos.cpu().numpy()[0], camera_quat.cpu().numpy()[0])
    cam_T_world = np.linalg.inv(world_T_cam)

    # Get PSM base poses
    psm1_base_link_pos = psm1.data.body_link_pos_w[0][0].cpu().numpy()
    psm1_base_link_quat = psm1.data.body_link_quat_w[0][0].cpu().numpy()
    world_T_psm1_base = pose_to_transformation_matrix(psm1_base_link_pos, psm1_base_link_quat)

    psm2_base_link_pos = psm2.data.body_link_pos_w[0][0].cpu().numpy()
    psm2_base_link_quat = psm2.data.body_link_quat_w[0][0].cpu().numpy()
    world_T_psm2_base = pose_to_transformation_matrix(psm2_base_link_pos, psm2_base_link_quat)

    print("[INFO] Starting playback...")

    frame_idx = 0
    while simulation_app.is_running():
        start_time = time.time()
        simulation_app.update()

        if frame_idx >= total_frames:
            print("Playback complete.")
            break

        row = df.iloc[frame_idx]

        # Load EE pose and orientation from CSV
        # --- PSM1 ---
        psm1_pos = np.array([
            row["PSM1_ee_x"],
            row["PSM1_ee_y"],
            row["PSM1_ee_z"],
        ])
        psm1_ori_matrix = np.array([
            [row["PSM1_Orientation_Matrix_[1,1]"], row["PSM1_Orientation_Matrix_[1,2]"], row["PSM1_Orientation_Matrix_[1,3]"]],
            [row["PSM1_Orientation_Matrix_[2,1]"], row["PSM1_Orientation_Matrix_[2,2]"], row["PSM1_Orientation_Matrix_[2,3]"]],
            [row["PSM1_Orientation_Matrix_[3,1]"], row["PSM1_Orientation_Matrix_[3,2]"], row["PSM1_Orientation_Matrix_[3,3]"]],
        ])
        psm1_quat = R.from_matrix(psm1_ori_matrix).as_quat()
        psm1_quat = np.concatenate([[psm1_quat[3]], psm1_quat[:3]])
        cam_T_psm1tip = pose_to_transformation_matrix(psm1_pos, psm1_quat)

        # --- PSM2 ---
        psm2_pos = np.array([
            row["PSM2_ee_x"],
            row["PSM2_ee_y"],
            row["PSM2_ee_z"],
        ])
        psm2_ori_matrix = np.array([
            [row["PSM2_Orientation_Matrix_[1,1]"], row["PSM2_Orientation_Matrix_[1,2]"], row["PSM2_Orientation_Matrix_[1,3]"]],
            [row["PSM2_Orientation_Matrix_[2,1]"], row["PSM2_Orientation_Matrix_[2,2]"], row["PSM2_Orientation_Matrix_[2,3]"]],
            [row["PSM2_Orientation_Matrix_[3,1]"], row["PSM2_Orientation_Matrix_[3,2]"], row["PSM2_Orientation_Matrix_[3,3]"]],
        ])
        psm2_quat = R.from_matrix(psm2_ori_matrix).as_quat()
        psm2_quat = np.concatenate([[psm2_quat[3]], psm2_quat[:3]])
        cam_T_psm2tip = pose_to_transformation_matrix(psm2_pos, psm2_quat)

        # Jaw angles
        psm1_jaw_angle = row["PSM1_jaw_angle"]
        psm2_jaw_angle = row["PSM2_jaw_angle"]

        # --- Now use your original process_actions function ---


        actions = process_actions(
            cam_T_psm1tip, world_T_psm1_base,
            cam_T_psm2tip, world_T_psm2_base,
            world_T_cam,
            env,
            psm1_jaw_angle, psm2_jaw_angle
        )

        # Step environment
        env.step(actions)

        frame_idx += 1

        # Maintain ~30Hz
        time.sleep(max(0.0, 1/30.0 - time.time() + start_time))

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
