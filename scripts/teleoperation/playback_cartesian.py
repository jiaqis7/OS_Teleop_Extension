import argparse
from omni.isaac.lab.app import AppLauncher
import time

# CLI arguments
parser = argparse.ArgumentParser(description="Playback teleop log using task-space commands.")
parser.add_argument("--csv_file", type=str, required=True)
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--task", type=str, default="Isaac-MTM-Teleop-v0")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch IsaacSim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# IsaacLab imports
import gymnasium as gym
import torch
import pandas as pd
import numpy as np
import sys
import os
from scipy.spatial.transform import Rotation as R
from omni.isaac.lab_tasks.utils import parse_env_cfg

# Utilities
sys.path.append(os.path.abspath("."))
from tf_utils import pose_to_transformation_matrix, transformation_matrix_to_pose
import custom_envs


def main():
    # Parse env config and create env
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()

    # Load data
    df = pd.read_csv(args_cli.csv_file)
    time_stamps = df["Time (Seconds)"].values.astype(np.float64)
    time_stamps -= time_stamps[0]

    print(f"[INFO] Starting task-space playback of {len(df)} frames...")

    # Get robot handles
    psm1 = env.unwrapped.scene["robot_1"]
    psm2 = env.unwrapped.scene["robot_2"]

    # Get world_T_base for both robots
    world_T_psm1_base = pose_to_transformation_matrix(
        psm1.data.body_link_pos_w[0][0].cpu().numpy(),
        psm1.data.body_link_quat_w[0][0].cpu().numpy()
    )
    world_T_psm2_base = pose_to_transformation_matrix(
        psm2.data.body_link_pos_w[0][0].cpu().numpy(),
        psm2.data.body_link_quat_w[0][0].cpu().numpy()
    )

    # Sync timing
    start_time_global = time.time()

    for frame_idx in range(len(df)):
        # Wait until target timestamp is reached
        target_timestamp = time_stamps[frame_idx]
        while True:
            current_time = time.time() - start_time_global
            sleep_time = target_timestamp - current_time
            if sleep_time > 0:
                time.sleep(min(sleep_time, 0.002))
            else:
                break

        # Extract world-frame PSM tip pose
        psm1_pos = df.loc[frame_idx, ["PSM1_ee_x", "PSM1_ee_y", "PSM1_ee_z"]].values.astype(np.float32)
        psm2_pos = df.loc[frame_idx, ["PSM2_ee_x", "PSM2_ee_y", "PSM2_ee_z"]].values.astype(np.float32)

        rot1 = np.array([
            [df.loc[frame_idx, 'PSM1_Orientation_Matrix_[1,1]'], df.loc[frame_idx, 'PSM1_Orientation_Matrix_[1,2]'], df.loc[frame_idx, 'PSM1_Orientation_Matrix_[1,3]']],
            [df.loc[frame_idx, 'PSM1_Orientation_Matrix_[2,1]'], df.loc[frame_idx, 'PSM1_Orientation_Matrix_[2,2]'], df.loc[frame_idx, 'PSM1_Orientation_Matrix_[2,3]']],
            [df.loc[frame_idx, 'PSM1_Orientation_Matrix_[3,1]'], df.loc[frame_idx, 'PSM1_Orientation_Matrix_[3,2]'], df.loc[frame_idx, 'PSM1_Orientation_Matrix_[3,3]']],
        ])
        rot2 = np.array([
            [df.loc[frame_idx, 'PSM2_Orientation_Matrix_[1,1]'], df.loc[frame_idx, 'PSM2_Orientation_Matrix_[1,2]'], df.loc[frame_idx, 'PSM2_Orientation_Matrix_[1,3]']],
            [df.loc[frame_idx, 'PSM2_Orientation_Matrix_[2,1]'], df.loc[frame_idx, 'PSM2_Orientation_Matrix_[2,2]'], df.loc[frame_idx, 'PSM2_Orientation_Matrix_[2,3]']],
            [df.loc[frame_idx, 'PSM2_Orientation_Matrix_[3,1]'], df.loc[frame_idx, 'PSM2_Orientation_Matrix_[3,2]'], df.loc[frame_idx, 'PSM2_Orientation_Matrix_[3,3]']],
        ])

        world_T_psm1_tip = np.eye(4)
        world_T_psm1_tip[:3, :3] = rot1
        world_T_psm1_tip[:3, 3] = psm1_pos

        world_T_psm2_tip = np.eye(4)
        world_T_psm2_tip[:3, :3] = rot2
        world_T_psm2_tip[:3, 3] = psm2_pos

        # Transform to base frame
        psm1_base_T_tip = np.linalg.inv(world_T_psm1_base) @ world_T_psm1_tip
        psm2_base_T_tip = np.linalg.inv(world_T_psm2_base) @ world_T_psm2_tip

        psm1_rel_pos, psm1_rel_quat = transformation_matrix_to_pose(psm1_base_T_tip)
        psm2_rel_pos, psm2_rel_quat = transformation_matrix_to_pose(psm2_base_T_tip)

        # Gripper control
        jaw1 = float(df.loc[frame_idx, "PSM1_jaw_angle"])
        jaw2 = float(df.loc[frame_idx, "PSM2_jaw_angle"])
        gripper1 = [-jaw1 / 2, jaw1 / 2]
        gripper2 = [-jaw2 / 2, jaw2 / 2]

        # Final action tensor
        action_vec = np.concatenate([
            psm1_rel_pos, psm1_rel_quat, gripper1,
            psm2_rel_pos, psm2_rel_quat, gripper2
        ], dtype=np.float32)

        action_tensor = torch.tensor(action_vec, device=env.unwrapped.device).unsqueeze(0)

        # Step simulation
        for _ in range(3):
            env.step(action_tensor)

    print("[INFO] Playback finished.")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
