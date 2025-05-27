import argparse
from omni.isaac.lab.app import AppLauncher
import time

# CLI arguments
parser = argparse.ArgumentParser(description="Playback teleop log using joint commands (3 PSMs).")
parser.add_argument("--csv_file", type=str, required=True)
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--task", type=str, default="Isaac-MTM-Teleop-pb-three-arm")
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
from omni.isaac.lab_tasks.utils import parse_env_cfg

# Utilities
sys.path.append(os.path.abspath("."))
import custom_envs
from teleop_logger import reset_cube_pose_from_json


def main():
    # Parse environment config
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()

    # Load cube pose from JSON and reset
    json_path = os.path.join(os.path.dirname(args_cli.csv_file), "pose.json")
    reset_cube_pose_from_json(env, json_path, cube_key="cube_rigid")  # or cube_deformable

    # Load CSV data
    df = pd.read_csv(args_cli.csv_file)
    time_stamps = df["Time (Seconds)"].values.astype(np.float64)
    time_stamps -= time_stamps[0]

    print(f"[INFO] Starting joint-space playback of {len(df)} frames...")

    # Joint columns
    psm1_joint_columns = [f"PSM1_joint_{i}" for i in range(1, 7)]
    psm2_joint_columns = [f"PSM2_joint_{i}" for i in range(1, 7)]
    psm3_joint_columns = [f"PSM3_joint_{i}" for i in range(1, 7)]
    psm1_jaw_column = "PSM1_jaw_angle"
    psm2_jaw_column = "PSM2_jaw_angle"
    psm3_jaw_column = "PSM3_jaw_angle"

    # Move to initial pose and hold
    print("[INFO] Holding initial pose from frame 0 for 3 seconds...")
    psm1_joints = df.loc[0, psm1_joint_columns].values.astype(np.float32)
    psm2_joints = df.loc[0, psm2_joint_columns].values.astype(np.float32)
    psm3_joints = df.loc[0, psm3_joint_columns].values.astype(np.float32)
    psm1_jaw = float(df.loc[0, psm1_jaw_column])
    psm2_jaw = float(df.loc[0, psm2_jaw_column])
    psm3_jaw = float(df.loc[0, psm3_jaw_column])
    psm1_gripper = [-psm1_jaw / 2, psm1_jaw / 2]
    psm2_gripper = [-psm2_jaw / 2, psm2_jaw / 2]
    psm3_gripper = [-psm3_jaw / 2, psm3_jaw / 2]

    init_action = np.concatenate([
        psm1_joints, psm1_gripper,
        psm2_joints, psm2_gripper,
        psm3_joints, psm3_gripper
    ], dtype=np.float32)

    init_tensor = torch.tensor(init_action, device=env.unwrapped.device).unsqueeze(0)
    for _ in range(90):  # 3 seconds at 30 Hz
        env.step(init_tensor)
        time.sleep(1.0 / 30.0)

    # Timed playback
    print("[INFO] Starting playback...")
    start_time_global = time.time()

    for frame_idx in range(len(df)):
        # Wait for the correct time
        target_timestamp = time_stamps[frame_idx]
        while True:
            current_time = time.time() - start_time_global
            sleep_time = target_timestamp - current_time
            if sleep_time > 0:
                time.sleep(min(sleep_time, 0.002))
            else:
                break

        # Get joint values
        psm1_joints = df.loc[frame_idx, psm1_joint_columns].values.astype(np.float32)
        psm2_joints = df.loc[frame_idx, psm2_joint_columns].values.astype(np.float32)
        psm3_joints = df.loc[frame_idx, psm3_joint_columns].values.astype(np.float32)
        psm1_jaw = float(df.loc[frame_idx, psm1_jaw_column])
        psm2_jaw = float(df.loc[frame_idx, psm2_jaw_column])
        psm3_jaw = float(df.loc[frame_idx, psm3_jaw_column])
        psm1_gripper = [-psm1_jaw / 2, psm1_jaw / 2]
        psm2_gripper = [-psm2_jaw / 2, psm2_jaw / 2]
        psm3_gripper = [-psm3_jaw / 2, psm3_jaw / 2]

        # Build action tensor
        action_vec = np.concatenate([
            psm1_joints, psm1_gripper,
            psm2_joints, psm2_gripper,
            psm3_joints, psm3_gripper
        ], dtype=np.float32)

        action_tensor = torch.tensor(action_vec, device=env.unwrapped.device).unsqueeze(0)

        # Step environment
        for _ in range(3):
            env.step(action_tensor)

    print("[INFO] Playback finished.")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
