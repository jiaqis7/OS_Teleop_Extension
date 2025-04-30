import argparse
from omni.isaac.lab.app import AppLauncher
import time

# Step 1: CLI arguments
parser = argparse.ArgumentParser(description="Playback teleop log using joint commands.")
parser.add_argument("--csv_file", type=str, required=True)
# parser.add_argument("--frame_id", type=int, default=0)
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--task", type=str, default="Isaac-MTM-Teleop-pb")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Step 2: Launch IsaacSim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Step 3: Now import IsaacLab / IsaacSim stuff
import gymnasium as gym
import torch
import pandas as pd
import numpy as np
import sys
import os
from omni.isaac.lab_tasks.utils import parse_env_cfg

sys.path.append(os.path.abspath("."))
import custom_envs




# def main():
#     env_cfg = parse_env_cfg(
#         args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
#     )
#     env = gym.make(args_cli.task, cfg=env_cfg)

#     # Read CSV
#     df = pd.read_csv(args_cli.csv_file)

#     psm1_joint_columns = [f"PSM1_joint_{i}" for i in range(1, 7)]
#     psm2_joint_columns = [f"PSM2_joint_{i}" for i in range(1, 7)]
#     psm1_jaw_column = "PSM1_jaw_angle"
#     psm2_jaw_column = "PSM2_jaw_angle"

#     time_stamps = df["Time (Seconds)"].values.astype(np.float64)
#     time_stamps -= time_stamps[0]  # Normalize to start from 0.0

#     print(f"[INFO] Start playing back {len(df)} frames based on original timestamps...")

#     env.reset()
#     start_time_global = time.time()

#     for frame_idx in range(len(df)):
#         # Wait until the timestamped time is reached
#         target_timestamp = time_stamps[frame_idx]
#         while True:
#             current_time = time.time() - start_time_global
#             sleep_time = target_timestamp - current_time
#             if sleep_time > 0:
#                 time.sleep(min(sleep_time, 0.002))  # sleep in small chunks
#             else:
#                 break

#         # Extract and process frame
#         psm1_joints = df.loc[frame_idx, psm1_joint_columns].values.astype(np.float32)
#         psm2_joints = df.loc[frame_idx, psm2_joint_columns].values.astype(np.float32)
#         psm1_jaw = float(df.loc[frame_idx, psm1_jaw_column])
#         psm2_jaw = float(df.loc[frame_idx, psm2_jaw_column])

#         psm1_gripper1 = -psm1_jaw / 2
#         psm1_gripper2 = psm1_jaw / 2
#         psm2_gripper1 = -psm2_jaw / 2
#         psm2_gripper2 = psm2_jaw / 2

#         action = np.concatenate([
#             psm1_joints, [psm1_gripper1, psm1_gripper2],
#             psm2_joints, [psm2_gripper1, psm2_gripper2]
#         ], dtype=np.float32)

#         action = torch.tensor(action, device=env.unwrapped.device).unsqueeze(0)
#         # env.step(action)

#         for _ in range(3):
#             env.step(action)
#             time.sleep(0.0)  # adjust to slow down playback if needed

#     print("[INFO] Playback finished. Stopping simulation.")
#     env.close()


def main():
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    env = gym.make(args_cli.task, cfg=env_cfg)

    df = pd.read_csv(args_cli.csv_file)

    psm1_joint_columns = [f"PSM1_joint_{i}" for i in range(1, 7)]
    psm2_joint_columns = [f"PSM2_joint_{i}" for i in range(1, 7)]
    psm1_jaw_column = "PSM1_jaw_angle"
    psm2_jaw_column = "PSM2_jaw_angle"

    time_stamps = df["Time (Seconds)"].values.astype(np.float64)
    time_stamps -= time_stamps[0]

    print(f"[INFO] Moving to initial position, then starting playback of {len(df)} frames...")

    env.reset()

    # Extract first frame joint values
    psm1_joints = df.loc[0, psm1_joint_columns].values.astype(np.float32)
    psm2_joints = df.loc[0, psm2_joint_columns].values.astype(np.float32)
    psm1_jaw = float(df.loc[0, psm1_jaw_column])
    psm2_jaw = float(df.loc[0, psm2_jaw_column])
    psm1_gripper1 = -psm1_jaw / 2
    psm1_gripper2 = psm1_jaw / 2
    psm2_gripper1 = -psm2_jaw / 2
    psm2_gripper2 = psm2_jaw / 2

    init_action = np.concatenate([
        psm1_joints, [psm1_gripper1, psm1_gripper2],
        psm2_joints, [psm2_gripper1, psm2_gripper2]
    ], dtype=np.float32)
    init_action = torch.tensor(init_action, device=env.unwrapped.device).unsqueeze(0)

    # Move to initial pose and hold for a few seconds
    print("[INFO] Holding initial pose from frame 0 for 3 seconds...")
    for _ in range(90):  # 30 Hz × 3 sec
        env.step(init_action)
        time.sleep(1.0 / 30.0)

    print("[INFO] Starting playback...")

    start_time_global = time.time()

    for frame_idx in range(len(df)):
        target_timestamp = time_stamps[frame_idx]
        while True:
            current_time = time.time() - start_time_global
            sleep_time = target_timestamp - current_time
            if sleep_time > 0:
                time.sleep(min(sleep_time, 0.002))
            else:
                break

        # Read current frame
        psm1_joints = df.loc[frame_idx, psm1_joint_columns].values.astype(np.float32)
        psm2_joints = df.loc[frame_idx, psm2_joint_columns].values.astype(np.float32)
        psm1_jaw = float(df.loc[frame_idx, psm1_jaw_column])
        psm2_jaw = float(df.loc[frame_idx, psm2_jaw_column])
        psm1_gripper1 = -psm1_jaw / 2
        psm1_gripper2 = psm1_jaw / 2
        psm2_gripper1 = -psm2_jaw / 2
        psm2_gripper2 = psm2_jaw / 2

        action = np.concatenate([
            psm1_joints, [psm1_gripper1, psm1_gripper2],
            psm2_joints, [psm2_gripper1, psm2_gripper2]
        ], dtype=np.float32)

        action = torch.tensor(action, device=env.unwrapped.device).unsqueeze(0)

        for _ in range(3):  # repeat for better tracking
            env.step(action)
            time.sleep(0.0)

    print("[INFO] Playback finished.")
    env.close()



if __name__ == "__main__":
    main()
    simulation_app.close()
