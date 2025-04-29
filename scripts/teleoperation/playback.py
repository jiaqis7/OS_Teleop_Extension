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

TARGET_FREQUENCY = 15  # Hz
TARGET_DT = 1.0 / TARGET_FREQUENCY  # 0.1 seconds


def main():
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()

    # Read CSV
    df = pd.read_csv(args_cli.csv_file)

    psm1_joint_columns = [f"PSM1_joint_{i}" for i in range(1, 7)]
    psm2_joint_columns = [f"PSM2_joint_{i}" for i in range(1, 7)]
    psm1_jaw_column = "PSM1_jaw_angle"
    psm2_jaw_column = "PSM2_jaw_angle"

    print(f"[INFO] Start playing back {len(df)} frames at {TARGET_FREQUENCY}Hz...")

    for frame_idx in range(len(df)):
        start_time = time.time()

        # Read frame
        psm1_joints = df.loc[frame_idx, psm1_joint_columns].values
        psm2_joints = df.loc[frame_idx, psm2_joint_columns].values
        psm1_jaw = df.loc[frame_idx, psm1_jaw_column]
        psm2_jaw = df.loc[frame_idx, psm2_jaw_column]

        psm1_gripper1 = -psm1_jaw / 2
        psm1_gripper2 = psm1_jaw / 2
        psm2_gripper1 = -psm2_jaw / 2
        psm2_gripper2 = psm2_jaw / 2

        # Generate action
        action = np.concatenate([
            psm1_joints, [psm1_gripper1, psm1_gripper2],
            psm2_joints, [psm2_gripper1, psm2_gripper2]
        ])
        action = np.array(action).astype(np.float32)
        action = torch.tensor(action, device=env.unwrapped.device).unsqueeze(0)

        # Step simulation
        env.step(action)

        # Sleep to maintain 10 Hz
        elapsed = time.time() - start_time
        if elapsed < TARGET_DT:
            time.sleep(TARGET_DT - elapsed)

    print("[INFO] Playback finished. Stopping simulation.")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
