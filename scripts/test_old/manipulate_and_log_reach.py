# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Random agent for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
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
import csv
import os


def main():
    """Random actions agent with Isaac Lab environment."""
    # create environment configuration
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
    # simulate environment
    count = 0
    # direction = 1

    def profile(val):
        if val < 49:
            return (val + 1) / 100.0 + 0.5
        if val < 149:
            return -val / 100.0 + 1.49
        return val / 100.0 - 1.49

    # CSV file setup
    log_file_path = "simulation_log.csv"
    with open(log_file_path, mode="w", newline="") as log_file:
        csv_writer = csv.writer(log_file)
        # Write header
        csv_writer.writerow(
            ["Timestep"] + [f"Joint_Pos_{i+1}" for i in range(6)] + [f"Last_Action_{i+1}" for i in range(6)]
        )
        while simulation_app.is_running():
            with torch.inference_mode():
                # actions = 2 * torch.tensor([[0.5, 0.5, count / 100.0, 0.5, 0.5, 0.5]]) - 1
                actions = (
                    2 * torch.tensor([[profile(count % 200) if i == int(count / 200) else 0.5 for i in range(6)]]) - 1
                )
                if count % 200 == 0:
                    print("Start moving joint at ", count / 200, " with ", actions)
                # if (count == 99 and direction == 1) or (count == 1 and direction == -1):
                #     direction *= -1

                # Apply actions
                obs, _, terminated, truncated, _ = env.step(actions)

                # Extract relevant parts of observation
                joint_positions = obs["policy"][:8]  # First 8 elements -> relative to the initial point
                # print(joint_positions)
                last_actions = obs["policy"][-6:]  # Last 6 elements

                # Write timestep, joint positions, and last actions to CSV
                csv_writer.writerow([count] + joint_positions.tolist() + last_actions.tolist())

                count += 1

                if count / 200 > 6:
                    break

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
