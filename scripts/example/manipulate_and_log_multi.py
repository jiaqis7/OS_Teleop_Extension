# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Example script to show how to log state values retrieved from the Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-MultiArm-dVRK-v0", help="Name of the task.")
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

import csv
import os
import numpy as np

import sys
import os
sys.path.append(os.path.abspath("."))
import custom_envs

def get_robot_states(env, robot_name):
    sim_env = env.unwrapped
    # Get the robot from the scene configuration
    robot = sim_env.scene[robot_name]  # This should be directly accessible
    joint_pos = robot.data.joint_pos[0].cpu().numpy()
    eef_pos = robot.data.body_link_pos_w[0][-1].cpu().numpy()
    eef_quat = robot.data.body_link_quat_w[0][-1].cpu().numpy()  # (w, x, y, z)

    return joint_pos, eef_pos, eef_quat


def main():
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

    # Get robots from the scene configuration
    robots = [f"robot_{i}" for i in range(1, 4)]
    joint_limits = [env.unwrapped.scene[robot].data.joint_limits[0] for robot in robots]
    joint_min = [limits[:, 0] for limits in joint_limits]
    joint_max = [limits[:, 1] for limits in joint_limits]

    # CSV file setup
    log_file_path = "simulation_log.csv"
    with open(log_file_path, mode="w", newline="") as log_file:
        csv_writer = csv.writer(log_file)
        # Write header
        header = ["Timestep", "Simulation Time"]
        for robot, limits in zip(robots, joint_limits):
            header += [f"{robot}_Joint_Pos_{j}" for j in range(1, limits.shape[0] + 1)]
            header += [
                f"{robot}_End_Effector_Pos_X",
                f"{robot}_End_Effector_Pos_Y",
                f"{robot}_End_Effector_Pos_Z",
                f"{robot}_End_Effector_Quat_W",
                f"{robot}_End_Effector_Quat_X",
                f"{robot}_End_Effector_Quat_Y",
                f"{robot}_End_Effector_Quat_Z",
            ]
        csv_writer.writerow(header)

        steps = 500
        count = 0

        for i in range(steps):
            with torch.inference_mode():
                actions = np.zeros(env.action_space.shape)
                # Move the first joint of robot_1 from 0 to min, back to 0, to max, and back to 0
                if i < steps / 4:
                    actions[0][0] = 0 + (joint_min[0][0] - 0) * (i / (steps / 4))
                elif i < steps / 2:
                    actions[0][0] = joint_min[0][0] + (0 - joint_min[0][0]) * ((i - steps / 4) / (steps / 4))
                elif i < 3 * steps / 4:
                    actions[0][0] = 0 + (joint_max[0][0] - 0) * ((i - steps / 2) / (steps / 4))
                else:
                    actions[0][0] = joint_max[0][0] + (0 - joint_max[0][0]) * ((i - 3 * steps / 4) / (steps / 4))

                actions = torch.tensor(actions, device=env.unwrapped.device)
                env.step(actions)
                states = [get_robot_states(env, robot) for robot in robots]

                # Write data grouped by robot
                row = [count, env.sim.current_time]
                for state in states:
                    row += list(state[0])  # Joint positions
                    row += list(state[1])  # End effector positions (x, y, z)
                    row += list(state[2])  # End effector quaternion (w, x, y, z)
                csv_writer.writerow(row)

                count += 1

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
