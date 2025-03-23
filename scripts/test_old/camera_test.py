# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run an environment with zero action agent."""

"""Launch Isaac Sim Simulator first."""

import argparse

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Zero agent for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-PO-Teleop-v0", help="Name of the task.")
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

import cv2
import numpy as np
import matplotlib.pyplot as plt


def main():
    """Zero actions agent with Isaac Lab environment."""
    # parse configuration
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
    camera_l = env.unwrapped.scene["camera_left"]
    camera_r = env.unwrapped.scene["camera_right"]
    joint_limits = env.unwrapped.scene["robot_4"].data.joint_limits[0]
    joint_min = joint_limits[:, 0]
    joint_max = joint_limits[:, 1]

    # Video writer setup
    frame_width = camera_l.data.output["rgb"][0].shape[1]
    frame_height = camera_l.data.output["rgb"][0].shape[0]
    out = cv2.VideoWriter("output.avi", cv2.VideoWriter_fourcc(*"XVID"), 30, (2 * frame_width, frame_height))

    # simulate environment
    steps = 300
    for i in range(steps):
        actions = np.zeros(env.action_space.shape)
        if i < steps / 2:
            actions[0][0] = -0.5 * (i / (steps / 2))
            actions[0][7] = 0.5 * (i / (steps / 2))
            actions[0][15] = actions[0][0]
        else:
            actions[0][0] = -0.5 + 0.5 * ((i - steps / 2) / (steps / 2))
            actions[0][7] = -actions[0][0]
            actions[0][15] = actions[0][0]

        actions = torch.tensor(actions, device=env.unwrapped.device)
        obs, _, terminated, truncated, _ = env.step(actions)

        cam_l_input = camera_l.data.output["rgb"][0].cpu().numpy()
        cam_l_input_bgr = cv2.cvtColor(cam_l_input, cv2.COLOR_RGB2BGR)
        cam_r_input = camera_r.data.output["rgb"][0].cpu().numpy()
        cam_r_input_bgr = cv2.cvtColor(cam_r_input, cv2.COLOR_RGB2BGR)

        # concatenate the two images and save
        cam_input_bgr = np.concatenate((cam_l_input_bgr, cam_r_input_bgr), axis=1)
        # cv2.imshow("Camera", cam_input_bgr)
        # cv2.waitKey(1)
        out.write(cam_input_bgr)

    for i in range(steps):
        actions = np.zeros(env.action_space.shape)
        if i < steps / 2:
            actions[0][22] = 0.5 * (i / (steps / 2))
        else:
            actions[0][22] = 0.5 - 0.5 * ((i - steps / 2) / (steps / 2))

        actions = torch.tensor(actions, device=env.unwrapped.device)
        obs, _, terminated, truncated, _ = env.step(actions)

        cam_l_input = camera_l.data.output["rgb"][0].cpu().numpy()
        cam_l_input_bgr = cv2.cvtColor(cam_l_input, cv2.COLOR_RGB2BGR)
        cam_r_input = camera_r.data.output["rgb"][0].cpu().numpy()
        cam_r_input_bgr = cv2.cvtColor(cam_r_input, cv2.COLOR_RGB2BGR)

        # concatenate the two images and save
        cam_input_bgr = np.concatenate((cam_l_input_bgr, cam_r_input_bgr), axis=1)
        # cv2.imshow("Camera", cam_input_bgr)
        # cv2.waitKey(1)
        out.write(cam_input_bgr)
    # for j in range(12, num_joints):
    #     for i in range(steps):
    #         # run everything in inference mode
    #         with torch.inference_mode():
    #             actions = np.zeros(env.action_space.shape)
    #             if j == 14:
    #                 # For the third joint, move from min to max and back to min
    #                 if i < steps / 2:
    #                     actions[0][j] = joint_min[j - 12] + (joint_max[j - 12] - joint_min[j - 12]) * (i / (steps / 2))
    #                 else:
    #                     actions[0][j] = joint_max[j - 12] - (joint_max[j - 12] - joint_min[j - 12]) * (
    #                         (i - steps / 2) / (steps / 2)
    #                     )
    #             else:
    #                 # For other joints, move from 0 to min to max to 0
    #                 if i < steps / 4:
    #                     actions[0][j] = 0 + (joint_min[j - 12] - 0) * (i / (steps / 4))
    #                 elif i < steps / 2:
    #                     actions[0][j] = joint_min[j - 12] + (joint_max[j - 12] - joint_min[j - 12]) * (
    #                         (i - steps / 4) / (steps / 4)
    #                     )
    #                 elif i < 3 * steps / 4:
    #                     actions[0][j] = joint_max[j - 12] - (joint_max[j - 12] - 0) * ((i - steps / 2) / (steps / 4))
    #                 else:
    #                     actions[0][j] = 0

    #             actions = torch.tensor(actions, device=env.unwrapped.device)
    #             obs, _, terminated, truncated, _ = env.step(actions)
    #             if terminated:
    #                 print("Terminated")
    #             if truncated:
    #                 print("Truncated")

                # cam_l_input = camera_l.data.output["rgb"][0].cpu().numpy()
                # cam_l_input_bgr = cv2.cvtColor(cam_l_input, cv2.COLOR_RGB2BGR)
                # cam_r_input = camera_r.data.output["rgb"][0].cpu().numpy()
                # cam_r_input_bgr = cv2.cvtColor(cam_r_input, cv2.COLOR_RGB2BGR)

                # # concatenate the two images and save
                # cam_input_bgr = np.concatenate((cam_l_input_bgr, cam_r_input_bgr), axis=1)
                # # cv2.imshow("Camera", cam_input_bgr)
                # # cv2.waitKey(1)
                # out.write(cam_input_bgr)

    # Release the video writer
    out.release()
    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
