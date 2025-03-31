import argparse

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Example script to show how to use camera outputs.")
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

import cv2
import numpy as np

import sys
import os
sys.path.append(os.path.abspath("."))
import custom_envs


def main():
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
            actions[0][7] = -actions[0][0]
        else:
            actions[0][0] = -0.5 + 0.5 * ((i - steps / 2) / (steps / 2))
            actions[0][7] = -actions[0][0]

        actions = torch.tensor(actions, device=env.unwrapped.device)
        obs, _, terminated, truncated, _ = env.step(actions)

        cam_l_input = camera_l.data.output["rgb"][0].cpu().numpy()
        cam_l_input_bgr = cv2.cvtColor(cam_l_input, cv2.COLOR_RGB2BGR)
        cam_r_input = camera_r.data.output["rgb"][0].cpu().numpy()
        cam_r_input_bgr = cv2.cvtColor(cam_r_input, cv2.COLOR_RGB2BGR)

        # concatenate the two images and save
        cam_input_bgr = np.concatenate((cam_l_input_bgr, cam_r_input_bgr), axis=1)
        out.write(cam_input_bgr)

    # Release the video writer
    out.release()
    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
