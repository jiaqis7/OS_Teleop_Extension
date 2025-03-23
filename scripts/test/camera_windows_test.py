import argparse

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Test Script to build seperate windows only for camera outputs.")
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

import omni.kit.viewport.utility as vp_utils
import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils import parse_env_cfg

import orbit.surgical.tasks  # noqa: F401

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

    view_port_l = vp_utils.create_viewport_window("Left Camera", width = 700, height = 500)
    view_port_l.viewport_api.camera_path = '/World/envs/env_0/Robot_4/ecm_end_link/camera_left' #camera_l.cfg.prim_path
    view_port_l.viewport_api.resolution = (camera_l.cfg.width, camera_l.cfg.height)

    view_port_r = vp_utils.create_viewport_window("Right Camera", width = 700, height = 500)
    view_port_r.viewport_api.camera_path = '/World/envs/env_0/Robot_4/ecm_end_link/camera_right' #camera_r.cfg.prim_path
    view_port_r.viewport_api.resolution = (camera_r.cfg.width, camera_r.cfg.height)

    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # compute zero actions
            actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            # apply actions
            env.step(actions)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
