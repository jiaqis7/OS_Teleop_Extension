import argparse

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Example script to show how to move robots and retrieve state values in the Isaac Sim")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-CustomTest-v1", help="Name of the task.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""
import numpy as np
import gymnasium as gym
import torch

from omni.isaac.lab_tasks.utils import parse_env_cfg

import sys
import os
sys.path.append(os.path.abspath("."))
import custom_envs

def print_states(env, robot_name):
    sim_env = env.unwrapped
    # Get the robot from the scene configuration
    robot = sim_env.scene[robot_name] 
    print("Joint Pose from SIM")
    print(robot.data.joint_pos[0])
    print("Base Link Position")
    print(robot.data.body_link_pos_w[0][0])
    print("Base Link Orientation")
    print(robot.data.body_link_quat_w[0][0])
    print("Tip Body Link Position")
    print(robot.data.body_link_pos_w[0][-1])
    print("Tip Body Link Orientation")
    print(robot.data.body_link_quat_w[0][-1])


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
    joint_limits = env.unwrapped.scene["robot_1"].data.joint_limits[0]
    joint_min = joint_limits[:, 0]
    joint_max = joint_limits[:, 1]
    print("joint_min:", joint_min)
    print("joint_max:", joint_max)

    # simulate environment
    steps = 200
    for j in range(6):
        for i in range(steps):
            # run everything in inference mode
            with torch.inference_mode():
                actions = np.zeros(env.action_space.shape)
                if j == 2:
                    # For the third joint, move from min to max and back to min
                    if i < steps / 2:
                        actions[0][j] = joint_min[j] + (joint_max[j] - joint_min[j]) * (i / (steps / 2))
                    else:
                        actions[0][j] = joint_max[j] - (joint_max[j] - joint_min[j]) * ((i - steps / 2) / (steps / 2))
                else:
                    # For other joints, move from 0 to min to max to 0
                    if i < steps / 4:
                        actions[0][j] = 0 + (joint_min[j] - 0) * (i / (steps / 4))
                    elif i < steps / 2:
                        actions[0][j] = joint_min[j] + (joint_max[j] - joint_min[j]) * ((i - steps / 4) / (steps / 4))
                    elif i < 3 * steps / 4:
                        actions[0][j] = joint_max[j] - (joint_max[j] - 0) * ((i - steps / 2) / (steps / 4))
                    else:
                        actions[0][j] = 0

                actions = torch.tensor(actions, device=env.unwrapped.device)
                env.step(actions)
                print_states(env, "robot_1")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
