# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to an environment with random action agent."""

"""Launch Isaac Sim Simulator first."""

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
import numpy as np
import gymnasium as gym
import torch

import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils import parse_env_cfg

from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.utils.stage import get_current_stage
from omni.isaac.core.utils.prims import get_prim_at_path

import orbit.surgical.tasks  # noqa: F401


def get_end_effector_pose(env, end_effector_name):
    sim_env = env.unwrapped
    # Get the robot from the scene configuration
    robot = sim_env.scene["robot"]  # This should be directly accessible
    # print("Joint Pose from SIM")
    # print(robot.data.joint_pos[0])
    print("Base Link Position")
    print(robot.data.body_link_pos_w[0][0])
    print("Base Link Orientation")
    print(robot.data.body_link_quat_w[0][0])
    # print("Tip Body Link Position")
    # print(robot.data.body_link_pos_w[0][-1])
    # print("Tip Body Link Orientation")
    # print(robot.data.body_link_quat_w[0][-1])

    # # Get the index of the end-effector link
    # eef_idx = robot.get_dof_index(end_effector_name)
    # if eef_idx is None:
    #     raise ValueError(f"End-effector '{end_effector_name}' not found in: {robot.get_link_names()}")

    # # Get position and orientation (quaternion)
    # eef_pos = robot.get_link_world_pose(eef_idx).p  # (x, y, z) in world frame
    # eef_quat = robot.get_link_world_pose(eef_idx).r  # (qx, qy, qz, qw)

    # return eef_pos, eef_quat


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
    joint_limits = env.unwrapped.scene["robot"].data.joint_limits[0]
    joint_min = joint_limits[:, 0]
    joint_max = joint_limits[:, 1]
    print("joint_min:", joint_min)
    print("joint_max:", joint_max)

    # simulate environment
    steps = 500
    # check only for the last two joints
    num_joints = env.action_space.shape[1]
    for j in range(num_joints):
        # for j in range(6, 8):
        for i in range(steps):
            # run everything in inference mode
            with torch.inference_mode():
                actions = np.zeros(env.action_space.shape)
                actions[0][2] = 0.3
                if j == 2:
                    break
                    # For the third joint, move from min to max and back to min
                    # if i < steps / 2:
                    #     actions[0][j] = joint_min[j] + (joint_max[j] - joint_min[j]) * (i / (steps / 2))
                    # else:
                    #     actions[0][j] = joint_max[j] - (joint_max[j] - joint_min[j]) * ((i - steps / 2) / (steps / 2))
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
                get_end_effector_pose(env, "psm_tool_tip_link")

    # while simulation_app.is_running():
    #     # run everything in inference mode
    #     with torch.inference_mode():
    #         #     # sample actions from -1 to 1
    #         #     actions = 2 * torch.rand(env.action_space.shape, device=env.unwrapped.device) - 1
    #         # apply actions
    #         actions = torch.tensor(
    #             [[0.0, 0.0, 0.1, 0.0, 0.0, 0.0, -0.09, 0.09]], device=env.unwrapped.device
    #         )  # JointPos
    #         #     actions = torch.tensor(
    #         #         [[0.0573, -0.0009, 0.0946 - 0.15, 0.9188, -0.0070, -0.3945, -0.0084]], device=env.unwrapped.device
    #         #     ) # IK_ABS
    #         #     actions = torch.tensor([[0.02, 0.0, 0.0, 0, 0, 0.1]], device=env.unwrapped.device)  # IK_REL
    #         env.step(actions)
    #         #     if terminated:
    #         #         print("Terminated")
    #         #     if truncated:
    #         #         print("Truncated")
    #         get_end_effector_pose(env, "psm_tool_tip_link")
    #     #     print("End Effector Position:", end_effector_pose[0])
    #     #     print("End Effector Orientation (Quaternion):", end_effector_pose[1])

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
