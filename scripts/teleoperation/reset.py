import argparse
from omni.isaac.lab.app import AppLauncher  # << Launch app first!

# Step 1: CLI arguments
parser = argparse.ArgumentParser(description="Reset PSMs to starting poses without closing the sim.")
parser.add_argument("--task", type=str, default="Isaac-MTM-Teleop-v0", help="Task name.")
parser.add_argument("--device", type=str, default="cuda:0", help="Compute device.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
parser.add_argument("--disable_fabric", action="store_true", help="Disable fabric.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Step 2: Launch Isaac Sim App
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Step 3: Now import heavy libraries
import torch
import gymnasium as gym
import time
import numpy as np
from omni.isaac.lab_tasks.utils import parse_env_cfg
from tf_utils import pose_to_transformation_matrix, transformation_matrix_to_pose

def main():
    parser = argparse.ArgumentParser(description="Reset PSMs to starting poses without closing the sim.")
    parser.add_argument("--task", type=str, default="Isaac-MTM-Teleop-v0", help="Task name.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Compute device.")
    parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
    parser.add_argument("--disable_fabric", action="store_true", help="Disable fabric.")
    args = parser.parse_args()

    # Parse the environment configuration
    env_cfg = parse_env_cfg(
        args.task, device=args.device, num_envs=args.num_envs, use_fabric=not args.disable_fabric
    )
    env_cfg.terminations.time_out = None

    # Connect to the running simulation
    env = gym.make(args.task, cfg=env_cfg)
    env.reset()

    print("[RESET] Moving PSMs back to initial poses...")

    # Initial pose and orientation from your MTMTeleopEnvCfg
    psm1_init_pos = np.array([0.1, 0.0, 0.1])
    psm1_init_quat = np.array([0.9659, 0.0, 0.2588, 0.0])

    psm2_init_pos = np.array([-0.1, 0.0, 0.1])
    psm2_init_quat = np.array([0.9659, 0.0, -0.2588, 0.0])

    # Compose transformation matrices
    cam_T_psm1tip = pose_to_transformation_matrix(psm1_init_pos, psm1_init_quat)
    cam_T_psm2tip = pose_to_transformation_matrix(psm2_init_pos, psm2_init_quat)

    # Get world transforms of base frames
    psm1 = env.unwrapped.scene["robot_1"]
    psm2 = env.unwrapped.scene["robot_2"]

    psm1_base_link_pos = psm1.data.body_link_pos_w[0][0].cpu().numpy()
    psm1_base_link_quat = psm1.data.body_link_quat_w[0][0].cpu().numpy()
    world_T_psm1_base = pose_to_transformation_matrix(psm1_base_link_pos, psm1_base_link_quat)

    psm2_base_link_pos = psm2.data.body_link_pos_w[0][0].cpu().numpy()
    psm2_base_link_quat = psm2.data.body_link_quat_w[0][0].cpu().numpy()
    world_T_psm2_base = pose_to_transformation_matrix(psm2_base_link_pos, psm2_base_link_quat)

    # You can assume world_T_cam = Identity during reset
    world_T_cam = np.eye(4)

    # Process action
    def process_reset_action(cam_T_psm1tip, w_T_psm1base, cam_T_psm2tip, w_T_psm2base):
        psm1base_T_psm1 = np.linalg.inv(w_T_psm1base) @ world_T_cam @ cam_T_psm1tip
        psm2base_T_psm2 = np.linalg.inv(w_T_psm2base) @ world_T_cam @ cam_T_psm2tip
        psm1_rel_pos, psm1_rel_quat = transformation_matrix_to_pose(psm1base_T_psm1)
        psm2_rel_pos, psm2_rel_quat = transformation_matrix_to_pose(psm2base_T_psm2)

        # Gripper open default: ~0.5
        gripper_open = 0.5
        gripper1 = np.array([-gripper_open, gripper_open])
        gripper2 = np.array([-gripper_open, gripper_open])

        actions = np.concatenate([
            psm1_rel_pos, psm1_rel_quat, gripper1,
            psm2_rel_pos, psm2_rel_quat, gripper2
        ])
        actions = torch.tensor(actions, device=env.unwrapped.device).repeat(env.unwrapped.num_envs, 1)
        return actions

    # Generate the action
    actions = process_reset_action(cam_T_psm1tip, world_T_psm1_base, cam_T_psm2tip, world_T_psm2_base)

    # Execute several steps to converge to target pose
    for _ in range(10):
        env.step(actions)
        time.sleep(0.05)

    print("[RESET] PSMs moved to initial pose successfully.")
    print("[RESET] You can now clutch again to start new teleoperation.")

    # DO NOT close simulation here — simply exit
    # (simulation_app.close() is NOT called)

if __name__ == "__main__":
    main()

