import argparse

from omni.isaac.lab.app import AppLauncher

parser = argparse.ArgumentParser(description="MTMR + ACT teleoperation for PSM2 and PSM1")
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--task", type=str, default="Isaac-MTML-ACT-v0")
parser.add_argument("--scale", type=float, default=0.4)
parser.add_argument("--is_simulated", type=bool, default=False)
parser.add_argument("--enable_logging", action="store_true")
parser.add_argument("--log_trigger_file", type=str, default="log_trigger.txt")
parser.add_argument("--model_trigger_file", type=str, default="model_trigger.txt")
parser.add_argument("--disable_viewport", action="store_true")
parser.add_argument("--demo_name", type=str, default=None)
parser.add_argument("--model_ckpt", type=str, default="Models/4_orbitsim_single_human_demos/Joint Control/20250516-053119_kind-opossum_train/policy_epoch_10000_seed_0.ckpt")
parser.add_argument("--ckpt_path", type=str, default="Models/4_orbitsim_single_human_demos/Joint Control/20250516-053119_kind-opossum_train/policy_epoch_10000_seed_0.ckpt")
parser.add_argument("--ckpt_strategy", type=str, default="none", choices=["best", "last", "none"])

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import sys
import os

sys.path.append(os.path.abspath("."))

import gymnasium as gym
import torch
import numpy as np
import time
from scipy.spatial.transform import Rotation as R
import omni.kit.viewport.utility as vp_utils

import custom_envs
from omni.isaac.lab_tasks.utils import parse_env_cfg
from tf_utils import pose_to_transformation_matrix, transformation_matrix_to_pose
from teleop_interface.MTM.se3_mtm import MTMTeleop
from teleop_interface.MTM.mtm_manipulator import MTMManipulator
from AdaptACT.procedures.controller import AutonomousController
from teleop_logger import TeleopLogger, log_current_pose


def main():
    scale = args_cli.scale

    # Load environment
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    env_cfg.terminations.time_out = None
    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()

    # Load MTM
    mtm_manipulator = MTMManipulator(); mtm_manipulator.home()
    mtm_interface = MTMTeleop(); mtm_interface.reset()

    # Load ACT controller
    controller = None
    if args_cli.ckpt_path and os.path.exists(args_cli.ckpt_path):
        try:
            controller = AutonomousController.from_train_dir(
                train_dir=args_cli.model_ckpt,
                ckpt_strategy=args_cli.ckpt_strategy,
                ckpt_path=args_cli.ckpt_path,
                rollout_len=500,
                device="cuda:0"
            )
            controller.reset()
            print(f"[INFO] Successfully loaded ACT model from: {args_cli.ckpt_path}")
        except Exception as e:
            print(f"[ERROR] Failed to load ACT model: {e}")
    else:
        print(f"[WARNING] Checkpoint file not found or not specified: {args_cli.ckpt_path}")

    # Setup camera & robot
    camera_l = env.unwrapped.scene["camera_left"]
    camera_r = env.unwrapped.scene["camera_right"]
    psm1 = env.unwrapped.scene["robot_1"]  # ACT
    psm2 = env.unwrapped.scene["robot_2"]  # MTMR

    if not args_cli.disable_viewport:
        vp_l = vp_utils.create_viewport_window(name="Left Camera", width=800, height=600)
        vp_l.viewport_api.camera_path = '/World/envs/env_0/Robot_4/ecm_end_link/camera_left'
        vp_r = vp_utils.create_viewport_window(name="Right Camera", width=800, height=600)
        vp_r.viewport_api.camera_path = '/World/envs/env_0/Robot_4/ecm_end_link/camera_right'

    # Logging
    teleop_logger = TeleopLogger(trigger_file=args_cli.log_trigger_file, psm_name_dict={"PSM1": "robot_1", "PSM2": "robot_2"}, log_duration=30.0)

    # Frame transform setup
    cam_pos = (camera_l.data.pos_w + camera_r.data.pos_w) / 2
    cam_quat = camera_l.data.quat_w_world
    world_T_cam = pose_to_transformation_matrix(cam_pos.cpu().numpy()[0], cam_quat.cpu().numpy()[0])
    cam_T_world = np.linalg.inv(world_T_cam)

    model_active = False

    while simulation_app.is_running():
        start_time = time.time()

        # Trigger model if requested
        if not model_active and os.path.exists(args_cli.model_trigger_file):
            print("[MODEL] Trigger detected. Enabling ACT for PSM1.")
            model_active = True
            if controller:
                controller.reset()
            os.remove(args_cli.model_trigger_file)

        # Fetch ACT inputs
        imgs = np.stack([
            camera_l.data.output["rgb"][0].cpu().numpy(),
            camera_r.data.output["rgb"][0].cpu().numpy()
        ]).astype(np.float32) / 255.0
        qpos_psm1 = psm1.data.joint_pos[0][:6].cpu().numpy()

        # ACT output for PSM1
        if model_active and controller is not None:
            joint_cmd_psm1 = controller.step(imgs, qpos_psm1)
        else:
            joint_cmd_psm1 = qpos_psm1.copy()

        # MTMR control for PSM2
        _, _, _, mtmr_pos, mtmr_rot, r_gripper_joint, mtm_clutch, _, _ = mtm_interface.advance()
        tip_pose = pose_to_transformation_matrix(
            psm2.data.body_link_pos_w[0][-1].cpu().numpy(),
            psm2.data.body_link_quat_w[0][-1].cpu().numpy()
        )
        if not mtm_clutch:
            mtmr_quat = R.from_rotvec(mtmr_rot).as_quat()
            mtmr_quat = np.concatenate([[mtmr_quat[3]], mtmr_quat[:3]])
            psm2_target = tip_pose[:3, 3] + (mtmr_pos * scale)
            cam_T_psm2tip = pose_to_transformation_matrix(psm2_target, mtmr_quat)
        else:
            cam_T_psm2tip = cam_T_world @ tip_pose

        psm2_base = pose_to_transformation_matrix(
            psm2.data.body_link_pos_w[0][0].cpu().numpy(),
            psm2.data.body_link_quat_w[0][0].cpu().numpy()
        )
        p2_pos, p2_quat = transformation_matrix_to_pose(np.linalg.inv(psm2_base) @ cam_T_world @ cam_T_psm2tip)

        # Gripper control
        jaw1 = [0.0, 0.0]  # PSM1 (ACT)
        jaw2 = [-0.0, 0.0] if r_gripper_joint is None else np.interp(r_gripper_joint, [-1.0, 1.0], [-0.523, 0.523]) * np.array([-1, 1])

        action = torch.tensor(
            np.concatenate([joint_cmd_psm1, jaw1, p2_pos, p2_quat, jaw2]),
            dtype=torch.float32, device=env.unwrapped.device
        ).unsqueeze(0)

        env.step(action)

        # Logging
        teleop_logger.check_and_start_logging(env)
        if teleop_logger.enable_logging and teleop_logger.frame_num == 0:
            log_current_pose(env, teleop_logger.log_dir)
        teleop_logger.stop_logging()

        if teleop_logger.enable_logging:
            frame_num = teleop_logger.frame_num + 1
            timestamp = time.time()
            robot_states = {}
            for name in ["robot_1", "robot_2"]:
                robot = env.unwrapped.scene[name]
                joint_positions = robot.data.joint_pos[0][:6].cpu().numpy()
                jaw_angle = abs(robot.data.joint_pos[0][-2].cpu().numpy()) + abs(robot.data.joint_pos[0][-1].cpu().numpy())
                ee_position = robot.data.body_link_pos_w[0][-1].cpu().numpy()
                ee_quat = robot.data.body_link_quat_w[0][-1].cpu().numpy()
                orientation_matrix = R.from_quat(np.concatenate([ee_quat[1:], [ee_quat[0]]])).as_matrix()
                robot_states[name] = dict(joint_positions=joint_positions, jaw_angle=jaw_angle, ee_position=ee_position, orientation_matrix=orientation_matrix)

            teleop_logger.enqueue(frame_num, timestamp, robot_states,
                                  camera_l.data.output["rgb"][0].cpu().numpy(),
                                  camera_r.data.output["rgb"][0].cpu().numpy())
            teleop_logger.frame_num = frame_num

        time.sleep(max(0.0, (1/200.0) - (time.time() - start_time)))

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
