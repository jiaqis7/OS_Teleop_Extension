import argparse

from omni.isaac.lab.app import AppLauncher
from teleop_logger import TeleopLogger, log_current_pose

# add argparse arguments
parser = argparse.ArgumentParser(description="MTMR + PO teleoperation for PSM2 and PSM1")
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--task", type=str, default="Isaac-MTM-Teleop-v0")
parser.add_argument("--scale", type=float, default=0.4)
parser.add_argument("--is_simulated", type=bool, default=False)
parser.add_argument("--enable_logging", action="store_true")
parser.add_argument("--log_trigger_file", type=str, default="log_trigger.txt")
parser.add_argument("--disable_viewport", action="store_true")
parser.add_argument("--demo_name", type=str, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
import carb
from omni.isaac.lab_tasks.utils import parse_env_cfg
import numpy as np
import time
from scipy.spatial.transform import Rotation as R
import omni.kit.viewport.utility as vp_utils
import sys
import os

sys.path.append(os.path.abspath("."))

from tf_utils import pose_to_transformation_matrix, transformation_matrix_to_pose
from teleop_interface.MTM.se3_mtm import MTMTeleop
from teleop_interface.phantomomni.se3_phantomomni import PhantomOmniTeleop
from teleop_interface.MTM.mtm_manipulator import MTMManipulator
import custom_envs



# ---------- Init ----------
psm_name_dict = {
    "robot_1": "PSM1_PO",
    "robot_2": "PSM2_MTM"
}
teleop_logger = TeleopLogger(trigger_file=args_cli.log_trigger_file, psm_name_dict=psm_name_dict)



def get_jaw_gripper_angles(gripper_command, robot_name):
    if gripper_command is None:
        return np.array([0.0, 0.0])
    g2_angle = 0.52359 / (1.06 + 1.72) * (gripper_command + 1.72)
    return np.array([-g2_angle, g2_angle])


def process_actions(cam_T_psm1, w_T_psm1base, cam_T_psm2, w_T_psm2base, w_T_cam, env, gripper1_command, gripper2_command):
    psm1base_T_psm1 = np.linalg.inv(w_T_psm1base) @ w_T_cam @ cam_T_psm1
    psm2base_T_psm2 = np.linalg.inv(w_T_psm2base) @ w_T_cam @ cam_T_psm2
    psm1_rel_pos, psm1_rel_quat = transformation_matrix_to_pose(psm1base_T_psm1)
    psm2_rel_pos, psm2_rel_quat = transformation_matrix_to_pose(psm2base_T_psm2)
    actions = np.concatenate([
        psm1_rel_pos, psm1_rel_quat, get_jaw_gripper_angles(gripper1_command, 'robot_1'),
        psm2_rel_pos, psm2_rel_quat, get_jaw_gripper_angles(gripper2_command, 'robot_2')
    ])
    return torch.tensor(actions, device=env.unwrapped.device).repeat(env.unwrapped.num_envs, 1)


def align_mtm_orientation_once(mtm_manipulator, mtm_interface, psm2, cam_T_world):
    print("[ALIGN] Aligning MTMR orientation with PSM2 pose...")
    # Get world-space PSM2 tip pose
    psm2_tip_pose = pose_to_transformation_matrix(
        psm2.data.body_link_pos_w[0][-1].cpu().numpy(),
        psm2.data.body_link_quat_w[0][-1].cpu().numpy()
    )

    # Convert to HRSV frame
    hrsv_T_mtmr = mtm_interface.simpose2hrsvpose(cam_T_world @ psm2_tip_pose)

    # Set orientation using this properly transformed matrix
    mtm_manipulator.adjust_orientation_right(hrsv_T_mtmr[:3, :3])

    print("[ALIGN] MTMR orientation set.")



def main():
    scale = args_cli.scale

    mtm_manipulator = MTMManipulator()
    mtm_manipulator.home()

    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    env_cfg.terminations.time_out = None
    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()

    mtm_interface = MTMTeleop()
    mtm_interface.reset()
    po_interface = PhantomOmniTeleop()
    po_interface.reset()

    camera_l = env.unwrapped.scene["camera_left"]
    camera_r = env.unwrapped.scene["camera_right"]

    if not args_cli.disable_viewport:
        view_port_l = vp_utils.create_viewport_window("Left Camera", width=800, height=600)
        view_port_l.viewport_api.camera_path = '/World/envs/env_0/Robot_4/ecm_end_link/camera_left'

        view_port_r = vp_utils.create_viewport_window("Right Camera", width=800, height=600)
        view_port_r.viewport_api.camera_path = '/World/envs/env_0/Robot_4/ecm_end_link/camera_right'

    psm1 = env.unwrapped.scene["robot_1"]
    psm2 = env.unwrapped.scene["robot_2"]

    was_in_mtm_clutch = True
    was_in_po_clutch = True
    init_psm1_tip_position = None
    init_psm2_tip_position = None
    init_mtmr_position = None
    init_stylus_position = None
    orientation_aligned = False

    psm_name_dict = {
        "PSM1": "robot_1",
        "PSM2": "robot_2"
    }

    teleop_logger = TeleopLogger(
        trigger_file="log_trigger.txt",
        psm_name_dict=psm_name_dict,
        log_duration=30.0,
    )

    print("Press the clutch button and release to start teleoperation.")
    while simulation_app.is_running():
        start_time = time.time()

        cam_pos = (camera_l.data.pos_w + camera_r.data.pos_w) / 2
        cam_quat = camera_l.data.quat_w_world
        world_T_cam = pose_to_transformation_matrix(cam_pos.cpu().numpy()[0], cam_quat.cpu().numpy()[0])
        cam_T_world = np.linalg.inv(world_T_cam)

        if not orientation_aligned:
            align_mtm_orientation_once(mtm_manipulator, mtm_interface, psm2, cam_T_world)
            orientation_aligned = True

        psm1_base = pose_to_transformation_matrix(
            psm1.data.body_link_pos_w[0][0].cpu().numpy(),
            psm1.data.body_link_quat_w[0][0].cpu().numpy()
        )
        psm2_base = pose_to_transformation_matrix(
            psm2.data.body_link_pos_w[0][0].cpu().numpy(),
            psm2.data.body_link_quat_w[0][0].cpu().numpy()
        )

        _, _, _, mtmr_pos, mtmr_rot, r_gripper_joint, mtm_clutch, _, _ = mtm_interface.advance()
        stylus_pose, po_gripper, po_clutch = po_interface.advance()

        if po_clutch is None or r_gripper_joint is None:
            time.sleep(0.05)
            continue

        if not mtm_clutch:
            if was_in_mtm_clutch:
                init_mtmr_position = mtmr_pos
                tip = pose_to_transformation_matrix(
                    psm2.data.body_link_pos_w[0][-1].cpu().numpy(),
                    psm2.data.body_link_quat_w[0][-1].cpu().numpy()
                )
                init_psm2_tip_position = (cam_T_world @ tip)[:3, 3]
            mtmr_orientation = R.from_rotvec(mtmr_rot).as_quat()
            mtmr_orientation = np.concatenate([[mtmr_orientation[3]], mtmr_orientation[:3]])
            psm2_target = init_psm2_tip_position + (mtmr_pos - init_mtmr_position) * scale
            cam_T_psm2tip = pose_to_transformation_matrix(psm2_target, mtmr_orientation)
            was_in_mtm_clutch = False
        else:
            was_in_mtm_clutch = True
            tip = pose_to_transformation_matrix(
                psm2.data.body_link_pos_w[0][-1].cpu().numpy(),
                psm2.data.body_link_quat_w[0][-1].cpu().numpy()
            )
            cam_T_psm2tip = cam_T_world @ tip

        if not po_clutch:
            if was_in_po_clutch:
                print("[PO] Clutch released. Initializing teleoperation.")
                init_stylus_position = stylus_pose[:3]
                tip_pose = psm1.data.body_link_pos_w[0][-1].cpu().numpy()
                tip_quat = psm1.data.body_link_quat_w[0][-1].cpu().numpy()
                world_T_psm1tip = pose_to_transformation_matrix(tip_pose, tip_quat)
                cam_T_psm1tip_init = cam_T_world @ world_T_psm1tip
                init_psm1_tip_position = cam_T_psm1tip_init[:3, 3]

            stylus_orientation = R.from_rotvec(stylus_pose[3:]).as_quat()
            stylus_orientation = np.concatenate([[stylus_orientation[3]], stylus_orientation[:3]])
            psm1_target = init_psm1_tip_position + (stylus_pose[:3] - init_stylus_position) * scale
            cam_T_psm1tip = pose_to_transformation_matrix(psm1_target, stylus_orientation)
            was_in_po_clutch = False
        else:
            if not was_in_po_clutch:
                print("[PO] Clutch pressed. Freezing teleop.")
            was_in_po_clutch = True
            tip_pose = psm1.data.body_link_pos_w[0][-1].cpu().numpy()
            tip_quat = psm1.data.body_link_quat_w[0][-1].cpu().numpy()
            world_T_psm1tip = pose_to_transformation_matrix(tip_pose, tip_quat)
            cam_T_psm1tip = cam_T_world @ world_T_psm1tip
            po_gripper = None

        actions = process_actions(cam_T_psm1tip, psm1_base, cam_T_psm2tip, psm2_base, world_T_cam, env, po_gripper, r_gripper_joint)
        env.step(actions)

        teleop_logger.check_and_start_logging(env)

        if teleop_logger.enable_logging and teleop_logger.frame_num == 0:
            log_current_pose(env, teleop_logger.log_dir)

        teleop_logger.stop_logging()

        if teleop_logger.enable_logging:
            frame_num = teleop_logger.frame_num + 1
            timestamp = time.time()

            robot_states = {}
            for psm, robot_name in psm_name_dict.items():
                robot = env.unwrapped.scene[robot_name]
                joint_positions = robot.data.joint_pos[0][:6].cpu().numpy()
                jaw_angle = abs(robot.data.joint_pos[0][-2].cpu().numpy()) + abs(robot.data.joint_pos[0][-1].cpu().numpy())
                ee_position = robot.data.body_link_pos_w[0][-1].cpu().numpy()
                ee_quat = robot.data.body_link_quat_w[0][-1].cpu().numpy()
                orientation_matrix = R.from_quat(np.concatenate([ee_quat[1:], [ee_quat[0]]])).as_matrix()

                robot_states[psm] = {
                    "joint_positions": joint_positions,
                    "jaw_angle": jaw_angle,
                    "ee_position": ee_position,
                    "orientation_matrix": orientation_matrix
                }

            cam_l_img = camera_l.data.output["rgb"][0].cpu().numpy()
            cam_r_img = camera_r.data.output["rgb"][0].cpu().numpy()

            teleop_logger.enqueue(frame_num, timestamp, robot_states, cam_l_img, cam_r_img)
            teleop_logger.frame_num = frame_num

        elapsed = time.time() - start_time
        sleep_time = max(0.0, (1/200.0) - elapsed)
        time.sleep(sleep_time)

    env.close()
    if os.path.exists(args_cli.log_trigger_file):
        os.remove(args_cli.log_trigger_file)



if __name__ == "__main__":
    main()
    simulation_app.close()


