import argparse

from omni.isaac.lab.app import AppLauncher






# add argparse arguments
parser = argparse.ArgumentParser(description="MTM teleoperation for Custom MultiArm dVRK environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-MTML-ACT-v0", help="Name of the task.")
parser.add_argument("--scale", type=float, default=0.4, help="Teleop scaling factor.")
parser.add_argument("--is_simulated", type=bool, default=False, help="Whether the MTM input is from the simulated model or not.")
# parser.add_argument("--enable_logging", type=bool, default=False, help="Whether to log the teleoperation output or not.")
parser.add_argument("--enable_logging", action="store_true", help="Enable logging from the start (default is off)")
# parser.add_argument("--sim_time", type=float, default=30.0, help="Duration (in seconds) for teleoperation before auto exit.")
parser.add_argument("--log_trigger_file", type=str, default="log_trigger.txt",
    help="Path to a file that enables logging when it exists.")

parser.add_argument("--disable_viewport", action="store_true", help="Disable extra viewport windows.")
parser.add_argument("--demo_name", type=str, default=None, help="Custom name for the logging folder (e.g., 'demo_1')")

parser.add_argument(
    "--model_control",
    type=str,
    choices=["psm1", "psm2", "both","none"],
    default="none",
    help="Choose which arm(s) are controlled by the model"
)


# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
"""Rest everything follows."""

# File for model trigger detection
import sys
import os

sys.path.append(os.path.abspath("."))

import gymnasium as gym
import torch
import carb
from omni.isaac.lab_tasks.utils import parse_env_cfg
import numpy as np
import time
from scipy.spatial.transform import Rotation as R
import omni.kit.viewport.utility as vp_utils


from tf_utils import pose_to_transformation_matrix, transformation_matrix_to_pose
from teleop_interface.MTM.se3_mtm import MTMTeleop
from teleop_interface.phantomomni.se3_phantomomni import PhantomOmniTeleop
from teleop_interface.MTM.mtm_manipulator import MTMManipulator
from scripts.teleoperation.teleop_logger_2_arm import TeleopLogger, reset_cube_pose, log_current_pose
import custom_envs

import global_cfg

# Overwrite the value from CLI args
global_cfg.model_control = args_cli.model_control




from AdaptACT.procedures.controller import AutonomousController

MODEL_TRIGGER_PATH = os.path.join(os.path.dirname(__file__), "model_trigger.txt")
RESET_TRIGGER_PATH = "reset_trigger.txt"




def get_jaw_gripper_angles(gripper_command):
    if gripper_command is None:
        return np.array([0.0, 0.0])
    norm_cmd = np.interp(gripper_command, [-1.0, 1.0], [-1.72, 1.06])
    angle = 0.52359 / (1.06 + 1.72) * (norm_cmd + 1.72)
    return np.array([angle, angle])

def get_jaw_angle(joint_pos):
    return abs(joint_pos[-2]) + abs(joint_pos[-1])


def process_actions_psm1(psm1_joint_action, cam_T_psm2, w_T_psm2base, w_T_cam, gripper2_command, env):
    psm2base_T_psm2 = np.linalg.inv(w_T_psm2base) @ w_T_cam @ cam_T_psm2
    psm2_rel_pos, psm2_rel_quat = transformation_matrix_to_pose(psm2base_T_psm2)
    g2_angle = np.interp(gripper2_command, [-1.0, 1.0], [-1.72, 1.06])
    jaw_angle = 0.52359 / (1.06 + 1.72) * (g2_angle + 1.72)
    g2_angles = np.array([-jaw_angle / 2, jaw_angle / 2])
    actions = np.concatenate([
        psm1_joint_action[:6],
        psm1_joint_action[6:8],
        psm2_rel_pos,
        psm2_rel_quat,
        g2_angles
    ])
    return torch.tensor(actions, device=env.unwrapped.device).repeat(env.unwrapped.num_envs, 1)


def process_actions_psm2(cam_T_psm1, w_T_psm1base, w_T_cam, gripper1_command, psm2_joint_action, env):
    psm1base_T_psm1 = np.linalg.inv(w_T_psm1base) @ w_T_cam @ cam_T_psm1
    psm1_rel_pos, psm1_rel_quat = transformation_matrix_to_pose(psm1base_T_psm1)
    g1_angle = np.interp(gripper1_command, [-1.0, 1.0], [-1.72, 1.06])
    jaw_angle = 0.52359 / (1.06 + 1.72) * (g1_angle + 1.72)
    g1_angles = np.array([-jaw_angle / 2, jaw_angle / 2])
    actions = np.concatenate([
        psm1_rel_pos,
        psm1_rel_quat,
        g1_angles,
        psm2_joint_action[:6],
        psm2_joint_action[6:8],
    ])
    return torch.tensor(actions, device=env.unwrapped.device).repeat(env.unwrapped.num_envs, 1)


def process_actions_both(psm1_joint_action, psm2_joint_action, env):
    psm1_action = np.concatenate([
        psm1_joint_action[:6],
        psm1_joint_action[6:8]
    ])
    psm2_action = np.concatenate([
        psm2_joint_action[:6],
        psm2_joint_action[6:8]
    ])
    actions = np.concatenate([psm1_action, psm2_action])
    return torch.tensor(actions, device=env.unwrapped.device).repeat(env.unwrapped.num_envs, 1)


def align_mtm_orientation_once(mtm_manipulator, mtm_interface, psm1, psm2, cam_T_world):
    print("[ALIGN] Aligning MTMR orientation with PSM2 pose...")
    psm1_tip_pose = pose_to_transformation_matrix(
        psm1.data.body_link_pos_w[0][-1].cpu().numpy(),
        psm1.data.body_link_quat_w[0][-1].cpu().numpy()
    )

    psm2_tip_pose = pose_to_transformation_matrix(
        psm2.data.body_link_pos_w[0][-1].cpu().numpy(),
        psm2.data.body_link_quat_w[0][-1].cpu().numpy()
    )
    hrsv_T_mtmr = mtm_interface.simpose2hrsvpose(cam_T_world @ psm2_tip_pose)
    hrsv_T_mtml = mtm_interface.simpose2hrsvpose(cam_T_world @ psm1_tip_pose)
    mtm_manipulator.adjust_orientation(hrsv_T_mtml[:3, :3], hrsv_T_mtmr[:3, :3])
    print("[ALIGN] MTMR orientation set.")




def reset_to_initial_pose_psm1(env, saved_psm1_joint_action,
                               saved_psm2_tip_pos_w, saved_psm2_tip_quat_w,
                               world_T_base_2):
    print("[RESET] Moving PSM1 and PSM2 to saved initial poses...")

    world_T_psm2_tip = pose_to_transformation_matrix(saved_psm2_tip_pos_w, saved_psm2_tip_quat_w)
    base2_T_tip = np.linalg.inv(world_T_base_2) @ world_T_psm2_tip
    psm2_rel_pos, psm2_rel_quat = transformation_matrix_to_pose(base2_T_tip)

    action = np.concatenate([
        saved_psm1_joint_action[:6],        # 6D joint space for PSM1
        saved_psm1_joint_action[6:8],       # 2D gripper
        psm2_rel_pos,                       # 3D position for PSM2
        psm2_rel_quat,                      # 4D orientation for PSM2
        [0.0, 0.0]                          # 2D gripper for PSM2
    ])

    assert len(action) == 17, f"Action length is {len(action)} but expected 17."

    action_tensor = torch.tensor(action, device=env.unwrapped.device).unsqueeze(0)

    for _ in range(90):
        env.step(action_tensor)

    print("[RESET] PSM1 and PSM2 moved to initial pose.")



def reset_to_initial_pose_psm2(env, saved_psm2_joint_action,
                               saved_psm1_tip_pos_w, saved_psm1_tip_quat_w,
                               world_T_base_1):
    print("[RESET] Moving PSM1 and PSM2 to saved initial poses...")

    world_T_psm1_tip = pose_to_transformation_matrix(saved_psm1_tip_pos_w, saved_psm1_tip_quat_w)
    base1_T_tip = np.linalg.inv(world_T_base_1) @ world_T_psm1_tip
    psm1_rel_pos, psm1_rel_quat = transformation_matrix_to_pose(base1_T_tip)

    action = np.concatenate([
        psm1_rel_pos,                       # 3D position for PSM1
        psm1_rel_quat,                      # 4D orientation for PSM1
        [0.0, 0.0],                         # 2D gripper for PSM1
        saved_psm2_joint_action[:6],        # 6D joint space for PSM2
        saved_psm2_joint_action[6:8],       # 2D gripper
    ])

    assert len(action) == 17, f"Action length is {len(action)} but expected 17."

    action_tensor = torch.tensor(action, device=env.unwrapped.device).unsqueeze(0)

    for _ in range(90):
        env.step(action_tensor)

    print("[RESET] PSM1 and PSM2 moved to initial pose.")


def reset_to_initial_pose_both(env, saved_psm1_joint_action,
                               saved_psm2_joint_action):
    print("[RESET] Moving PSM1 and PSM2 to saved initial poses...")


    action = np.concatenate([
        saved_psm1_joint_action[:6],        # 6D joint space for PSM1
        saved_psm1_joint_action[6:8],       # 2D gripper
        saved_psm2_joint_action[:6],        # 6D joint space for PSM2
        saved_psm2_joint_action[6:8],       # 2D gripper
    ])

    assert len(action) == 16, f"Action length is {len(action)} but expected 16."

    action_tensor = torch.tensor(action, device=env.unwrapped.device).unsqueeze(0)

    for _ in range(90):
        env.step(action_tensor)

    print("[RESET] PSM1 and PSM2 moved to initial pose.")



def reorient_mtm_to_match_psm(
    mtm_manipulator, teleop_interface,
    saved_psm1_tip_pos_w, saved_psm1_tip_quat_w,
    saved_psm2_tip_pos_w, saved_psm2_tip_quat_w,
    cam_T_world
):
    print("[RESET] Reorienting MTM to match PSM tip pose...")

    world_T_psm1tip = pose_to_transformation_matrix(saved_psm1_tip_pos_w, saved_psm1_tip_quat_w)
    cam_T_psm1tip = cam_T_world @ world_T_psm1tip

    world_T_psm2tip = pose_to_transformation_matrix(saved_psm2_tip_pos_w, saved_psm2_tip_quat_w)
    cam_T_psm2tip = cam_T_world @ world_T_psm2tip

    hrsv_T_mtml = teleop_interface.simpose2hrsvpose(cam_T_psm1tip)
    hrsv_T_mtmr = teleop_interface.simpose2hrsvpose(cam_T_psm2tip)

    mtm_manipulator.home()
    time.sleep(2.0)
    mtm_manipulator.adjust_orientation(hrsv_T_mtml[:3, :3],hrsv_T_mtmr[:3, :3])

    print("[RESET] MTMR reoriented.")


# def _compute_base_relative_action_psm1(env, psm1, psm2, jaw1, jaw2):
#     """Helper to compute and return action tensor given PSMs and target jaw angles."""
#     # Get world-frame tip poses
#     psm1_tip_pos = psm1.data.body_link_pos_w[0][-1].cpu().numpy()
#     psm1_tip_quat = psm1.data.body_link_quat_w[0][-1].cpu().numpy()
#     psm2_tip_pos = psm2.data.body_link_pos_w[0][-1].cpu().numpy()
#     psm2_tip_quat = psm2.data.body_link_quat_w[0][-1].cpu().numpy()
#     world_T_psm1tip = pose_to_transformation_matrix(psm1_tip_pos, psm1_tip_quat)
#     world_T_psm2tip = pose_to_transformation_matrix(psm2_tip_pos, psm2_tip_quat)

#     # Get base transforms
#     psm1_base_link_pos = psm1.data.body_link_pos_w[0][0].cpu().numpy()
#     psm1_base_link_quat = psm1.data.body_link_quat_w[0][0].cpu().numpy()
#     psm2_base_link_pos = psm2.data.body_link_pos_w[0][0].cpu().numpy()
#     psm2_base_link_quat = psm2.data.body_link_quat_w[0][0].cpu().numpy()
#     world_T_psm1_base = pose_to_transformation_matrix(psm1_base_link_pos, psm1_base_link_quat)
#     world_T_psm2_base = pose_to_transformation_matrix(psm2_base_link_pos, psm2_base_link_quat)

#     # Compute base-relative poses
#     psm1_rel_pos, psm1_rel_quat = transformation_matrix_to_pose(np.linalg.inv(world_T_psm1_base) @ world_T_psm1tip)
#     psm2_rel_pos, psm2_rel_quat = transformation_matrix_to_pose(np.linalg.inv(world_T_psm2_base) @ world_T_psm2tip)

#     psm1_joint_action = psm1.data.joint_pos[0][:6].cpu().numpy()

#     # Construct action tensor
#     action_tensor = torch.tensor(
#         np.concatenate([psm1_joint_action, jaw1, psm2_rel_pos, psm2_rel_quat, jaw2], dtype=np.float32),
#         device=env.unwrapped.device
#     ).unsqueeze(0)

#     return action_tensor

# def _compute_base_relative_action_psm2(env, psm1, psm2, jaw1, jaw2):
#     """Helper to compute and return action tensor given PSMs and target jaw angles."""
#     # Get world-frame tip poses
#     psm1_tip_pos = psm1.data.body_link_pos_w[0][-1].cpu().numpy()
#     psm1_tip_quat = psm1.data.body_link_quat_w[0][-1].cpu().numpy()
#     psm2_tip_pos = psm2.data.body_link_pos_w[0][-1].cpu().numpy()
#     psm2_tip_quat = psm2.data.body_link_quat_w[0][-1].cpu().numpy()
#     world_T_psm1tip = pose_to_transformation_matrix(psm1_tip_pos, psm1_tip_quat)
#     world_T_psm2tip = pose_to_transformation_matrix(psm2_tip_pos, psm2_tip_quat)

#     # Get base transforms
#     psm1_base_link_pos = psm1.data.body_link_pos_w[0][0].cpu().numpy()
#     psm1_base_link_quat = psm1.data.body_link_quat_w[0][0].cpu().numpy()
#     psm2_base_link_pos = psm2.data.body_link_pos_w[0][0].cpu().numpy()
#     psm2_base_link_quat = psm2.data.body_link_quat_w[0][0].cpu().numpy()
#     world_T_psm1_base = pose_to_transformation_matrix(psm1_base_link_pos, psm1_base_link_quat)
#     world_T_psm2_base = pose_to_transformation_matrix(psm2_base_link_pos, psm2_base_link_quat)

#     # Compute base-relative poses
#     psm1_rel_pos, psm1_rel_quat = transformation_matrix_to_pose(np.linalg.inv(world_T_psm1_base) @ world_T_psm1tip)
#     psm2_rel_pos, psm2_rel_quat = transformation_matrix_to_pose(np.linalg.inv(world_T_psm2_base) @ world_T_psm2tip)

#     psm2_joint_action = psm2.data.joint_pos[0][:6].cpu().numpy()

#     # Construct action tensor
#     action_tensor = torch.tensor(
#         np.concatenate([psm1_rel_pos, psm1_rel_quat, jaw1, psm2_joint_action, jaw2], dtype=np.float32),
#         device=env.unwrapped.device
#     ).unsqueeze(0)

#     return action_tensor

# def _compute_base_relative_action_both(env, psm1, psm2, jaw1, jaw2):
#     """Helper to compute and return action tensor given PSMs and target jaw angles."""
    
#     psm1_joint_action = psm1.data.joint_pos[0][:6].cpu().numpy()
#     psm2_joint_action = psm2.data.joint_pos[0][:6].cpu().numpy()

#     # Construct action tensor
#     action_tensor = torch.tensor(
#         np.concatenate([psm1_joint_action, jaw1, psm2_joint_action, jaw2], dtype=np.float32),
#         device=env.unwrapped.device
#     ).unsqueeze(0)

#     return action_tensor

# def open_jaws(env, psm1, psm2, target=1.04):
#     """Open PSM jaws to match a target angle."""
#     jaw_angle = 0.52359 / (1.06 + 1.72) * (target + 1.72)
#     jaw1 = [-jaw_angle, jaw_angle]
#     jaw2 = [-jaw_angle, jaw_angle]

#     print(f"[INIT] Opening jaws to approximate MTM input: {target} → [{jaw1[0]:.3f}, {jaw1[1]:.3f}]")
#     if global_cfg.model_control == "psm1":
#         action_tensor = _compute_base_relative_action_psm1(env, psm1, psm2, jaw1=jaw1, jaw2=jaw2)
#     elif global_cfg.model_control == "psm2":
#         action_tensor = _compute_base_relative_action_psm2(env, psm1, psm2, jaw1=jaw1, jaw2=jaw2)
#     elif global_cfg.model_control == "both":
#         action_tensor = _compute_base_relative_action_both(env, psm1, psm2, jaw1=jaw1, jaw2=jaw2)

#     for _ in range(30):
#         env.step(action_tensor)
#     print("[INIT] PSM jaws opened.")



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

    controller = AutonomousController.from_train_dir(
        train_dir="/home/stanford/Demo_collections1/Models/4_orbitsim_single_human_demos/Joint Control/20250516-053119_kind-opossum_train",
        ckpt_path="/home/stanford/Demo_collections1/Models/4_orbitsim_single_human_demos/Joint Control/20250516-053119_kind-opossum_train/policy_epoch_10000_seed_0.ckpt",
        ckpt_strategy="none",
        device=args_cli.device
    )
    controller.reset()

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
    orientation_aligned = False
    model_triggered = False
    printed_trigger = False

    # Initial pose cache
    saved_psm1_tip_pos_w = None
    saved_psm1_tip_quat_w = None
    saved_psm2_tip_pos_w = None
    saved_psm2_tip_quat_w = None

    psm_name_dict = {
        "PSM1": "robot_1",
        "PSM2": "robot_2"
    }

    # Capture PSM1 joint pose immediately after reset to keep it steady
    saved_psm1_joint_action = psm1.data.joint_pos[0].cpu().numpy()
    saved_psm2_joint_action = psm2.data.joint_pos[0].cpu().numpy()


    teleop_logger = TeleopLogger(
        trigger_file=args_cli.log_trigger_file,
        psm_name_dict=psm_name_dict,
        log_duration=30.0
    )

    print("Press the clutch button and release to start teleoperation.")
    while simulation_app.is_running():
        start_time = time.time()

        cam_pos = (camera_l.data.pos_w + camera_r.data.pos_w) / 2
        cam_quat = camera_l.data.quat_w_world
        world_T_cam = pose_to_transformation_matrix(cam_pos.cpu().numpy()[0], cam_quat.cpu().numpy()[0])
        cam_T_world = np.linalg.inv(world_T_cam)

        if not orientation_aligned:
            align_mtm_orientation_once(mtm_manipulator, mtm_interface, psm1, psm2, cam_T_world)
            orientation_aligned = True
            saved_psm1_tip_pos_w = psm1.data.body_link_pos_w[0][-1].cpu().numpy()
            saved_psm1_tip_quat_w = psm1.data.body_link_quat_w[0][-1].cpu().numpy()
            saved_psm2_tip_pos_w = psm2.data.body_link_pos_w[0][-1].cpu().numpy()
            saved_psm2_tip_quat_w = psm2.data.body_link_quat_w[0][-1].cpu().numpy()
            # open_jaws(env, psm1, psm2, target=0.8)

        psm1_base = pose_to_transformation_matrix(
            psm1.data.body_link_pos_w[0][0].cpu().numpy(),
            psm1.data.body_link_quat_w[0][0].cpu().numpy()
        )
        psm2_base = pose_to_transformation_matrix(
            psm2.data.body_link_pos_w[0][0].cpu().numpy(),
            psm2.data.body_link_quat_w[0][0].cpu().numpy()
        )

        (mtml_pos, mtml_rot, l_gripper_joint, 
        mtmr_pos, mtmr_rot, r_gripper_joint, 
        mtm_clutch, _, _) = mtm_interface.advance()

        # Handle clutch release logic
        if not mtm_clutch:
            if was_in_mtm_clutch:
                init_mtml_pos = mtml_pos
                init_mtmr_pos = mtmr_pos

                world_T_psm1tip = pose_to_transformation_matrix(
                    psm1.data.body_link_pos_w[0][-1].cpu().numpy(),
                    psm1.data.body_link_quat_w[0][-1].cpu().numpy()
                )
                world_T_psm2tip = pose_to_transformation_matrix(
                    psm2.data.body_link_pos_w[0][-1].cpu().numpy(),
                    psm2.data.body_link_quat_w[0][-1].cpu().numpy()
                )

                init_psm1_tip_pos = (cam_T_world @ world_T_psm1tip)[:3, 3]
                init_psm2_tip_pos = (cam_T_world @ world_T_psm2tip)[:3, 3]
                
            mtml_quat = R.from_rotvec(mtml_rot).as_quat()
            mtml_quat = np.concatenate([[mtml_quat[3]], mtml_quat[:3]])
            psm1_target_pos = init_psm1_tip_pos + (mtml_pos - init_mtml_pos) * scale
            cam_T_psm1tip = pose_to_transformation_matrix(psm1_target_pos, mtml_quat)

            mtmr_quat = R.from_rotvec(mtmr_rot).as_quat()
            mtmr_quat = np.concatenate([[mtmr_quat[3]], mtmr_quat[:3]])
            psm2_target_pos = init_psm2_tip_pos + (mtmr_pos - init_mtmr_pos) * scale
            cam_T_psm2tip = pose_to_transformation_matrix(psm2_target_pos, mtmr_quat)
            was_in_mtm_clutch = False

        else:
            was_in_mtm_clutch = True

            world_T_psm1tip = pose_to_transformation_matrix(
                    psm1.data.body_link_pos_w[0][-1].cpu().numpy(),
                    psm1.data.body_link_quat_w[0][-1].cpu().numpy()
            )
            world_T_psm2tip = pose_to_transformation_matrix(
                psm2.data.body_link_pos_w[0][-1].cpu().numpy(),
                psm2.data.body_link_quat_w[0][-1].cpu().numpy()
            )

            cam_T_psm1tip = cam_T_world @ world_T_psm1tip
            cam_T_psm2tip = cam_T_world @ world_T_psm2tip


        # === Check for trigger and start logging ===
        teleop_logger.check_and_start_logging(env)

        # === Stop logging only when time exceeds duration ===
        teleop_logger.stop_logging()

        # === If just started logging, log initial pose ===
        if teleop_logger.enable_logging and teleop_logger.frame_num == 0:
            log_current_pose(env, teleop_logger.log_dir)

        # === Per-frame logging ===
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

        if os.path.exists("reset_trigger.txt"):
            print("[RESET] Detected reset trigger. Resetting cube and both PSMs...")

            # Reset cube
            cube_pos = [np.random.uniform(0.0, 0.05), np.random.uniform(-0.025, 0.025), 0.0]
            cube_yaw = np.random.uniform(-np.pi, np.pi)
            cube_quat = R.from_euler("z", cube_yaw).as_quat()
            cube_ori = [cube_quat[3], cube_quat[0], cube_quat[1], cube_quat[2]]
            reset_cube_pose(env, "teleop_logs/cube_latest", cube_pos, cube_ori)

            # Recompute cam_T_world and base transforms
            cam_pos = (camera_l.data.pos_w + camera_r.data.pos_w) / 2
            cam_quat = camera_l.data.quat_w_world
            world_T_cam = pose_to_transformation_matrix(cam_pos.cpu().numpy()[0], cam_quat.cpu().numpy()[0])
            cam_T_world = np.linalg.inv(world_T_cam)

            psm1_base = pose_to_transformation_matrix(
                psm1.data.body_link_pos_w[0][0].cpu().numpy(),
                psm1.data.body_link_quat_w[0][0].cpu().numpy()
            )
            psm2_base = pose_to_transformation_matrix(
                psm2.data.body_link_pos_w[0][0].cpu().numpy(),
                psm2.data.body_link_quat_w[0][0].cpu().numpy()
            )

            if global_cfg.model_control == "psm1":
                reset_to_initial_pose_psm1(
                    env,
                    saved_psm1_joint_action,
                    saved_psm2_tip_pos_w, saved_psm2_tip_quat_w,
                    psm2_base,
                )
                controller.reset()

                reorient_mtm_to_match_psm(
                    mtm_manipulator, mtm_interface,
                    saved_psm1_tip_pos_w, saved_psm1_tip_quat_w,
                    saved_psm2_tip_pos_w, saved_psm2_tip_quat_w,
                    cam_T_world
                )

            elif global_cfg.model_control == "psm2":
                reset_to_initial_pose_psm2(
                    env,
                    saved_psm2_joint_action,
                    saved_psm1_tip_pos_w, saved_psm1_tip_quat_w,
                    psm1_base,
                )
                controller.reset()

                reorient_mtm_to_match_psm(
                    mtm_manipulator, mtm_interface,
                    saved_psm1_tip_pos_w, saved_psm1_tip_quat_w,
                    saved_psm2_tip_pos_w, saved_psm2_tip_quat_w,
                    cam_T_world
                )

            elif global_cfg.model_control == "both":
                reset_to_initial_pose_both(
                    env, 
                    saved_psm1_joint_action,
                    saved_psm2_joint_action,
                )
                controller.reset()
                reorient_mtm_to_match_psm(
                    mtm_manipulator, mtm_interface,
                    saved_psm1_tip_pos_w, saved_psm1_tip_quat_w,
                    saved_psm2_tip_pos_w, saved_psm2_tip_quat_w,
                    cam_T_world
                )

            # open_jaws(env, psm1, psm2, target=0.8)


            # Reset clutch and state
            was_in_mtm_clutch = True
            orientation_aligned = False
            model_triggered = False
            printed_trigger = False

            os.remove("reset_trigger.txt")

            print("[RESET] Reset complete. Reclutch both inputs to resume.")
            continue


        # Enable ACT control when model_trigger file appears
        if os.path.exists(MODEL_TRIGGER_PATH) and not model_triggered:
            print("[MODEL] Trigger detected. Starting ACT control of PSM1.")
            model_triggered = True
            printed_trigger = True
            try:
                os.remove(MODEL_TRIGGER_PATH)
            except Exception as e:
                print(f"[MODEL] Failed to remove trigger file: {e}")

        # # === Process ACT if active ===
        if model_triggered:
            try:
                # Get joint states
                psm1_q = psm1.data.joint_pos[0].cpu().numpy()
                psm2_q = psm2.data.joint_pos[0].cpu().numpy()
                jaw1, jaw2 = get_jaw_angle(psm1_q), get_jaw_angle(psm2_q)

                # Model input
                model_input = np.concatenate([psm1_q[:-2], [jaw1], psm2_q[:-2], [jaw2]])
                imgs = np.stack([
                    camera_l.data.output["rgb"][0].cpu().numpy(),
                    camera_r.data.output["rgb"][0].cpu().numpy()
                ]) / 255.0

                # ACT model inference
                act = controller.step(imgs, model_input)
                psm1_joints = act[:6]
                psm2_joints = act[7:-1]
                psm1_jaw = act[6]
                psm2_jaw = act[-1]

                psm1_act = np.concatenate([psm1_joints, [-psm1_jaw / 2, psm1_jaw / 2]])
                psm2_act = np.concatenate([psm2_joints, [-psm2_jaw / 2, psm2_jaw / 2]])

                last_psm1_act = psm1_act
                last_psm2_act = psm2_act

            except RuntimeError as e:
                if "Rollout was already completed" in str(e):
                    if printed_trigger:
                        print("[MODEL] ACT rollout completed. Freezing controlled PSM(s) at last pose.")
                        printed_trigger = False
                    model_triggered = False

                    if global_cfg.model_control == "psm1":
                        psm1_act = last_psm1_act
                        psm2_act = saved_psm2_joint_action
                    elif global_cfg.model_control == "psm2":
                        psm1_act = saved_psm1_joint_action
                        psm2_act = last_psm2_act
                    elif global_cfg.model_control == "both":
                        psm1_act = last_psm1_act
                        psm2_act = last_psm2_act
                    else:
                        psm1_act = saved_psm1_joint_action
                        psm2_act = saved_psm2_joint_action
                else:
                    raise
        else:
            psm1_act = saved_psm1_joint_action
            psm2_act = saved_psm2_joint_action

        # === Compute base transforms
        psm1_base = pose_to_transformation_matrix(
            psm1.data.body_link_pos_w[0][0].cpu().numpy(),
            psm1.data.body_link_quat_w[0][0].cpu().numpy()
        )
        psm2_base = pose_to_transformation_matrix(
            psm2.data.body_link_pos_w[0][0].cpu().numpy(),
            psm2.data.body_link_quat_w[0][0].cpu().numpy()
        )

        # === Choose process_actions function
        if global_cfg.model_control == "psm1":
            actions = process_actions_psm1(
                psm1_act,
                cam_T_psm2tip,
                psm2_base,
                world_T_cam,
                r_gripper_joint,
                env
            )

        elif global_cfg.model_control == "psm2":
            actions = process_actions_psm2(
                cam_T_psm1tip,
                psm1_base,
                world_T_cam,
                l_gripper_joint,
                psm2_act,
                env
            )

        elif global_cfg.model_control == "both":
            actions = process_actions_both(
                psm1_act,
                psm2_act,
                env
            )

        else:
            raise ValueError(f"Invalid model_control value: {global_cfg.model_control}")

        # === Step the environment
        env.step(actions)
        time.sleep(max(0.0, 1 / 200.0 - (time.time() - start_time)))
    # Clean up the trigger files but do not close env or simulation
    if os.path.exists(args_cli.log_trigger_file):
        os.remove(args_cli.log_trigger_file)
    if os.path.exists(MODEL_TRIGGER_PATH):
        os.remove(MODEL_TRIGGER_PATH)

    print("[INFO] Rollout complete. Simulation is still running. Press Ctrl+C to exit.")

    # Idle loop to keep the app alive
    while simulation_app.is_running():
        time.sleep(0.1)



if __name__ == "__main__":
    main()
    simulation_app.close()

