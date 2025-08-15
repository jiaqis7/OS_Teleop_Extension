# --------------------------------------------
# 1. CLI ARGUMENTS AND ISAAC SIM LAUNCH SETUP
# --------------------------------------------

import argparse
from omni.isaac.lab.app import AppLauncher

# Parse CLI arguments
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

# Launch Isaac Sim App
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app



# ---------------------------------------------------------
# 2. IMPORTS AND INITIAL SETUP FOR TELEOP, ENV, LOGGING
# ---------------------------------------------------------

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
from scripts.teleoperation.teleop_logger_2_arm import TeleopLogger, log_current_pose, reset_cube_pose


# -------------------------------
# 3. TELEOP UTILITIES
# -------------------------------

def get_jaw_gripper_angles(gripper_command, robot_name):
    if gripper_command is None:
        return np.array([0.0, 0.0])
    norm_cmd = np.interp(gripper_command, [-1.0, 1.0], [-1.72, 1.06])
    g2_angle = 0.52359 / (1.06 + 1.72) * (norm_cmd + 1.72)
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



def reset_psm_to_initial_pose(
    env,
    saved_psm1_tip_pos_w, saved_psm1_tip_quat_w,
    saved_psm2_tip_pos_w, saved_psm2_tip_quat_w,
    world_T_psm1_base, world_T_psm2_base,
    num_steps=30
):
    print("[RESET] Moving PSMs to saved initial pose...")

    world_T_psm1_tip = pose_to_transformation_matrix(saved_psm1_tip_pos_w, saved_psm1_tip_quat_w)
    world_T_psm2_tip = pose_to_transformation_matrix(saved_psm2_tip_pos_w, saved_psm2_tip_quat_w)

    psm1_base_T_tip = np.linalg.inv(world_T_psm1_base) @ world_T_psm1_tip
    psm2_base_T_tip = np.linalg.inv(world_T_psm2_base) @ world_T_psm2_tip

    psm1_rel_pos, psm1_rel_quat = transformation_matrix_to_pose(psm1_base_T_tip)
    psm2_rel_pos, psm2_rel_quat = transformation_matrix_to_pose(psm2_base_T_tip)

    action_vec = np.concatenate([
        psm1_rel_pos, psm1_rel_quat, [0.0, 0.0],
        psm2_rel_pos, psm2_rel_quat, [0.0, 0.0]
    ], dtype=np.float32)

    action_tensor = torch.tensor(action_vec, device=env.unwrapped.device).unsqueeze(0)

    for _ in range(90):
        env.step(action_tensor)

    print("[RESET] PSMs moved to initial pose.")


def reorient_mtm_to_match_psm(
    mtm_manipulator, teleop_interface,
    saved_psm1_tip_pos_w, saved_psm1_tip_quat_w,
    saved_psm2_tip_pos_w, saved_psm2_tip_quat_w,
    cam_T_world
):
    print("[RESET] Reorienting MTMs to match PSM tip pose...")

    world_T_psm1tip = pose_to_transformation_matrix(saved_psm1_tip_pos_w, saved_psm1_tip_quat_w)
    world_T_psm2tip = pose_to_transformation_matrix(saved_psm2_tip_pos_w, saved_psm2_tip_quat_w)

    cam_T_psm1tip = cam_T_world @ world_T_psm1tip
    cam_T_psm2tip = cam_T_world @ world_T_psm2tip

    hrsv_T_mtml = teleop_interface.simpose2hrsvpose(cam_T_psm1tip)
    hrsv_T_mtmr = teleop_interface.simpose2hrsvpose(cam_T_psm2tip)

    mtm_manipulator.home()
    time.sleep(2.0)
    mtm_manipulator.adjust_orientation(hrsv_T_mtml, hrsv_T_mtmr)

    print("[RESET] MTMs reoriented.")

def reorient_mtm_to_match_psm_right_only(
    mtm_manipulator, teleop_interface,
    saved_psm2_tip_pos_w, saved_psm2_tip_quat_w,
    cam_T_world
):
    print("[RESET] Reorienting **only MTMR** to match PSM2 tip pose...")

    world_T_psm2tip = pose_to_transformation_matrix(saved_psm2_tip_pos_w, saved_psm2_tip_quat_w)
    cam_T_psm2tip = cam_T_world @ world_T_psm2tip

    hrsv_T_mtmr = teleop_interface.simpose2hrsvpose(cam_T_psm2tip)

    mtm_manipulator.home()
    time.sleep(2.0)
    mtm_manipulator.adjust_orientation_right(hrsv_T_mtmr[:3, :3])

    print("[RESET] MTMR reoriented.")


def _compute_base_relative_action(env, psm1, psm2, jaw1, jaw2):
    """Helper to compute and return action tensor given PSMs and target jaw angles."""
    # Get world-frame tip poses
    psm1_tip_pos = psm1.data.body_link_pos_w[0][-1].cpu().numpy()
    psm1_tip_quat = psm1.data.body_link_quat_w[0][-1].cpu().numpy()
    psm2_tip_pos = psm2.data.body_link_pos_w[0][-1].cpu().numpy()
    psm2_tip_quat = psm2.data.body_link_quat_w[0][-1].cpu().numpy()
    world_T_psm1tip = pose_to_transformation_matrix(psm1_tip_pos, psm1_tip_quat)
    world_T_psm2tip = pose_to_transformation_matrix(psm2_tip_pos, psm2_tip_quat)

    # Get base transforms
    psm1_base_link_pos = psm1.data.body_link_pos_w[0][0].cpu().numpy()
    psm1_base_link_quat = psm1.data.body_link_quat_w[0][0].cpu().numpy()
    psm2_base_link_pos = psm2.data.body_link_pos_w[0][0].cpu().numpy()
    psm2_base_link_quat = psm2.data.body_link_quat_w[0][0].cpu().numpy()
    world_T_psm1_base = pose_to_transformation_matrix(psm1_base_link_pos, psm1_base_link_quat)
    world_T_psm2_base = pose_to_transformation_matrix(psm2_base_link_pos, psm2_base_link_quat)

    # Compute base-relative poses
    psm1_rel_pos, psm1_rel_quat = transformation_matrix_to_pose(np.linalg.inv(world_T_psm1_base) @ world_T_psm1tip)
    psm2_rel_pos, psm2_rel_quat = transformation_matrix_to_pose(np.linalg.inv(world_T_psm2_base) @ world_T_psm2tip)

    # Construct action tensor
    action_tensor = torch.tensor(
        np.concatenate([psm1_rel_pos, psm1_rel_quat, jaw1, psm2_rel_pos, psm2_rel_quat, jaw2], dtype=np.float32),
        device=env.unwrapped.device
    ).unsqueeze(0)

    return action_tensor

def open_jaws(env, psm1, psm2, target=1.04):
    """Open PSM jaws to match a target angle."""
    jaw_angle = 0.52359 / (1.06 + 1.72) * (target + 1.72)
    jaw1 = [0.0, 0.0]
    jaw2 = [-jaw_angle, jaw_angle]

    print(f"[INIT] Opening jaws to approximate MTM input: {target} → [{jaw1[0]:.3f}, {jaw1[1]:.3f}]")
    action_tensor = _compute_base_relative_action(env, psm1, psm2, jaw1=jaw1, jaw2=jaw2)
    for _ in range(30):
        env.step(action_tensor)
    print("[INIT] PSM2 jaws opened.")


def set_jaws_closed(env, psm1, psm2):
    """Force-close jaws after reset to override any input artifacts."""
    print("[FIX] Forcing PSM jaws closed...")
    action_tensor = _compute_base_relative_action(env, psm1, psm2, jaw1=[0.0, 0.0], jaw2=[-0.0, 0.0])
    for _ in range(30):
        env.step(action_tensor)
    print("[FIX] PSM jaws closed.")



def main():

    # --------------------------
    # 1. Initialize Flags and Variables
    # --------------------------
    # Save tip pose of PSMs for resetting
    saved_psm1_tip_pos_w = None
    saved_psm1_tip_quat_w = None
    saved_psm2_tip_pos_w = None
    saved_psm2_tip_quat_w = None

    # Command-line options
    scale = args_cli.scale
    is_simulated = args_cli.is_simulated
    enable_logging = args_cli.enable_logging

    # PSM name mapping dictionary
    psm_name_dict = {
        "PSM1": "robot_1",
        "PSM2": "robot_2"
    }

    # --------------------------
    # 2. Initialize Logger, MTM, and Isaac Gym Env
    # --------------------------

    teleop_logger = TeleopLogger(
        trigger_file="log_trigger.txt",
        psm_name_dict=psm_name_dict,
        log_duration=30.0,
    )

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


    # --------------------------
    # 3. Simulation/Teleop Session State Variables
    # --------------------------

    camera_l = env.unwrapped.scene["camera_left"]
    camera_r = env.unwrapped.scene["camera_right"]

    if not args_cli.disable_viewport:
        view_port_l = vp_utils.create_viewport_window("Left Camera", width=800, height=600)
        view_port_l.viewport_api.camera_path = '/World/envs/env_0/Robot_4/ecm_end_link/camera_left'

        view_port_r = vp_utils.create_viewport_window("Right Camera", width=800, height=600)
        view_port_r.viewport_api.camera_path = '/World/envs/env_0/Robot_4/ecm_end_link/camera_right'

    psm1 = env.unwrapped.scene["robot_1"]
    psm2 = env.unwrapped.scene["robot_2"]


    # --------------------------
    # 4. Flags for PSM Teleop Logic
    # --------------------------

    was_in_mtm_clutch = True
    was_in_po_clutch = True
    po_waiting_for_clutch = False

    init_psm1_tip_position = None
    init_psm2_tip_position = None
    init_mtmr_position = None
    init_stylus_position = None
    orientation_aligned = False

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

            saved_psm1_tip_pos_w = psm1.data.body_link_pos_w[0][-1].cpu().numpy()
            saved_psm1_tip_quat_w = psm1.data.body_link_quat_w[0][-1].cpu().numpy()
            saved_psm2_tip_pos_w = psm2.data.body_link_pos_w[0][-1].cpu().numpy()
            saved_psm2_tip_quat_w = psm2.data.body_link_quat_w[0][-1].cpu().numpy()

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
            if po_waiting_for_clutch:
                continue  # ✅ don't resume until clutch is pressed again
            if was_in_po_clutch:
                print("[PO] Clutch released. Initializing teleoperation.")
                init_stylus_position = stylus_pose[:3]
                tip_pose = psm1.data.body_link_pos_w[0][-1].cpu().numpy()
                tip_quat = psm1.data.body_link_quat_w[0][-1].cpu().numpy()
                world_T_psm1tip = pose_to_transformation_matrix(tip_pose, tip_quat)
                cam_T_psm1tip_init = cam_T_world @ world_T_psm1tip
                init_psm1_tip_position = cam_T_psm1tip_init[:3, 3]
                was_in_po_clutch = False
            elif init_stylus_position is not None and init_psm1_tip_position is not None:
                stylus_orientation = R.from_rotvec(stylus_pose[3:]).as_quat()
                stylus_orientation = np.concatenate([[stylus_orientation[3]], stylus_orientation[:3]])
                psm1_target = init_psm1_tip_position + (stylus_pose[:3] - init_stylus_position) * scale
                cam_T_psm1tip = pose_to_transformation_matrix(psm1_target, stylus_orientation)
        else:
            if po_waiting_for_clutch:
                print("[PO] Clutch pressed after reset. Ready to resume.")
                po_waiting_for_clutch = False
                was_in_po_clutch = True
                continue
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



        # Check for RESET TRIGGER to randomize cube pose
        if os.path.exists("reset_trigger.txt"):
            print("[RESET] Detected reset trigger. Resetting environment...")

            # Check if this is a multi-cube environment
            scene = env.unwrapped.scene if hasattr(env, 'unwrapped') else env.scene
            scene_entities = list(scene.keys())
            
            if all(f"cube_rigid_{i}" in scene_entities for i in range(1, 5)):
                # Multi-cube environment - reset all 4 cubes
                print("[RESET] Multi-cube environment detected. Resetting all 4 cubes...")
                
                # Import the proper reset function for multi-cube
                from scripts.teleoperation.teleop_logger_3_arm import reset_cube_pose as reset_cube_pose_3arm
                
                # Reset colored blocks (1-3) to random positions
                cube_configs = [
                    ([np.random.uniform(-0.01, 0.01), np.random.uniform(-0.01, 0.01), 0.0], "cube_rigid_1"),
                    ([np.random.uniform(0.045, 0.055), np.random.uniform(0.0, 0.03), 0.0], "cube_rigid_2"),
                    ([np.random.uniform(-0.055, -0.045), np.random.uniform(0.0, 0.03), 0.0], "cube_rigid_3"),
                    ([0.0, 0.055, 0.0], "cube_rigid_4")  # White block - fixed position
                ]
                
                for i, (cube_pos, cube_key) in enumerate(cube_configs):
                    if i < 3:  # Colored blocks - random orientation
                        cube_yaw = np.random.uniform(-np.pi/2, np.pi/2)
                    else:  # White block - fixed orientation
                        cube_yaw = 0.0
                    
                    cube_quat = R.from_euler("z", cube_yaw).as_quat()
                    cube_ori = [cube_quat[3], cube_quat[0], cube_quat[1], cube_quat[2]]
                    reset_cube_pose_3arm(env, f"teleop_logs/cube_latest_{i+1}", cube_pos, cube_ori, cube_key=cube_key)
                
            else:
                # Single cube environment - reset as before
                print("[RESET] Single cube environment. Resetting single cube...")
                cube_pos = [np.random.uniform(0.0, 0.1), np.random.uniform(-0.05, 0.05), 0.0]
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

            # Reset PSMs to saved poses
            reset_psm_to_initial_pose(
                env,
                saved_psm1_tip_pos_w, saved_psm1_tip_quat_w,
                saved_psm2_tip_pos_w, saved_psm2_tip_quat_w,
                psm1_base, psm2_base,
                num_steps=30
            )

            # Reorient MTM
            reorient_mtm_to_match_psm_right_only(
                mtm_manipulator, mtm_interface,
                saved_psm2_tip_pos_w, saved_psm2_tip_quat_w,
                cam_T_world
            )

            open_jaws(env, psm1, psm2, target=0.8)

            # Reset clutch and state
            was_in_mtm_clutch = True
            was_in_po_clutch = True
            po_waiting_for_clutch = True
            init_psm1_tip_position = None
            init_psm2_tip_position = None
            init_mtmr_position = None
            init_stylus_position = None
            os.remove("reset_trigger.txt")

            print("[RESET] Reset complete. Reclutch both inputs to resume.")
            continue


        # Logging logic: trigger-based
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
            
            # Check contacts and log success if all blocks are in contact
            teleop_logger.check_contacts(env)

        # Sim loop rate control
        elapsed = time.time() - start_time
        sleep_time = max(0.0, (1/200.0) - elapsed)
        time.sleep(sleep_time)

    # close the simulator
    env.close()
    if os.path.exists(args_cli.log_trigger_file):
        os.remove(args_cli.log_trigger_file)



if __name__ == "__main__":
    main()
    teleop_logger.shutdown()
    simulation_app.close()


