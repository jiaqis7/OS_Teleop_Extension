# --------------------------------------------
# 1. CLI ARGUMENTS AND ISAAC SIM LAUNCH SETUP
# --------------------------------------------
import argparse
from omni.isaac.lab.app import AppLauncher

# Parse CLI arguments
parser = argparse.ArgumentParser(description="MTML+MTMR+PO Teleop")
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--task", type=str, default="Isaac-MTML-MTMR-PO-Teleop-v0")
parser.add_argument("--scale", type=float, default=0.4)
parser.add_argument("--is_simulated", type=bool, default=False)
parser.add_argument("--disable_viewport", action="store_true")
parser.add_argument("--log_trigger_file", type=str, default="log_trigger.txt")
parser.add_argument("--enable_logging", action="store_true")
parser.add_argument("--demo_name", type=str, default=None)

parser.add_argument(
    "--log_duration", type=float, default=22.0,
    help="Duration of logging in seconds"
)


AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch Isaac Sim App
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


# ---------------------------------------------------------
# 2. IMPORTS AND INITIAL SETUP FOR TELEOP, ENV, LOGGING
# ---------------------------------------------------------

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))


# CRITICAL: Set global config BEFORE launching app and creating environment
import global_cfg
global_cfg.log_duration = args_cli.log_duration

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
from teleop_interface.phantomomni.se3_phantomomni import PhantomOmniTeleop
from scripts.teleoperation.teleop_logger_3_arm import TeleopLogger, log_current_pose, reset_cube_pose


# -------------------------------
# 3. TELEOP UTILITIES
# -------------------------------

# --- [Gripper Map] ---
def get_jaw_gripper_angles(gripper_command, robot_name):
    if gripper_command is None:
        return np.array([0.0, 0.0])
    norm_cmd = np.interp(gripper_command, [-1.0, 1.0], [-1.72, 1.06])
    g2_angle = 0.52359 / (1.06 + 1.72) * (norm_cmd + 1.72)
    return np.array([-g2_angle, g2_angle])

# --- [Reset Helpers] ---
def reset_psms_to_initial_pose(env, saved_tips, base_matrices, num_steps=30):
    actions = []
    for i in range(3):
        T_world_tip = pose_to_transformation_matrix(*saved_tips[i])
        T_base = base_matrices[i]
        base_T_tip = np.linalg.inv(T_base) @ T_world_tip
        pos, quat = transformation_matrix_to_pose(base_T_tip)
        jaw = [0.0, 0.0]
        actions.extend([*pos, *quat, *jaw])
    action_tensor = torch.tensor(np.array(actions, dtype=np.float32), device=env.unwrapped.device).unsqueeze(0)
    for _ in range(num_steps):
        env.step(action_tensor)

# --- [Process Actions] ---
def process_actions(cam_T_psm1, w_T_psm1base, cam_T_psm2, w_T_psm2base,
                    cam_T_psm3, w_T_psm3base, w_T_cam, env, g1, g2, g3):
    def base_relative_pose(cam_T_tip, world_T_base):
        return transformation_matrix_to_pose(np.linalg.inv(world_T_base) @ w_T_cam @ cam_T_tip)

    a = []
    for cam_T, base, g, robot in zip([cam_T_psm1, cam_T_psm2, cam_T_psm3],
                                     [w_T_psm1base, w_T_psm2base, w_T_psm3base],
                                     [g1, g2, g3],
                                     ['robot_1', 'robot_2', 'robot_3']):
        pos, quat = base_relative_pose(cam_T, base)
        jaw = get_jaw_gripper_angles(g, robot)
        a.extend([*pos, *quat, *jaw])
    return torch.tensor(a, device=env.unwrapped.device).unsqueeze(0).repeat(env.unwrapped.num_envs, 1)


def _compute_base_relative_action(env, psm1, psm2, psm3, jaw1, jaw2, jaw3):
    """Helper to compute and return action tensor given PSMs and target jaw angles."""
    # Get world-frame tip poses
    psm1_tip_pos = psm1.data.body_link_pos_w[0][-1].cpu().numpy()
    psm1_tip_quat = psm1.data.body_link_quat_w[0][-1].cpu().numpy()
    psm2_tip_pos = psm2.data.body_link_pos_w[0][-1].cpu().numpy()
    psm2_tip_quat = psm2.data.body_link_quat_w[0][-1].cpu().numpy()
    psm3_tip_pos = psm3.data.body_link_pos_w[0][-1].cpu().numpy()
    psm3_tip_quat = psm3.data.body_link_quat_w[0][-1].cpu().numpy()
    world_T_psm1tip = pose_to_transformation_matrix(psm1_tip_pos, psm1_tip_quat)
    world_T_psm2tip = pose_to_transformation_matrix(psm2_tip_pos, psm2_tip_quat)
    world_T_psm3tip = pose_to_transformation_matrix(psm3_tip_pos, psm3_tip_quat)

    # Get base transforms
    psm1_base_link_pos = psm1.data.body_link_pos_w[0][0].cpu().numpy()
    psm1_base_link_quat = psm1.data.body_link_quat_w[0][0].cpu().numpy()
    psm2_base_link_pos = psm2.data.body_link_pos_w[0][0].cpu().numpy()
    psm2_base_link_quat = psm2.data.body_link_quat_w[0][0].cpu().numpy()
    psm3_base_link_pos = psm3.data.body_link_pos_w[0][0].cpu().numpy()
    psm3_base_link_quat = psm3.data.body_link_quat_w[0][0].cpu().numpy()
    world_T_psm1_base = pose_to_transformation_matrix(psm1_base_link_pos, psm1_base_link_quat)
    world_T_psm2_base = pose_to_transformation_matrix(psm2_base_link_pos, psm2_base_link_quat)
    world_T_psm3_base = pose_to_transformation_matrix(psm3_base_link_pos, psm3_base_link_quat)

    # Compute base-relative poses
    psm1_rel_pos, psm1_rel_quat = transformation_matrix_to_pose(np.linalg.inv(world_T_psm1_base) @ world_T_psm1tip)
    psm2_rel_pos, psm2_rel_quat = transformation_matrix_to_pose(np.linalg.inv(world_T_psm2_base) @ world_T_psm2tip)
    psm3_rel_pos, psm3_rel_quat = transformation_matrix_to_pose(np.linalg.inv(world_T_psm3_base) @ world_T_psm3tip)

    # Construct action tensor
    action_tensor = torch.tensor(
        np.concatenate([psm1_rel_pos, psm1_rel_quat, jaw1, psm2_rel_pos, psm2_rel_quat, jaw2, psm3_rel_pos, psm3_rel_quat, jaw3], dtype=np.float32),
        device=env.unwrapped.device
    ).unsqueeze(0)

    return action_tensor


def open_jaws(env, psm1, psm2, psm3, target=1.04):
    """Open PSM jaws to match a target angle."""
    jaw_angle = 0.52359 / (1.06 + 1.72) * (target + 1.72)
    jaw1 = [-jaw_angle, jaw_angle]
    jaw2 = [-jaw_angle, jaw_angle]
    jaw3 = [0, 0]  # PSM3 does not have jaws, so we set it to zero

    print(f"[INIT] Opening jaws to approximate MTM input: {target} → [{jaw1[0]:.3f}, {jaw1[1]:.3f}]")
    action_tensor = _compute_base_relative_action(env, psm1, psm2, psm3, jaw1=jaw1, jaw2=jaw2, jaw3=jaw3)
    for _ in range(30):
        env.step(action_tensor)
    print("[INIT] jaws opened.")


# --- [Main Loop] ---
def main():

    # Define robot name mapping
    psm_name_dict = {
        "PSM1": "robot_1",
        "PSM2": "robot_2",
        "PSM3": "robot_3"
    }

    # Initialize TeleopLogger
    teleop_logger = TeleopLogger(
        trigger_file=args_cli.log_trigger_file,
        psm_name_dict=psm_name_dict,
        log_duration=args_cli.log_duration  # seconds
    )

    scale = args_cli.scale
    is_simulated = args_cli.is_simulated

    mtm = MTMManipulator(); mtm.home()
    mtm_teleop = MTMTeleop(); mtm_teleop.reset()
    po_teleop = PhantomOmniTeleop(); po_teleop.reset()

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric)
    env_cfg.terminations.time_out = None
    env = gym.make(args_cli.task, cfg=env_cfg); env.reset()

    cam_left = env.unwrapped.scene["camera_left"]
    cam_right = env.unwrapped.scene["camera_right"]

    if not is_simulated and not args_cli.disable_viewport:
        view_port_l = vp_utils.create_viewport_window("Left Camera", width=800, height=600)
        view_port_l.viewport_api.camera_path = '/World/envs/env_0/Robot_4/ecm_end_link/camera_left'

        view_port_r = vp_utils.create_viewport_window("Right Camera", width=800, height=600)
        view_port_r.viewport_api.camera_path = '/World/envs/env_0/Robot_4/ecm_end_link/camera_right'

    psm1, psm2, psm3 = env.unwrapped.scene["robot_1"], env.unwrapped.scene["robot_2"], env.unwrapped.scene["robot_3"]

    saved_tips = [None] * 3
    was_in_mtm_clutch, was_in_po_clutch = True, True
    po_waiting_for_clutch = False
    mtm_synced = False

    

    print("[INFO] Waiting for MTM and PO clutch to initialize...")

    while simulation_app.is_running():
        start = time.time()
        cam_pos = ((cam_left.data.pos_w + cam_right.data.pos_w) / 2).cpu().numpy()[0]
        cam_quat = cam_left.data.quat_w_world.cpu().numpy()[0]
        world_T_cam = pose_to_transformation_matrix(cam_pos, cam_quat)
        cam_T_world = np.linalg.inv(world_T_cam)

        def get_T_base(robot):
            return pose_to_transformation_matrix(
                robot.data.body_link_pos_w[0][0].cpu().numpy(),
                robot.data.body_link_quat_w[0][0].cpu().numpy()
            )

        world_T_b = [get_T_base(r) for r in [psm1, psm2, psm3]]

        mtml_pos, mtml_rot, g1, mtmr_pos, mtmr_rot, g2, mtm_clutch, *_ = mtm_teleop.advance()
        if g1 is None: time.sleep(0.05); continue

        mtm_q1 = R.from_rotvec(mtml_rot).as_quat(); mtm_q1 = np.concatenate([[mtm_q1[3]], mtm_q1[:3]])
        mtm_q2 = R.from_rotvec(mtmr_rot).as_quat(); mtm_q2 = np.concatenate([[mtm_q2[3]], mtm_q2[:3]])

        if not mtm_synced:
            print("[SYNC] Aligning MTM orientation to PSM tips...")
            def tip_T(robot):
                return pose_to_transformation_matrix(
                    robot.data.body_link_pos_w[0][-1].cpu().numpy(),
                    robot.data.body_link_quat_w[0][-1].cpu().numpy()
                )
            mtm.adjust_orientation(mtm_teleop.simpose2hrsvpose(cam_T_world @ tip_T(psm1)),
                                   mtm_teleop.simpose2hrsvpose(cam_T_world @ tip_T(psm2)))
            saved_tips[0] = (psm1.data.body_link_pos_w[0][-1].cpu().numpy(), psm1.data.body_link_quat_w[0][-1].cpu().numpy())
            saved_tips[1] = (psm2.data.body_link_pos_w[0][-1].cpu().numpy(), psm2.data.body_link_quat_w[0][-1].cpu().numpy())
            saved_tips[2] = (psm3.data.body_link_pos_w[0][-1].cpu().numpy(), psm3.data.body_link_quat_w[0][-1].cpu().numpy())
            mtm_synced = True
            print("[READY] All MTMs aligned. Press and release clutch to start teleoperation.")
            continue

        stylus_pose, g3, po_clutch = po_teleop.advance()
        if po_clutch is None: time.sleep(0.05); continue
        po_q = R.from_rotvec(stylus_pose[3:]).as_quat(); po_q = np.concatenate([[po_q[3]], po_q[:3]])

        if not mtm_clutch:
            if was_in_mtm_clutch:
                init_mtml_pos = mtml_pos
                init_mtmr_pos = mtmr_pos
                def init_tip(robot): return (cam_T_world @ pose_to_transformation_matrix(
                    robot.data.body_link_pos_w[0][-1].cpu().numpy(),
                    robot.data.body_link_quat_w[0][-1].cpu().numpy()
                ))[:3, 3]
                init_psm1_tip_pos, init_psm2_tip_pos = init_tip(psm1), init_tip(psm2)
            p1_target = init_psm1_tip_pos + (mtml_pos - init_mtml_pos) * scale
            p2_target = init_psm2_tip_pos + (mtmr_pos - init_mtmr_pos) * scale
            cam_T_psm1 = pose_to_transformation_matrix(p1_target, mtm_q1)
            cam_T_psm2 = pose_to_transformation_matrix(p2_target, mtm_q2)
            was_in_mtm_clutch = False
        else:
            was_in_mtm_clutch = True
            def cur_tip(robot):
                return cam_T_world @ pose_to_transformation_matrix(
                    robot.data.body_link_pos_w[0][-1].cpu().numpy(),
                    robot.data.body_link_quat_w[0][-1].cpu().numpy()
                )
            cam_T_psm1, cam_T_psm2 = cur_tip(psm1), cur_tip(psm2)

        cam_T_psm3 = cam_T_world @ pose_to_transformation_matrix(
            psm3.data.body_link_pos_w[0][-1].cpu().numpy(),
            psm3.data.body_link_quat_w[0][-1].cpu().numpy()
        )

        if not po_clutch:
            if po_waiting_for_clutch:
                pass
            elif was_in_po_clutch:
                print("[PO] Clutch released. Initializing teleoperation.")
                init_stylus_position = stylus_pose[:3]
                tip_pose = psm3.data.body_link_pos_w[0][-1].cpu().numpy()
                tip_quat = psm3.data.body_link_quat_w[0][-1].cpu().numpy()
                world_T_psm3tip = pose_to_transformation_matrix(tip_pose, tip_quat)
                cam_T_psm3tip_init = cam_T_world @ world_T_psm3tip
                init_psm3_tip_position = cam_T_psm3tip_init[:3, 3]
                was_in_po_clutch = False

            if init_stylus_position is not None and init_psm3_tip_position is not None:
                stylus_orientation = R.from_rotvec(stylus_pose[3:]).as_quat()
                stylus_orientation = np.concatenate([[stylus_orientation[3]], stylus_orientation[:3]])
                psm3_target = init_psm3_tip_position + (stylus_pose[:3] - init_stylus_position) * scale
                cam_T_psm3 = pose_to_transformation_matrix(psm3_target, stylus_orientation)

        else:
            if po_waiting_for_clutch:
                print("[PO] Clutch pressed after reset. Ready to resume.")
                po_waiting_for_clutch = False
                was_in_po_clutch = True
            if not was_in_po_clutch:
                print("[PO] Clutch pressed. Freezing teleop.")
            was_in_po_clutch = True
            po_gripper = None


        actions = process_actions(
            cam_T_psm1, world_T_b[0], cam_T_psm2, world_T_b[1],
            cam_T_psm3, world_T_b[2], world_T_cam, env,
            g1, g2, g3
        )
        env.step(actions)

        # Check for RESET TRIGGER to randomize cube pose
        if os.path.exists("reset_trigger.txt"):
            print("[RESET] Trigger detected. Resetting cube and all PSMs...")

            cube_pos1 = [np.random.uniform(-0.01, 0.01), np.random.uniform(-0.01, 0.01), 0.0]
            cube_yaw1 = np.random.uniform(-np.pi/2, np.pi/2)
            cube_quat1 = R.from_euler("z", cube_yaw1).as_quat()
            cube_ori1 = [cube_quat1[3], cube_quat1[0], cube_quat1[1], cube_quat1[2]]
            reset_cube_pose(env, "teleop_logs/cube_latest_1", cube_pos1, cube_ori1)


            # Reset Cube 2
            cube_pos2 = [np.random.uniform(0.045, 0.055), np.random.uniform(0.0, 0.03), 0.0]
            cube_yaw2 = np.random.uniform(-np.pi / 2, np.pi / 2)
            cube_quat2 = R.from_euler("z", cube_yaw2).as_quat()
            cube_ori2 = [cube_quat2[3], cube_quat2[0], cube_quat2[1], cube_quat2[2]]
            reset_cube_pose(env, "teleop_logs/cube_latest_2", cube_pos2, cube_ori2, cube_key="cube_rigid_2")

            # Reset Cube 3
            cube_pos3 = [np.random.uniform(-0.055, -0.045), np.random.uniform(0.0, 0.03), 0.0]
            cube_yaw3 = np.random.uniform(-np.pi / 2, np.pi / 2)
            cube_quat3 = R.from_euler("z", cube_yaw3).as_quat()
            cube_ori3 = [cube_quat3[3], cube_quat3[0], cube_quat3[1], cube_quat3[2]]
            reset_cube_pose(env, "teleop_logs/cube_latest_3", cube_pos3, cube_ori3, cube_key="cube_rigid_3")


            cube_pos4 = [np.random.uniform(0.0, 0.0), np.random.uniform(0.055, 0.055), 0.0]
            cube_yaw4 = np.random.uniform(0, 0)
            cube_quat4 = R.from_euler("z", cube_yaw4).as_quat()
            cube_ori4 = [cube_quat4[3], cube_quat4[0], cube_quat4[1], cube_quat4[2]]
            reset_cube_pose(env, "teleop_logs/cube_latest_4", cube_pos4, cube_ori4, cube_key="cube_rigid_4")



            reset_psms_to_initial_pose(env, saved_tips, world_T_b, num_steps=30)

            mtm.home()
            time.sleep(2.0)
            mtm.adjust_orientation(
                mtm_teleop.simpose2hrsvpose(cam_T_world @ pose_to_transformation_matrix(*saved_tips[0])),
                mtm_teleop.simpose2hrsvpose(cam_T_world @ pose_to_transformation_matrix(*saved_tips[1]))
            )

            open_jaws(env, psm1, psm2, psm3, target=0.8)

            was_in_mtm_clutch = True
            was_in_po_clutch = True
            po_waiting_for_clutch = True
            init_stylus_position = None
            init_psm3_tip_position = None
            os.remove("reset_trigger.txt")
            print("[READY] Reset complete. Reclutch MTM and PO to resume independently.")


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

            cam_l_img = cam_left.data.output["rgb"][0].cpu().numpy()
            cam_r_img = cam_right.data.output["rgb"][0].cpu().numpy()

            teleop_logger.enqueue(frame_num, timestamp, robot_states, cam_l_img, cam_r_img)
            teleop_logger.frame_num = frame_num


        time.sleep(max(0.0, 1/30.0 - time.time() + start))

    env.close()
    teleop_logger.shutdown()
    simulation_app.close()

if __name__ == "__main__":
    main()
