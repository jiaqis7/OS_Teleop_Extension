import argparse

from omni.isaac.lab.app import AppLauncher



from teleop_logger import TeleopLogger, reset_cube_pose, log_current_pose


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
import custom_envs

from AdaptACT.procedures.controller import AutonomousController

MODEL_TRIGGER_PATH = os.path.join(os.path.dirname(__file__), "model_trigger.txt")


def get_jaw_gripper_angles(gripper_command):
    if gripper_command is None:
        return np.array([0.0, 0.0])
    norm_cmd = np.interp(gripper_command, [-1.0, 1.0], [-1.72, 1.06])
    g2_angle = 0.52359 / (1.06 + 1.72) * (norm_cmd + 1.72)
    return np.array([-g2_angle, g2_angle])


def process_actions(psm1_joint_action, cam_T_psm2, w_T_psm2base, w_T_cam, gripper2_command, env):
    psm2base_T_psm2 = np.linalg.inv(w_T_psm2base) @ w_T_cam @ cam_T_psm2
    psm2_rel_pos, psm2_rel_quat = transformation_matrix_to_pose(psm2base_T_psm2)

    g2_angles = get_jaw_gripper_angles(gripper2_command)
    
    actions = np.concatenate([
        psm1_joint_action[:6],     # arm joints
        psm1_joint_action[6:8],    # model-controlled gripper
        psm2_rel_pos,
        psm2_rel_quat,
        g2_angles                  # teleop gripper
    ])
    return torch.tensor(actions, device=env.unwrapped.device).repeat(env.unwrapped.num_envs, 1)




def align_mtm_orientation_once(mtm_manipulator, mtm_interface, psm2, cam_T_world):
    print("[ALIGN] Aligning MTMR orientation with PSM2 pose...")
    psm2_tip_pose = pose_to_transformation_matrix(
        psm2.data.body_link_pos_w[0][-1].cpu().numpy(),
        psm2.data.body_link_quat_w[0][-1].cpu().numpy()
    )
    hrsv_T_mtmr = mtm_interface.simpose2hrsvpose(cam_T_world @ psm2_tip_pose)
    mtm_manipulator.adjust_orientation_right(hrsv_T_mtmr[:3, :3])
    print("[ALIGN] MTMR orientation set.")




def reset_psms_to_initial_pose(env, saved_psm1_joint_action,
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
    model_trigger_enabled = False
    already_printed_model_trigger = False

    # Initial pose cache
    saved_psm1_tip_pos = None
    saved_psm1_tip_quat = None 
    saved_psm2_tip_pos = None
    saved_psm2_tip_quat = None

    psm_name_dict = {
        "PSM1": "robot_1",
        "PSM2": "robot_2"
    }

    # Capture PSM1 joint pose immediately after reset to keep it steady
    saved_psm1_joint_action = psm1.data.joint_pos[0].cpu().numpy()


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

        # Handle clutch release logic
        if not mtm_clutch:
            if was_in_mtm_clutch:
                init_mtmr_pos = mtmr_pos
                tip = pose_to_transformation_matrix(
                    psm2.data.body_link_pos_w[0][-1].cpu().numpy(),
                    psm2.data.body_link_quat_w[0][-1].cpu().numpy()
                )
                init_psm2_tip_pos = (cam_T_world @ tip)[:3, 3]
                

            mtmr_quat = R.from_rotvec(mtmr_rot).as_quat()
            mtmr_quat = np.concatenate([[mtmr_quat[3]], mtmr_quat[:3]])
            psm2_target_pos = init_psm2_tip_pos + (mtmr_pos - init_mtmr_pos) * scale
            cam_T_psm2tip = pose_to_transformation_matrix(psm2_target_pos, mtmr_quat)
            was_in_mtm_clutch = False
        else:
            was_in_mtm_clutch = True
            tip = pose_to_transformation_matrix(
                psm2.data.body_link_pos_w[0][-1].cpu().numpy(),
                psm2.data.body_link_quat_w[0][-1].cpu().numpy()
            )
            cam_T_psm2tip = cam_T_world @ tip

        # Enable ACT control when model_trigger file appears
        model_trigger_enabled |= os.path.exists(MODEL_TRIGGER_PATH)
        if model_trigger_enabled and not already_printed_model_trigger:
            print("[MODEL] Trigger detected. Starting ACT control of PSM1.")
            already_printed_model_trigger = True
            try:
                os.remove(MODEL_TRIGGER_PATH)
                print("[MODEL] Trigger file removed after activation.")
            except Exception as e:
                print(f"[MODEL] Failed to remove trigger file: {e}")

        if model_trigger_enabled:
            psm1_obs = {
                "qpos": psm1.data.joint_pos[0].cpu().numpy(),
                "images": np.stack([
                    camera_l.data.output["rgb"][0].cpu().numpy(),
                    camera_r.data.output["rgb"][0].cpu().numpy()
                ]) / 255.0
            }
            psm1_act = controller.step(psm1_obs["images"], psm1_obs["qpos"])
        else:
            psm1_act = saved_psm1_joint_action

        psm2_base = pose_to_transformation_matrix(
            psm2.data.body_link_pos_w[0][0].cpu().numpy(),
            psm2.data.body_link_quat_w[0][0].cpu().numpy()
        )

        actions = process_actions(
            psm1_joint_action=psm1_act,
            cam_T_psm2=cam_T_psm2tip,
            w_T_psm2base=psm2_base,
            w_T_cam=world_T_cam,
            gripper2_command=r_gripper_joint,
            env=env
        )



        env.step(actions)

        elapsed = time.time() - start_time
        time.sleep(max(0.0, (1 / 200.0) - elapsed))

    env.close()
    if os.path.exists(args_cli.log_trigger_file):
        os.remove(args_cli.log_trigger_file)



if __name__ == "__main__":
    main()
    simulation_app.close()

