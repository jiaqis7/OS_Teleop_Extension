import argparse
from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="MTM teleoperation for Custom MultiArm dVRK environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-Three-ACT-v0", help="Name of the task.")
parser.add_argument("--scale", type=float, default=0.4, help="Teleop scaling factor.")
parser.add_argument("--is_simulated", type=bool, default=False, help="Whether the MTM input is from the simulated model or not.")
parser.add_argument("--enable_logging", action="store_true", help="Enable logging from the start (default is off)")
parser.add_argument("--log_trigger_file", type=str, default="log_trigger.txt",
    help="Path to a file that enables logging when it exists.")
parser.add_argument("--disable_viewport", action="store_true", help="Disable extra viewport windows.")
parser.add_argument("--demo_name", type=str, default=None, help="Custom name for the logging folder (e.g., 'demo_1')")

# Extended model control choices
parser.add_argument(
    "--model_control",
    type=str,
    choices=["psm1", "psm2", "psm3", "psm12", "all", "none"],
    default="none",
    help="Choose which arm(s) are controlled by the model"
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

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
from scripts.teleoperation.teleop_logger_3_arm import TeleopLogger, reset_cube_pose, log_current_pose
import custom_envs
import global_cfg

from AdaptACT.procedures.controller import AutonomousController

# Configuration for control modes
CONTROL_CONFIGS = {
    "psm1": {"PSM1": "model", "PSM2": "human", "PSM3": "human"},
    "psm2": {"PSM1": "human", "PSM2": "model", "PSM3": "human"},
    "psm3": {"PSM1": "human", "PSM2": "human", "PSM3": "model"},
    "psm12": {"PSM1": "model", "PSM2": "model", "PSM3": "human"},
    "all": {"PSM1": "model", "PSM2": "model", "PSM3": "model"},
    "none": {"PSM1": "human", "PSM2": "human", "PSM3": "human"}
}

# Mapping from PSM names to robot scene names
PSM_TO_ROBOT = {
    "PSM1": "robot_1",
    "PSM2": "robot_2", 
    "PSM3": "robot_3"
}

# Constants
MODEL_TRIGGER_PATH = os.path.join(os.path.dirname(__file__), "model_trigger.txt")
RESET_TRIGGER_PATH = "reset_trigger.txt"

# Overwrite the value from CLI args
global_cfg.model_control = args_cli.model_control


class PSMActionGenerator:
    """Unified action generation system for all PSMs"""
    
    def __init__(self, env):
        self.env = env
        self.scale = args_cli.scale
        
    def get_jaw_gripper_angles(self, gripper_command):
        """Convert gripper command to jaw angles"""
        if gripper_command is None:
            return np.array([0.0, 0.0])
        norm_cmd = np.interp(gripper_command, [-1.0, 1.0], [-1.72, 1.06])
        angle = 0.52359 / (1.06 + 1.72) * (norm_cmd + 1.72)
        return np.array([-angle/2, angle/2])
    
    def get_jaw_angle(self, joint_pos):
        """Get single jaw angle from joint positions"""
        return abs(joint_pos[-2]) + abs(joint_pos[-1])
    
    def get_human_taskspace_action(self, psm_name, cam_T_psm_tip, world_T_psm_base, gripper_command):
        """Generate task-space action for human teleoperation"""
        # Convert world pose to base-relative
        base_T_tip = np.linalg.inv(world_T_psm_base) @ self.world_T_cam @ cam_T_psm_tip
        rel_pos, rel_quat = transformation_matrix_to_pose(base_T_tip)
        gripper_angles = self.get_jaw_gripper_angles(gripper_command)
        
        return np.concatenate([rel_pos, rel_quat, gripper_angles])  # 9D
    
    def get_model_jointspace_action(self, psm_index, model_output):
        """Extract joint-space action for specific PSM from model output"""
        start_idx = psm_index * 7  # Each PSM: 6 joints + 1 jaw
        joints = model_output[start_idx:start_idx + 6]
        jaw = model_output[start_idx + 6]
        gripper_angles = np.array([-jaw/2, jaw/2])
        
        return np.concatenate([joints, gripper_angles])  # 8D
    
    def get_frozen_action(self, psm_name, saved_actions):
        """Get frozen/saved action for PSM"""
        psm_saved = saved_actions[psm_name]
        return np.concatenate([psm_saved[:6], psm_saved[6:8]])  # 8D


class PSMController:
    """Unified controller for all PSMs with flexible control modes"""
    
    def __init__(self, env, control_config):
        self.env = env
        self.control_config = control_config
        self.action_generator = PSMActionGenerator(env)
        
        # Determine if we need mixed control (task-space + joint-space)
        self.is_mixed_control = any(mode == "human" for mode in control_config.values())
        
        # Calculate expected action dimension
        if self.is_mixed_control:
            # Mixed: task-space (9D) + joint-space (8D) per PSM
            human_count = sum(1 for mode in control_config.values() if mode == "human")
            model_count = sum(1 for mode in control_config.values() if mode == "model")
            self.action_dim = human_count * 9 + model_count * 8
        else:
            # All joint-space: 8D per PSM
            self.action_dim = len(control_config) * 8
    
    def get_actions(self, human_data=None, model_output=None, saved_actions=None):
        """Generate unified action vector based on control configuration"""
        actions = []
        
        for psm_name in ["PSM1", "PSM2", "PSM3"]:
            control_mode = self.control_config[psm_name]
            
            if control_mode == "human" and human_data:
                # Task-space control
                cam_T_tip = human_data["cam_T_tips"][psm_name]
                world_T_base = human_data["world_T_bases"][psm_name]
                gripper_cmd = human_data["gripper_commands"][psm_name]
                
                action = self.action_generator.get_human_taskspace_action(
                    psm_name, cam_T_tip, world_T_base, gripper_cmd
                )
                
            elif control_mode == "model" and model_output is not None:
                # Joint-space control
                psm_index = ["PSM1", "PSM2", "PSM3"].index(psm_name)
                action = self.action_generator.get_model_jointspace_action(psm_index, model_output)
                
            else:
                # Frozen/saved action
                action = self.action_generator.get_frozen_action(psm_name, saved_actions)
            
            actions.append(action)
        
        # Concatenate all actions
        final_actions = np.concatenate(actions)
        
        # Validate action dimension
        assert len(final_actions) == self.action_dim, f"Action length is {len(final_actions)} but expected {self.action_dim}."
        
        return torch.tensor(final_actions, device=self.env.unwrapped.device).repeat(self.env.unwrapped.num_envs, 1)


class PSMResetManager:
    """Handles reset operations for different control configurations"""
    
    def __init__(self, env, control_config):
        self.env = env
        self.control_config = control_config
        self.is_mixed_control = any(mode == "human" for mode in control_config.values())
    
    def reset_to_initial_poses(self, saved_poses, world_T_bases=None):
        """Reset PSMs to initial poses based on control configuration"""
        print(f"[RESET] Moving PSMs to saved initial poses (config: {self.control_config})...")
        
        actions = []
        
        for psm_name in ["PSM1", "PSM2", "PSM3"]:
            control_mode = self.control_config[psm_name]
            saved_data = saved_poses[psm_name]
            
            if control_mode == "human" and self.is_mixed_control:
                # Task-space reset
                world_T_tip = pose_to_transformation_matrix(
                    saved_data["tip_pos_w"], saved_data["tip_quat_w"]
                )
                base_T_tip = np.linalg.inv(world_T_bases[psm_name]) @ world_T_tip
                rel_pos, rel_quat = transformation_matrix_to_pose(base_T_tip)
                gripper_angles = np.array([0.0, 0.0])  # Default open
                action = np.concatenate([rel_pos, rel_quat, gripper_angles])
                
            else:
                # Joint-space reset
                action = np.concatenate([
                    saved_data["joint_action"][:6],
                    saved_data["joint_action"][6:8]
                ])
            
            actions.append(action)
        
        final_action = np.concatenate(actions)
        action_tensor = torch.tensor(final_action, device=self.env.unwrapped.device).unsqueeze(0)
        
        # Execute reset
        for _ in range(90):
            self.env.step(action_tensor)
        
        print("[RESET] PSMs moved to initial poses.")


def align_mtm_orientation_once(mtm_manipulator, mtm_interface, psm1, psm2, cam_T_world):
    """Align MTM orientation with PSM poses (only needed for human control)"""
    print("[ALIGN] Aligning MTMs orientation with PSMs pose...")
    
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
    
    print("[ALIGN] MTML/MTMR orientation set.")


def reorient_mtm_to_match_psm(mtm_manipulator, teleop_interface, saved_poses, cam_T_world):
    """Reorient MTM to match saved PSM poses"""
    print("[RESET] Reorienting MTM to match PSM tip pose...")
    
    world_T_psm1tip = pose_to_transformation_matrix(
        saved_poses["PSM1"]["tip_pos_w"], saved_poses["PSM1"]["tip_quat_w"]
    )
    world_T_psm2tip = pose_to_transformation_matrix(
        saved_poses["PSM2"]["tip_pos_w"], saved_poses["PSM2"]["tip_quat_w"]
    )
    
    cam_T_psm1tip = cam_T_world @ world_T_psm1tip
    cam_T_psm2tip = cam_T_world @ world_T_psm2tip
    
    hrsv_T_mtml = teleop_interface.simpose2hrsvpose(cam_T_psm1tip)
    hrsv_T_mtmr = teleop_interface.simpose2hrsvpose(cam_T_psm2tip)
    
    mtm_manipulator.home()
    time.sleep(2.0)
    mtm_manipulator.adjust_orientation(hrsv_T_mtml[:3, :3], hrsv_T_mtmr[:3, :3])
    
    print("[RESET] MTMs reoriented.")


def main():
    # Initialize control configuration
    control_config = CONTROL_CONFIGS[args_cli.model_control]
    needs_human_control = any(mode == "human" for mode in control_config.values())
    needs_model_control = any(mode == "model" for mode in control_config.values())
    
    print(f"[CONFIG] Control mode: {args_cli.model_control}")
    print(f"[CONFIG] Control configuration: {control_config}")
    
    # Setup hardware (only if needed)
    mtm_manipulator = None
    mtm_interface = None
    if needs_human_control:
        mtm_manipulator = MTMManipulator()
        mtm_manipulator.home()
        mtm_interface = MTMTeleop()
        mtm_interface.reset()
    
    # Setup environment
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, 
        use_fabric=not args_cli.disable_fabric
    )
    env_cfg.terminations.time_out = None
    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()
    
    # Setup model controller (only if needed)
    controller = None
    if needs_model_control:
        controller = AutonomousController.from_train_dir(
            train_dir="/home/stanford/Demo_collections1/Models/4_orbitsim_single_human_demos/Joint Control/20250602-222005_stupendous-skunk_train",
            ckpt_path="/home/stanford/Demo_collections1/Models/4_orbitsim_single_human_demos/Joint Control/20250602-222005_stupendous-skunk_train/policy_epoch_20000_seed_0.ckpt",
            ckpt_strategy="none",
            device=args_cli.device
        )
        controller.reset()
    
    # Setup cameras and viewports
    camera_l = env.unwrapped.scene["camera_left"]
    camera_r = env.unwrapped.scene["camera_right"]
    
    if not args_cli.disable_viewport:
        view_port_l = vp_utils.create_viewport_window("Left Camera", width=800, height=600)
        view_port_l.viewport_api.camera_path = '/World/envs/env_0/Robot_4/ecm_end_link/camera_left'
        view_port_r = vp_utils.create_viewport_window("Right Camera", width=800, height=600)
        view_port_r.viewport_api.camera_path = '/World/envs/env_0/Robot_4/ecm_end_link/camera_right'
    
    # Get PSM references
    psms = {
        "PSM1": env.unwrapped.scene["robot_1"],
        "PSM2": env.unwrapped.scene["robot_2"],
        "PSM3": env.unwrapped.scene["robot_3"]
    }
    
    # Initialize controllers
    psm_controller = PSMController(env, control_config)
    reset_manager = PSMResetManager(env, control_config)
    
    # State variables
    was_in_mtm_clutch = True
    orientation_aligned = False
    model_triggered = False
    printed_trigger = False
    
    # Save initial poses and actions
    saved_poses = {}
    for psm_name, psm in psms.items():
        saved_poses[psm_name] = {
            "tip_pos_w": None,
            "tip_quat_w": None,
            "joint_action": psm.data.joint_pos[0].cpu().numpy()
        }
    
    # Setup logging
    psm_name_dict = {"PSM1": "robot_1", "PSM2": "robot_2", "PSM3": "robot_3"}
    teleop_logger = TeleopLogger(
        trigger_file=args_cli.log_trigger_file,
        psm_name_dict=psm_name_dict,
        log_duration=30.0
    )
    
    print("Press the clutch button and release to start teleoperation.")
    
    while simulation_app.is_running():
        start_time = time.time()
        
        # Get camera transforms
        cam_pos = (camera_l.data.pos_w + camera_r.data.pos_w) / 2
        cam_quat = camera_l.data.quat_w_world
        world_T_cam = pose_to_transformation_matrix(cam_pos.cpu().numpy()[0], cam_quat.cpu().numpy()[0])
        cam_T_world = np.linalg.inv(world_T_cam)
        psm_controller.action_generator.world_T_cam = world_T_cam
        
        # Initial alignment (only if human control is needed)
        if not orientation_aligned and needs_human_control:
            align_mtm_orientation_once(mtm_manipulator, mtm_interface, psms["PSM1"], psms["PSM2"], cam_T_world)
            orientation_aligned = True
            
            # Save initial tip poses
            for psm_name, psm in psms.items():
                saved_poses[psm_name]["tip_pos_w"] = psm.data.body_link_pos_w[0][-1].cpu().numpy()
                saved_poses[psm_name]["tip_quat_w"] = psm.data.body_link_quat_w[0][-1].cpu().numpy()
        
        # Get human teleoperation data (only if needed)
        human_data = None
        if needs_human_control:
            # Get MTM data
            (mtml_pos, mtml_rot, l_gripper_joint, 
             mtmr_pos, mtmr_rot, r_gripper_joint, 
             mtm_clutch, _, _) = mtm_interface.advance()
            
            # Handle clutch logic
            if not mtm_clutch:
                if was_in_mtm_clutch:
                    init_mtml_pos = mtml_pos
                    init_mtmr_pos = mtmr_pos
                    
                    # Calculate initial PSM positions
                    init_psm_positions = {}
                    for psm_name, psm in psms.items():
                        world_T_tip = pose_to_transformation_matrix(
                            psm.data.body_link_pos_w[0][-1].cpu().numpy(),
                            psm.data.body_link_quat_w[0][-1].cpu().numpy()
                        )
                        init_psm_positions[psm_name] = (cam_T_world @ world_T_tip)[:3, 3]
                
                # Calculate target poses
                mtml_quat = R.from_rotvec(mtml_rot).as_quat()
                mtml_quat = np.concatenate([[mtml_quat[3]], mtml_quat[:3]])
                psm1_target_pos = init_psm_positions["PSM1"] + (mtml_pos - init_mtml_pos) * args_cli.scale
                cam_T_psm1tip = pose_to_transformation_matrix(psm1_target_pos, mtml_quat)
                
                mtmr_quat = R.from_rotvec(mtmr_rot).as_quat()
                mtmr_quat = np.concatenate([[mtmr_quat[3]], mtmr_quat[:3]])
                psm2_target_pos = init_psm_positions["PSM2"] + (mtmr_pos - init_mtmr_pos) * args_cli.scale
                cam_T_psm2tip = pose_to_transformation_matrix(psm2_target_pos, mtmr_quat)
                
                was_in_mtm_clutch = False
                
            else:
                was_in_mtm_clutch = True
                
                # Use current PSM poses
                for psm_name, psm in psms.items():
                    world_T_tip = pose_to_transformation_matrix(
                        psm.data.body_link_pos_w[0][-1].cpu().numpy(),
                        psm.data.body_link_quat_w[0][-1].cpu().numpy()
                    )
                    if psm_name == "PSM1":
                        cam_T_psm1tip = cam_T_world @ world_T_tip
                    elif psm_name == "PSM2":
                        cam_T_psm2tip = cam_T_world @ world_T_tip
            
            # Prepare human data
            world_T_bases = {}
            for psm_name, psm in psms.items():
                world_T_bases[psm_name] = pose_to_transformation_matrix(
                    psm.data.body_link_pos_w[0][0].cpu().numpy(),
                    psm.data.body_link_quat_w[0][0].cpu().numpy()
                )
            
            human_data = {
                "cam_T_tips": {
                    "PSM1": cam_T_psm1tip,
                    "PSM2": cam_T_psm2tip,
                    "PSM3": cam_T_world @ pose_to_transformation_matrix(
                        psms["PSM3"].data.body_link_pos_w[0][-1].cpu().numpy(),
                        psms["PSM3"].data.body_link_quat_w[0][-1].cpu().numpy()
                    )
                },
                "world_T_bases": world_T_bases,
                "gripper_commands": {
                    "PSM1": l_gripper_joint,
                    "PSM2": r_gripper_joint,
                    "PSM3": 0.0  # Default for PSM3
                }
            }
        
        # Handle logging
        teleop_logger.check_and_start_logging(env)
        teleop_logger.stop_logging()
        
        if teleop_logger.enable_logging and teleop_logger.frame_num == 0:
            log_current_pose(env, teleop_logger.log_dir)
        
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
        
        # Handle reset trigger
        if os.path.exists("reset_trigger.txt"):
            print("[RESET] Detected reset trigger. Resetting cube and all PSMs...")
            
            # Reset cubes (same as before)
            cube_configs = [
                ([np.random.uniform(-0.01, 0.01), np.random.uniform(-0.01, 0.01), 0.0], "teleop_logs/cube_latest_1", "cube_rigid"),
                ([np.random.uniform(0.045, 0.055), np.random.uniform(0.0, 0.03), 0.0], "teleop_logs/cube_latest_2", "cube_rigid_2"),
                ([np.random.uniform(-0.055, -0.045), np.random.uniform(0.0, 0.03), 0.0], "teleop_logs/cube_latest_3", "cube_rigid_3"),
                ([0.0, 0.055, 0.0], "teleop_logs/cube_latest_4", "cube_rigid_4")
            ]
            
            for i, (cube_pos, log_path, cube_key) in enumerate(cube_configs):
                cube_yaw = np.random.uniform(-np.pi/2, np.pi/2) if i < 3 else 0
                cube_quat = R.from_euler("z", cube_yaw).as_quat()
                cube_ori = [cube_quat[3], cube_quat[0], cube_quat[1], cube_quat[2]]
                reset_cube_pose(env, log_path, cube_pos, cube_ori, cube_key=cube_key if i > 0 else "cube_rigid")
            
            # Recompute transforms
            cam_pos = (camera_l.data.pos_w + camera_r.data.pos_w) / 2
            cam_quat = camera_l.data.quat_w_world
            world_T_cam = pose_to_transformation_matrix(cam_pos.cpu().numpy()[0], cam_quat.cpu().numpy()[0])
            cam_T_world = np.linalg.inv(world_T_cam)
            
            world_T_bases = {}
            for psm_name, psm in psms.items():
                world_T_bases[psm_name] = pose_to_transformation_matrix(
                    psm.data.body_link_pos_w[0][0].cpu().numpy(),
                    psm.data.body_link_quat_w[0][0].cpu().numpy()
                )
            
            # Reset PSMs
            reset_manager.reset_to_initial_poses(saved_poses, world_T_bases)
            
            if controller:
                controller.reset()
            
            # Reorient MTM if needed
            if needs_human_control:
                reorient_mtm_to_match_psm(mtm_manipulator, mtm_interface, saved_poses, cam_T_world)
            
            # Reset state
            was_in_mtm_clutch = True
            orientation_aligned = False
            model_triggered = False
            printed_trigger = False
            
            os.remove("reset_trigger.txt")
            print("[RESET] Reset complete. Reclutch both inputs to resume.")
            continue
        
        # Handle model trigger
        if os.path.exists(MODEL_TRIGGER_PATH) and not model_triggered:
            print(f"[MODEL] Trigger detected. Starting ACT control of {[k for k, v in control_config.items() if v == 'model']}.")
            model_triggered = True
            printed_trigger = True
            try:
                os.remove(MODEL_TRIGGER_PATH)
            except Exception as e:
                print(f"[MODEL] Failed to remove trigger file: {e}")
        
        # Get model output (only if needed and triggered)
        model_output = None
        if needs_model_control and model_triggered:
            try:
                # Get joint states for model input
                model_input_parts = []
                for psm_name in ["PSM1", "PSM2", "PSM3"]:
                    psm = psms[psm_name]
                    psm_q = psm.data.joint_pos[0].cpu().numpy()
                    jaw_angle = psm_controller.action_generator.get_jaw_angle(psm_q)
                    model_input_parts.extend([psm_q[:-2], [jaw_angle]])
                
                model_input = np.concatenate([np.concatenate(part) for part in model_input_parts])
                
                imgs = np.stack([
                    camera_l.data.output["rgb"][0].cpu().numpy(),
                    camera_r.data.output["rgb"][0].cpu().numpy()
                ]) / 255.0
                
                # ACT model inference
                model_output = controller.step(imgs, model_input)
                
                # Save last successful output
                last_model_output = model_output
                
            except RuntimeError as e:
                if "Rollout was already completed" in str(e):
                    if printed_trigger:
                        controlled_psms = [k for k, v in control_config.items() if v == 'model']
                        print(f"[MODEL] ACT rollout completed. Freezing controlled PSM(s): {controlled_psms} at last pose.")
                        printed_trigger = False
                    model_triggered = False
                    model_output = last_model_output if 'last_model_output' in locals() else None
                else:
                    raise
        
        # Generate actions using the unified controller
        actions = psm_controller.get_actions(
            human_data=human_data,
            model_output=model_output,
            saved_actions=saved_poses
        )
        
        # Step the environment
        env.step(actions)
        time.sleep(max(0.0, 1 / 200.0 - (time.time() - start_time)))
    
    # Cleanup
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