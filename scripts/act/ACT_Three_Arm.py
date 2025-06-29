# import argparse
# import os
# import sys
# import time
# from omni.isaac.lab.app import AppLauncher

# # add argparse arguments
# parser = argparse.ArgumentParser(description="MTM+Model teleoperation for Custom MultiArm dVRK environments.")
# parser.add_argument("--disable_fabric", action="store_true", default=False)
# parser.add_argument("--num_envs", type=int, default=1)
# parser.add_argument("--task", type=str, default="Isaac-Three-ACT-v0")
# parser.add_argument("--scale", type=float, default=0.4)
# parser.add_argument("--is_simulated", type=bool, default=False)
# parser.add_argument("--enable_logging", action="store_true")
# parser.add_argument("--log_trigger_file", type=str, default="log_trigger.txt")
# parser.add_argument("--disable_viewport", action="store_true")
# parser.add_argument("--demo_name", type=str, default=None)

# # Model control argument
# parser.add_argument(
#     "--model_control",
#     type=str,
#     choices=["psm1", "psm2", "psm3", "psm12", "all", "none"],
#     default="none",
#     help="Choose which arm(s) are controlled by the model"
# )

# # Model paths
# parser.add_argument("--model_train_dir", type=str, 
#     default="/home/stanford/Demo_collections1/Models/4_orbitsim_single_human_demos/Joint Control/20250602-222005_stupendous-skunk_train",
#     help="Path to model training directory")
# parser.add_argument("--model_ckpt_path", type=str,
#     default="/home/stanford/Demo_collections1/Models/4_orbitsim_single_human_demos/Joint Control/20250602-222005_stupendous-skunk_train/policy_epoch_20000_seed_0.ckpt",
#     help="Path to model checkpoint")

# AppLauncher.add_app_launcher_args(parser)
# args_cli = parser.parse_args()

# # launch omniverse app
# app_launcher = AppLauncher(args_cli)
# simulation_app = app_launcher.app

# # Fix import path
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# import gymnasium as gym
# import torch
# import numpy as np
# from scipy.spatial.transform import Rotation as R
# import omni.kit.viewport.utility as vp_utils

# from omni.isaac.lab_tasks.utils import parse_env_cfg
# from scripts.teleoperation.tf_utils import pose_to_transformation_matrix, transformation_matrix_to_pose
# from teleop_interface.MTM.se3_mtm import MTMTeleop
# from teleop_interface.phantomomni.se3_phantomomni import PhantomOmniTeleop
# from teleop_interface.MTM.mtm_manipulator import MTMManipulator
# from scripts.teleoperation.teleop_logger_3_arm import TeleopLogger, reset_cube_pose, log_current_pose
# import custom_envs

# # Import model controller only if needed
# try:
#     from AdaptACT.procedures.controller import AutonomousController
#     MODEL_AVAILABLE = True
# except ImportError:
#     print("[WARNING] AdaptACT not available. Model control disabled.")
#     MODEL_AVAILABLE = False

# # Configuration for control modes
# CONTROL_CONFIGS = {
#     "psm1": {"PSM1": "model", "PSM2": "human", "PSM3": "human"},
#     "psm2": {"PSM1": "human", "PSM2": "model", "PSM3": "human"},
#     "psm3": {"PSM1": "human", "PSM2": "human", "PSM3": "model"},
#     "psm12": {"PSM1": "model", "PSM2": "model", "PSM3": "human"},
#     "all": {"PSM1": "model", "PSM2": "model", "PSM3": "model"},
#     "none": {"PSM1": "human", "PSM2": "human", "PSM3": "human"}
# }

# # Constants
# MODEL_TRIGGER_PATH = os.path.join(os.path.dirname(__file__), "model_trigger.txt")
# RESET_TRIGGER_PATH = "reset_trigger.txt"


import argparse
import os
import sys
import time
from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="MTM+Model teleoperation for Custom MultiArm dVRK environments.")
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--task", type=str, default="Isaac-Three-ACT-v0")
parser.add_argument("--scale", type=float, default=0.4)
parser.add_argument("--is_simulated", type=bool, default=False)
parser.add_argument("--enable_logging", action="store_true")
parser.add_argument("--log_trigger_file", type=str, default="log_trigger.txt")
parser.add_argument("--disable_viewport", action="store_true")
parser.add_argument("--demo_name", type=str, default=None)

# Model control argument
parser.add_argument(
    "--model_control",
    type=str,
    choices=["psm1", "psm2", "psm3", "psm12", "all", "none"],
    default="none",
    help="Choose which arm(s) are controlled by the model"
)

# Model paths
parser.add_argument("--model_train_dir", type=str,
    default="/home/stanford/Demo_collections1/Models/4_orbitsim_single_human_demos/Joint Control/20250602-222005_stupendous-skunk_train",
    help="Path to model training directory")
parser.add_argument("--model_ckpt_path", type=str,
    default="/home/stanford/Demo_collections1/Models/4_orbitsim_single_human_demos/Joint Control/20250602-222005_stupendous-skunk_train/policy_epoch_20000_seed_0.ckpt",
    help="Path to model checkpoint")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# CRITICAL: Set global config BEFORE launching app and creating environment
import global_cfg
global_cfg.model_control = args_cli.model_control

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Fix import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import gymnasium as gym
import torch
import numpy as np
from scipy.spatial.transform import Rotation as R
import omni.kit.viewport.utility as vp_utils

from omni.isaac.lab_tasks.utils import parse_env_cfg
from scripts.teleoperation.tf_utils import pose_to_transformation_matrix, transformation_matrix_to_pose
from teleop_interface.MTM.se3_mtm import MTMTeleop
from teleop_interface.phantomomni.se3_phantomomni import PhantomOmniTeleop
from teleop_interface.MTM.mtm_manipulator import MTMManipulator
from scripts.teleoperation.teleop_logger_3_arm import TeleopLogger, reset_cube_pose, log_current_pose
import custom_envs

# Import model controller only if needed
try:
    from AdaptACT.procedures.controller import AutonomousController
    MODEL_AVAILABLE = True
except ImportError:
    print("[WARNING] AdaptACT not available. Model control disabled.")
    MODEL_AVAILABLE = False

# Configuration for control modes
CONTROL_CONFIGS = {
    "psm1": {"PSM1": "model", "PSM2": "human", "PSM3": "human"},
    "psm2": {"PSM1": "human", "PSM2": "model", "PSM3": "human"},
    "psm3": {"PSM1": "human", "PSM2": "human", "PSM3": "model"},
    "psm12": {"PSM1": "model", "PSM2": "model", "PSM3": "human"},
    "all": {"PSM1": "model", "PSM2": "model", "PSM3": "model"},
    "none": {"PSM1": "human", "PSM2": "human", "PSM3": "human"}
}

# Constants
MODEL_TRIGGER_PATH = os.path.join(os.path.dirname(__file__), "model_trigger.txt")
RESET_TRIGGER_PATH = "reset_trigger.txt"


class TeleopActionGenerator:
    """Generate actions for human teleoperation (task-space control)"""
    
    def __init__(self, scale=0.4):
        self.scale = scale
    
    def get_jaw_gripper_angles(self, gripper_command):
        """Convert gripper command to jaw angles"""
        if gripper_command is None:
            return np.array([0.0, 0.0])
        norm_cmd = np.interp(gripper_command, [-1.0, 1.0], [-1.72, 1.06])
        g2_angle = 0.52359 / (1.06 + 1.72) * (norm_cmd + 1.72)
        return np.array([-g2_angle, g2_angle])
    
    def process_actions(self, cam_T_psm1, w_T_psm1base, cam_T_psm2, w_T_psm2base,
                       cam_T_psm3, w_T_psm3base, w_T_cam, env, g1, g2, g3):
        """Process teleoperation actions - same as original working code"""
        def base_relative_pose(cam_T_tip, world_T_base):
            return transformation_matrix_to_pose(np.linalg.inv(world_T_base) @ w_T_cam @ cam_T_tip)

        actions = []
        for cam_T, base, g in zip([cam_T_psm1, cam_T_psm2, cam_T_psm3],
                                 [w_T_psm1base, w_T_psm2base, w_T_psm3base],
                                 [g1, g2, g3]):
            pos, quat = base_relative_pose(cam_T, base)
            jaw = self.get_jaw_gripper_angles(g)
            actions.extend([*pos, *quat, *jaw])
        
        return torch.tensor(actions, device=env.unwrapped.device).unsqueeze(0).repeat(env.unwrapped.num_envs, 1)


class ModelActionGenerator:
    """Generate actions for model control (joint-space control)"""
    
    def __init__(self):
        pass
    
    def get_jaw_angle(self, joint_pos):
        """Get single jaw angle from joint positions"""
        return abs(joint_pos[-2]) + abs(joint_pos[-1])
    
    def get_model_jointspace_action(self, psm_index, model_output):
        """Extract joint-space action for specific PSM from model output"""
        start_idx = psm_index * 7  # Each PSM: 6 joints + 1 jaw
        joints = model_output[start_idx:start_idx + 6]
        jaw = model_output[start_idx + 6]
        gripper_angles = np.array([-jaw/2, jaw/2])
        return np.concatenate([joints, gripper_angles])  # 8D


class MixedController:
    """Controller that handles both human teleoperation and model control with correct action dimensions"""
    
    def __init__(self, env, control_config, scale=0.4):
        self.env = env
        self.control_config = control_config
        self.teleop_generator = TeleopActionGenerator(scale)
        self.model_generator = ModelActionGenerator()
        
        # Determine control modes
        self.needs_human = any(mode == "human" for mode in control_config.values())
        self.needs_model = any(mode == "model" for mode in control_config.values())
        self.is_mixed = self.needs_human and self.needs_model
        
        # Calculate expected action dimension based on control config
        self.action_dim = 0
        for psm_name in ["PSM1", "PSM2", "PSM3"]:
            if control_config[psm_name] == "human":
                self.action_dim += 9  # 7 task space + 2 jaw
            else:  # model or saved
                self.action_dim += 8  # 6 joint space + 2 jaw
        
        print(f"[CONTROLLER] Mixed control: {self.is_mixed}, Human: {self.needs_human}, Model: {self.needs_model}")
        print(f"[CONTROLLER] Expected action dimension: {self.action_dim}")
    
    def get_actions(self, human_data=None, model_output=None, saved_poses=None):
        """Generate actions with correct dimensions based on control configuration"""
        if not self.is_mixed:
            # Pure human control - use original teleop logic (all 9D)
            if self.needs_human and human_data:
                return self.teleop_generator.process_actions(**human_data)
            # Pure model control - all joint space (all 8D)
            elif self.needs_model and model_output is not None:
                return self._get_pure_model_actions(model_output)
            else:
                # Fallback to saved poses
                return self._get_saved_actions(saved_poses)
        else:
            # Mixed control - combine different action types with correct dimensions
            return self._get_mixed_actions(human_data, model_output, saved_poses)
    
    def _get_pure_model_actions(self, model_output):
        """Generate pure model actions (all 8D joint space)"""
        actions = []
        for i in range(3):  # 3 PSMs
            model_action = self.model_generator.get_model_jointspace_action(i, model_output)
            actions.extend(model_action)  # 8D: 6 joints + 2 jaw
        
        return torch.tensor(actions, device=self.env.unwrapped.device).unsqueeze(0).repeat(self.env.unwrapped.num_envs, 1)
    
    def _get_mixed_actions(self, human_data, model_output, saved_poses):
        """Handle mixed human+model control with correct action dimensions per PSM"""
        actions = []
        
        for psm_name in ["PSM1", "PSM2", "PSM3"]:
            control_mode = self.control_config[psm_name]
            
            if control_mode == "human" and human_data:
                # Generate 9D task-space action for this PSM
                psm_action = self._get_single_psm_human_action(psm_name, human_data)
                actions.extend(psm_action)  # 9D (7 task space + 2 jaw)
                
            elif control_mode == "model" and model_output is not None:
                # Generate 8D joint-space action for this PSM
                psm_index = ["PSM1", "PSM2", "PSM3"].index(psm_name)
                model_action = self.model_generator.get_model_jointspace_action(psm_index, model_output)
                actions.extend(model_action)  # 8D (6 joint space + 2 jaw)
                
            else:
                # Use saved pose - generate action with correct dimension based on control mode
                if control_mode == "human":
                    # Generate 9D action from saved pose
                    saved_action = self._saved_pose_to_taskspace(psm_name, saved_poses)
                    actions.extend(saved_action)  # 9D
                else:
                    # Generate 8D action from saved pose  
                    saved_action = self._saved_pose_to_jointspace(psm_name, saved_poses)
                    actions.extend(saved_action)  # 8D
        
        # Verify action dimension matches expectation
        assert len(actions) == self.action_dim, f"Action length {len(actions)} != expected {self.action_dim}"
        
        return torch.tensor(actions, device=self.env.unwrapped.device).unsqueeze(0).repeat(self.env.unwrapped.num_envs, 1)
    
    def _get_single_psm_human_action(self, psm_name, human_data):
        """Extract 9D task-space action for a single PSM from human data"""
        # Extract the correct transforms and gripper for this PSM
        if psm_name == "PSM1":
            cam_T_tip = human_data['cam_T_psm1']
            world_T_base = human_data['w_T_psm1base']
            gripper = human_data['g1']
        elif psm_name == "PSM2":
            cam_T_tip = human_data['cam_T_psm2']
            world_T_base = human_data['w_T_psm2base']
            gripper = human_data['g2']
        else:  # PSM3
            cam_T_tip = human_data['cam_T_psm3']
            world_T_base = human_data['w_T_psm3base']
            gripper = human_data['g3']
        
        # Convert to base-relative pose (same logic as teleop_generator)
        world_T_cam = human_data['w_T_cam']
        base_T_tip = np.linalg.inv(world_T_base) @ world_T_cam @ cam_T_tip
        pos, quat = transformation_matrix_to_pose(base_T_tip)
        jaw = self.teleop_generator.get_jaw_gripper_angles(gripper)
        
        return np.concatenate([pos, quat, jaw])  # 9D: 3 pos + 4 quat + 2 jaw
    
    def _saved_pose_to_taskspace(self, psm_name, saved_poses):
        """Convert saved pose to 9D task-space action"""
        saved = saved_poses[psm_name]["joint_action"]
        # For simplicity, use joint positions as pose approximation
        pos = saved[:3]  # First 3 joints as position
        quat = [1, 0, 0, 0]  # Default quaternion
        jaw = saved[6:8]  # Existing jaw angles
        return np.concatenate([pos, quat, jaw])  # 9D
    
    def _saved_pose_to_jointspace(self, psm_name, saved_poses):
        """Convert saved pose to 8D joint-space action"""
        saved = saved_poses[psm_name]["joint_action"]
        return saved[:8]  # 8D: 6 joints + 2 jaw
    
    def _get_saved_actions(self, saved_poses):
        """Generate actions from saved poses based on control configuration"""
        actions = []
        for psm_name in ["PSM1", "PSM2", "PSM3"]:
            control_mode = self.control_config[psm_name]
            
            if control_mode == "human":
                # Generate 9D action
                saved_action = self._saved_pose_to_taskspace(psm_name, saved_poses)
                actions.extend(saved_action)
            else:
                # Generate 8D action
                saved_action = self._saved_pose_to_jointspace(psm_name, saved_poses)
                actions.extend(saved_action)
        
        return torch.tensor(actions, device=self.env.unwrapped.device).unsqueeze(0).repeat(self.env.unwrapped.num_envs, 1)
    


# def reset_psms_to_initial_pose(env, saved_tips, base_matrices, num_steps=30):
#     """Reset PSMs to initial poses"""
#     actions = []
#     for i in range(3):
#         T_world_tip = pose_to_transformation_matrix(*saved_tips[i])
#         T_base = base_matrices[i]
#         base_T_tip = np.linalg.inv(T_base) @ T_world_tip
#         pos, quat = transformation_matrix_to_pose(base_T_tip)
#         jaw = [0.0, 0.0]
#         actions.extend([*pos, *quat, *jaw])
    
#     action_tensor = torch.tensor(np.array(actions, dtype=np.float32), device=env.unwrapped.device).unsqueeze(0)
#     for _ in range(num_steps):
#         env.step(action_tensor)


def reset_psms_to_initial_pose(env, saved_tips, base_matrices, control_config, num_steps=30):
    """Reset PSMs to initial poses with correct action dimensions"""
    actions = []
    
    for i, psm_name in enumerate(["PSM1", "PSM2", "PSM3"]):
        T_world_tip = pose_to_transformation_matrix(*saved_tips[i])
        T_base = base_matrices[i]
        base_T_tip = np.linalg.inv(T_base) @ T_world_tip
        
        if control_config[psm_name] == "human":
            # 9D task-space action
            pos, quat = transformation_matrix_to_pose(base_T_tip)
            jaw = [0.0, 0.0]
            actions.extend([*pos, *quat, *jaw])  # 9D
        else:
            # 8D joint-space action - use saved joint positions
            joint_pos = env.unwrapped.scene[f"robot_{i+1}"].data.joint_pos[0].cpu().numpy()
            actions.extend(joint_pos[:8])  # 8D: 6 joints + 2 jaw
    
    action_tensor = torch.tensor(np.array(actions, dtype=np.float32), device=env.unwrapped.device).unsqueeze(0)
    for _ in range(num_steps):
        env.step(action_tensor)


def main():
    # Initialize control configuration
    control_config = CONTROL_CONFIGS[args_cli.model_control]
    needs_human_control = any(mode == "human" for mode in control_config.values())
    needs_model_control = any(mode == "model" for mode in control_config.values())
    
    print(f"[CONFIG] Control mode: {args_cli.model_control}")
    print(f"[CONFIG] Control configuration: {control_config}")
    
    # Validate model availability
    if needs_model_control and not MODEL_AVAILABLE:
        print("[ERROR] Model control requested but AdaptACT not available!")
        return
    
    # Setup hardware (only if needed)
    mtm_manipulator = None
    mtm_interface = None
    po_teleop = None
    
    if needs_human_control:
        try:
            mtm_manipulator = MTMManipulator()
            mtm_manipulator.home()
            mtm_interface = MTMTeleop()
            mtm_interface.reset()
            po_teleop = PhantomOmniTeleop()
            po_teleop.reset()
            print("[HARDWARE] MTM and Phantom Omni initialized successfully")
        except Exception as e:
            print(f"[ERROR] Hardware initialization failed: {e}")
            return
    
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
    last_model_output = None  # Initialize to prevent undefined variable
    
    if needs_model_control:
        try:
            controller = AutonomousController.from_train_dir(
                train_dir=args_cli.model_train_dir,
                ckpt_path=args_cli.model_ckpt_path,
                ckpt_strategy="none",
                device=args_cli.device
            )
            controller.reset()
            print("[MODEL] Controller loaded successfully")
        except Exception as e:
            print(f"[ERROR] Model controller initialization failed: {e}")
            return
    
    # Setup cameras and viewports
    camera_l = env.unwrapped.scene["camera_left"]
    camera_r = env.unwrapped.scene["camera_right"]
    
    if not args_cli.disable_viewport:
        view_port_l = vp_utils.create_viewport_window("Left Camera", width=800, height=600)
        view_port_l.viewport_api.camera_path = '/World/envs/env_0/Robot_4/ecm_end_link/camera_left'
        view_port_r = vp_utils.create_viewport_window("Right Camera", width=800, height=600)
        view_port_r.viewport_api.camera_path = '/World/envs/env_0/Robot_4/ecm_end_link/camera_right'
    
    # Get PSM references
    psm1 = env.unwrapped.scene["robot_1"]
    psm2 = env.unwrapped.scene["robot_2"] 
    psm3 = env.unwrapped.scene["robot_3"]
    psms = {"PSM1": psm1, "PSM2": psm2, "PSM3": psm3}
    
    # Initialize mixed controller
    mixed_controller = MixedController(env, control_config, args_cli.scale)
    
    # State variables
    saved_tips = [None] * 3
    was_in_mtm_clutch = True
    was_in_po_clutch = True
    po_waiting_for_clutch = False
    mtm_synced = False
    model_triggered = False
    printed_trigger = False


    # Initialize saved poses
    saved_poses = {}
    for psm_name, psm in psms.items():
        saved_poses[psm_name] = {
            "tip_pos_w": None,
            "tip_quat_w": None,
            "joint_action": psm.data.joint_pos[0].cpu().numpy()
        }


    # ALWAYS save initial poses regardless of control mode
    print("[INIT] Saving initial PSM poses...")
    for i, psm in enumerate([psm1, psm2, psm3]):
        saved_tips[i] = (
            psm.data.body_link_pos_w[0][-1].cpu().numpy(),
            psm.data.body_link_quat_w[0][-1].cpu().numpy()
        )
        psm_name = f"PSM{i+1}"
        saved_poses[psm_name]["tip_pos_w"] = saved_tips[i][0]
        saved_poses[psm_name]["tip_quat_w"] = saved_tips[i][1]
    
    
    # Setup logging
    psm_name_dict = {"PSM1": "robot_1", "PSM2": "robot_2", "PSM3": "robot_3"}
    teleop_logger = TeleopLogger(
        trigger_file=args_cli.log_trigger_file,
        psm_name_dict=psm_name_dict,
        log_duration=22.0
    )
    
    print("[INFO] System initialized. Waiting for input...")
    if needs_human_control:
        print("[INFO] Press and release MTM clutch to start teleoperation.")
    if needs_model_control:
        print(f"[INFO] Create '{MODEL_TRIGGER_PATH}' to start model control.")
    
    # Main control loop
    while simulation_app.is_running():
        start_time = time.time()
        
        # Get camera transforms
        cam_pos = ((camera_l.data.pos_w + camera_r.data.pos_w) / 2).cpu().numpy()[0]
        cam_quat = camera_l.data.quat_w_world.cpu().numpy()[0]
        world_T_cam = pose_to_transformation_matrix(cam_pos, cam_quat)
        cam_T_world = np.linalg.inv(world_T_cam)
        
        # Get base transforms
        def get_T_base(robot):
            return pose_to_transformation_matrix(
                robot.data.body_link_pos_w[0][0].cpu().numpy(),
                robot.data.body_link_quat_w[0][0].cpu().numpy()
            )
        world_T_b = [get_T_base(r) for r in [psm1, psm2, psm3]]
        
        # Handle human teleoperation data
        human_data = None
        if needs_human_control:
            # Get MTM data
            mtml_pos, mtml_rot, g1, mtmr_pos, mtmr_rot, g2, mtm_clutch, *_ = mtm_interface.advance()
            if g1 is None:
                time.sleep(0.05)
                continue
            
            # MTM synchronization
            if not mtm_synced:
                print("[SYNC] Aligning MTM orientation to PSM tips...")
                def tip_T(robot):
                    return pose_to_transformation_matrix(
                        robot.data.body_link_pos_w[0][-1].cpu().numpy(),
                        robot.data.body_link_quat_w[0][-1].cpu().numpy()
                    )
                mtm_manipulator.adjust_orientation(
                    mtm_interface.simpose2hrsvpose(cam_T_world @ tip_T(psm1)),
                    mtm_interface.simpose2hrsvpose(cam_T_world @ tip_T(psm2))
                )
                
                # Save initial poses
                for i, psm in enumerate([psm1, psm2, psm3]):
                    saved_tips[i] = (
                        psm.data.body_link_pos_w[0][-1].cpu().numpy(),
                        psm.data.body_link_quat_w[0][-1].cpu().numpy()
                    )
                    psm_name = f"PSM{i+1}"
                    saved_poses[psm_name]["tip_pos_w"] = saved_tips[i][0]
                    saved_poses[psm_name]["tip_quat_w"] = saved_tips[i][1]
                
                mtm_synced = True
                print("[READY] MTMs aligned. Press and release clutch to start teleoperation.")
                continue
            
            # MTM quaternion processing
            mtm_q1 = R.from_rotvec(mtml_rot).as_quat()
            mtm_q1 = np.concatenate([[mtm_q1[3]], mtm_q1[:3]])
            mtm_q2 = R.from_rotvec(mtmr_rot).as_quat()
            mtm_q2 = np.concatenate([[mtm_q2[3]], mtm_q2[:3]])
            
            # MTM clutch handling
            if not mtm_clutch:
                if was_in_mtm_clutch:
                    init_mtml_pos = mtml_pos
                    init_mtmr_pos = mtmr_pos
                    def init_tip(robot):
                        return (cam_T_world @ pose_to_transformation_matrix(
                            robot.data.body_link_pos_w[0][-1].cpu().numpy(),
                            robot.data.body_link_quat_w[0][-1].cpu().numpy()
                        ))[:3, 3]
                    init_psm1_tip_pos = init_tip(psm1)
                    init_psm2_tip_pos = init_tip(psm2)
                
                p1_target = init_psm1_tip_pos + (mtml_pos - init_mtml_pos) * args_cli.scale
                p2_target = init_psm2_tip_pos + (mtmr_pos - init_mtmr_pos) * args_cli.scale
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
                cam_T_psm1 = cur_tip(psm1)
                cam_T_psm2 = cur_tip(psm2)
            
            # Phantom Omni handling
            stylus_pose, g3, po_clutch = po_teleop.advance()
            if po_clutch is None:
                time.sleep(0.05)
                continue
            
            cam_T_psm3 = cam_T_world @ pose_to_transformation_matrix(
                psm3.data.body_link_pos_w[0][-1].cpu().numpy(),
                psm3.data.body_link_quat_w[0][-1].cpu().numpy()
            )
            
            # Prepare human data for controller
            human_data = {
                'cam_T_psm1': cam_T_psm1,
                'w_T_psm1base': world_T_b[0],
                'cam_T_psm2': cam_T_psm2,
                'w_T_psm2base': world_T_b[1],
                'cam_T_psm3': cam_T_psm3,
                'w_T_psm3base': world_T_b[2],
                'w_T_cam': world_T_cam,
                'env': env,
                'g1': g1,
                'g2': g2,
                'g3': g3
            }
        
        # Handle model trigger
        if os.path.exists(MODEL_TRIGGER_PATH) and not model_triggered:
            print(f"[MODEL] Trigger detected. Starting ACT control.")
            model_triggered = True
            printed_trigger = True
            try:
                os.remove(MODEL_TRIGGER_PATH)
            except Exception as e:
                print(f"[MODEL] Failed to remove trigger file: {e}")
        
        # Get model output
        model_output = None
        if needs_model_control and model_triggered and controller:
            try:
                # Prepare model input
                model_input_parts = []
                for psm in [psm1, psm2, psm3]:
                    psm_q = psm.data.joint_pos[0].cpu().numpy()
                    jaw_angle = abs(psm_q[-2]) + abs(psm_q[-1])
                    # Ensure jaw_angle is properly shaped
                    joint_part = psm_q[:-2].flatten()  # 6 joints
                    jaw_part = np.array([jaw_angle])   # 1 jaw angle as array
                    model_input_parts.append(joint_part)
                    model_input_parts.append(jaw_part)
                
                model_input = np.concatenate(model_input_parts)
                
                imgs = np.stack([
                    camera_l.data.output["rgb"][0].cpu().numpy(),
                    camera_r.data.output["rgb"][0].cpu().numpy()
                ]) / 255.0
                
                # Model inference
                model_output = controller.step(imgs, model_input)
                last_model_output = model_output
                
            except RuntimeError as e:
                if "Rollout was already completed" in str(e):
                    if printed_trigger:
                        print("[MODEL] ACT rollout completed. Freezing at last pose.")
                        printed_trigger = False
                    model_triggered = False
                    model_output = last_model_output
                else:
                    raise
        
        # Generate and execute actions
        actions = mixed_controller.get_actions(
            human_data=human_data,
            model_output=model_output,
            saved_poses=saved_poses
        )
        
        env.step(actions)
        
        # Handle reset trigger
        if os.path.exists(RESET_TRIGGER_PATH):
            print("[RESET] Detected reset trigger. Resetting environment...")
            
            # Reset cubes with randomization - Fix cube naming
            cube_configs = [
                ([np.random.uniform(-0.01, 0.01), np.random.uniform(-0.01, 0.01), 0.0], "teleop_logs/cube_latest_1", "cube_rigid_1"),
                ([np.random.uniform(0.045, 0.055), np.random.uniform(0.0, 0.03), 0.0], "teleop_logs/cube_latest_2", "cube_rigid_2"),
                ([np.random.uniform(-0.055, -0.045), np.random.uniform(0.0, 0.03), 0.0], "teleop_logs/cube_latest_3", "cube_rigid_3"),
                ([0.0, 0.055, 0.0], "teleop_logs/cube_latest_4", "cube_rigid_4")
            ]
            
            for i, (cube_pos, log_path, cube_key) in enumerate(cube_configs):
                cube_yaw = np.random.uniform(-np.pi/2, np.pi/2) if i < 3 else 0
                cube_quat = R.from_euler("z", cube_yaw).as_quat()
                cube_ori = [cube_quat[3], cube_quat[0], cube_quat[1], cube_quat[2]]
                reset_cube_pose(env, log_path, cube_pos, cube_ori, cube_key=cube_key)
            
            # Reset PSMs
            reset_psms_to_initial_pose(env, saved_tips, world_T_b, control_config, num_steps=30)
            
            # Reset controllers
            if controller:
                controller.reset()
            
            if needs_human_control:
                mtm_manipulator.home()
                time.sleep(2.0)
                # Re-align MTM
                mtm_manipulator.adjust_orientation(
                    mtm_interface.simpose2hrsvpose(cam_T_world @ pose_to_transformation_matrix(*saved_tips[0])),
                    mtm_interface.simpose2hrsvpose(cam_T_world @ pose_to_transformation_matrix(*saved_tips[1]))
                )
            
            # Reset state
            was_in_mtm_clutch = True
            was_in_po_clutch = True
            po_waiting_for_clutch = True
            model_triggered = False
            printed_trigger = False
            
            os.remove(RESET_TRIGGER_PATH)
            print("[RESET] Reset complete. System ready.")
            continue
        
        # Handle logging
        teleop_logger.check_and_start_logging(env)
        if teleop_logger.enable_logging and teleop_logger.frame_num == 0:
            log_current_pose(env, teleop_logger.log_dir)
        teleop_logger.stop_logging()
        
        if teleop_logger.enable_logging:
            frame_num = teleop_logger.frame_num + 1
            timestamp = time.time()
            robot_states = {}
            
            for psm_key, robot_name in psm_name_dict.items():
                robot = env.unwrapped.scene[robot_name]
                joint_positions = robot.data.joint_pos[0][:6].cpu().numpy()
                jaw_angle = abs(robot.data.joint_pos[0][-2].cpu().numpy()) + abs(robot.data.joint_pos[0][-1].cpu().numpy())
                ee_position = robot.data.body_link_pos_w[0][-1].cpu().numpy()
                ee_quat = robot.data.body_link_quat_w[0][-1].cpu().numpy()
                orientation_matrix = R.from_quat(np.concatenate([ee_quat[1:], [ee_quat[0]]])).as_matrix()
                
                robot_states[psm_key] = {
                    "joint_positions": joint_positions,
                    "jaw_angle": jaw_angle,
                    "ee_position": ee_position,
                    "orientation_matrix": orientation_matrix
                }
            
            cam_l_img = camera_l.data.output["rgb"][0].cpu().numpy()
            cam_r_img = camera_r.data.output["rgb"][0].cpu().numpy()
            
            teleop_logger.enqueue(frame_num, timestamp, robot_states, cam_l_img, cam_r_img)
            teleop_logger.frame_num = frame_num
        
        # Control loop timing
        time.sleep(max(0.0, 1/30.0 - (time.time() - start_time)))
    
    # Cleanup
    try:
        if os.path.exists(args_cli.log_trigger_file):
            os.remove(args_cli.log_trigger_file)
        if os.path.exists(MODEL_TRIGGER_PATH):
            os.remove(MODEL_TRIGGER_PATH)
    except:
        pass
    
    print("[INFO] Shutting down...")
    teleop_logger.shutdown()
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()