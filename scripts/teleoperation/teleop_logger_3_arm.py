import os
import time
import threading
import json
from datetime import datetime
import cv2
import numpy as np
import torch
from pathlib import Path
from scripts.teleoperation.logger_utils import CSVLogger

from scipy.spatial.transform import Rotation as R
import shutil
import re
import glob
from scripts.teleoperation.contact_detector_simple import ContactDetectorSimple


def reset_cube_pose(env, log_dir, position, orientation, cube_key="cube_rigid_1", initial_state=None):
    cube = env.scene[cube_key]
    device = cube.data.root_state_w.device

    # Use initial clean state if provided, otherwise clone current (less ideal)
    if initial_state is not None:
        root_state = initial_state.clone()
    else:
        root_state = cube.data.root_state_w.clone()
    
    # Set the desired position and orientation
    root_state[:, 0:3] = torch.tensor(position, dtype=torch.float32, device=device)
    root_state[:, 3:7] = torch.tensor(orientation, dtype=torch.float32, device=device)
    root_state[:, 7:13] = 0.0  # zero velocities
    
    cube.write_root_state_to_sim(root_state)
    
    print(f"[RESET] Cube '{cube_key}' pose applied: pos={position[:3]}, ori={orientation[:4]}")



def log_current_pose(env, log_dir, cube_keys=["cube_rigid_1", "cube_rigid_2", "cube_rigid_3", "cube_rigid_4"]):
    log_data = {
        "meta": {
            "arm_names": ["PSM1", "PSM2", "PSM3"],
            "teleop1_connection": "MTML-PSM1",
            "teleop2_connection": "MTMR-PSM2",
            "teleop3_connection": "PO-PSM3",
            "surgeon_name": "Alaa",
            "assistant_name": "Jiaqi",
            "mtm_scale": 0.4,
            "duration": 30
        }
    }

    # Get scene object properly
    scene = env.unwrapped.scene if hasattr(env, 'unwrapped') else env.scene

    # Log cube poses separately
    for cube_key in cube_keys:
        cube = scene[cube_key]
        pos = cube.data.root_pos_w[0].cpu().numpy().tolist()
        quat = cube.data.root_quat_w[0].cpu().numpy().tolist()
        log_data[cube_key] = {
            "position": pos,
            "orientation": quat
        }

    # Log robot tip poses
    psm1 = scene["robot_1"]
    psm2 = scene["robot_2"]
    psm3 = scene["robot_3"]

    log_data["robot_1_tip"] = {
        "position": psm1.data.body_link_pos_w[0][-1].cpu().numpy().tolist(),
        "orientation": psm1.data.body_link_quat_w[0][-1].cpu().numpy().tolist()
    }
    log_data["robot_2_tip"] = {
        "position": psm2.data.body_link_pos_w[0][-1].cpu().numpy().tolist(),
        "orientation": psm2.data.body_link_quat_w[0][-1].cpu().numpy().tolist()
    }
    log_data["robot_3_tip"] = {
        "position": psm3.data.body_link_pos_w[0][-1].cpu().numpy().tolist(),
        "orientation": psm3.data.body_link_quat_w[0][-1].cpu().numpy().tolist()
    }

    os.makedirs(log_dir, exist_ok=True)
    json_path = os.path.join(log_dir, "pose.json")
    with open(json_path, "w") as f:
        json.dump(log_data, f, indent=4)

    print(f"[LOG] Logged cube and robot tip poses to {json_path}")




def reset_cube_pose_from_json(env, json_path):
    """
    Reset cube_rigid_1 to cube_rigid_4 in the simulation to the poses stored in a JSON file.
    Now works with JSON files where cube poses are stored at the top level.

    :param env: Isaac Lab environment.
    :param json_path: Path to the JSON file with cube poses stored at top level.
    """
    if not os.path.exists(json_path):
        print(f"[WARNING] pose.json not found at {json_path}. Skipping reset.")
        return

    with open(json_path, "r") as f:
        pose_data = json.load(f)

    # Get the scene object using the recommended approach
    scene = env.unwrapped.scene if hasattr(env, 'unwrapped') else env.scene

    # Get all available entities in the scene
    scene_entities = list(scene.keys())  # This gets all available entity keys

    for cube_key in ["cube_rigid_1", "cube_rigid_2", "cube_rigid_3", "cube_rigid_4"]:
        # Check if cube exists in scene by looking in the list of entities
        if cube_key not in scene_entities:
            print(f"[WARNING] Cube '{cube_key}' not found in scene. Available entities: {scene_entities}. Skipping.")
            continue
        if cube_key not in pose_data:
            print(f"[WARNING] Cube '{cube_key}' not found in JSON. Skipping.")
            continue

        cube_pose = pose_data[cube_key]
        position = cube_pose["position"]
        orientation = cube_pose["orientation"]

        cube = scene[cube_key]
        device = cube.data.root_state_w.device
        root_state = cube.data.root_state_w.clone()

        root_state[:, 0:3] = torch.tensor(position, dtype=torch.float32, device=device)
        root_state[:, 3:7] = torch.tensor(orientation, dtype=torch.float32, device=device)
        root_state[:, 7:13] = 0.0  # zero linear and angular velocities

        cube.write_root_state_to_sim(root_state)
        print(f"[RESET] Cube '{cube_key}' reset to pose from JSON.")


class RingBuffer:
    def __init__(self, size=256):
        self.buffer = [None] * size
        self.size = size
        self.write_index = 0
        self.read_index = 0
        self.lock = threading.Lock()

    def enqueue(self, item):
        with self.lock:
            next_index = (self.write_index + 1) % self.size
            if next_index == self.read_index:
                self.read_index = (self.read_index + 1) % self.size
            self.buffer[self.write_index] = item
            self.write_index = next_index

    def dequeue(self):
        with self.lock:
            if self.read_index == self.write_index:
                return None
            item = self.buffer[self.read_index]
            self.buffer[self.read_index] = None
            self.read_index = (self.read_index + 1) % self.size
            return item


class TeleopLogger:
    def __init__(self, trigger_file, psm_name_dict, log_duration=60.0):
        self.trigger_file = trigger_file
        self.psm_name_dict = psm_name_dict
        self.log_duration = log_duration

        self.enable_logging = False
        self.frame_num = 0
        self.start_time = None
        self.log_dir = None

        self.buffer = RingBuffer()
        self.logger_thread = threading.Thread(target=self._logger_loop, daemon=True)
        self.logger_thread.start()
        
        # Initialize contact detector
        self.contact_detector = ContactDetectorSimple()


    def check_and_start_logging(self, env):
        if self.enable_logging:
            return

        # Look for any file like log_trigger_demo_X.txt
        trigger_files = glob.glob("log_trigger_*.txt")
        if not trigger_files:
            return

        trigger_file = trigger_files[0]
        match = re.match(r"log_trigger_(.+)\.txt", os.path.basename(trigger_file))
        if not match:
            print(f"[LOG] Invalid trigger filename format: {trigger_file}")
            return

        folder_name = match.group(1)  # e.g., 'demo_1'
        self.log_dir = os.path.join(os.getcwd(), folder_name)

        # Replace folder if it already exists
        if os.path.exists(self.log_dir):
            print(f"[LOG] Replacing existing folder: {folder_name}")
            shutil.rmtree(self.log_dir)

        print(f"[LOG] Logging into folder: {self.log_dir}")
        os.makedirs(os.path.join(self.log_dir, "left_images"), exist_ok=True)
        os.makedirs(os.path.join(self.log_dir, "right_images"), exist_ok=True)

        self.start_time = time.time()
        self.logger = CSVLogger(os.path.join(self.log_dir, "teleop_log.csv"), self.psm_name_dict, start_time=self.start_time)
        self.enable_logging = True
        self.frame_num = 0

        log_current_pose(env, self.log_dir)
        
        # Reset contact detector for new logging session
        self.contact_detector.reset()

        os.remove(trigger_file)


    def stop_logging(self, env=None):
        if self.enable_logging and (time.time() - self.start_time) > self.log_duration:
            print("[LOG] Logging duration ended. Stopping.")
            
            # Update contacts one final time before saving
            if env and self.log_dir:
                try:
                    scene = env.unwrapped.scene if hasattr(env, 'unwrapped') else env.scene
                    self.contact_detector.update_contacts(scene)
                except Exception as e:
                    print(f"[LOG] Warning: Could not update final contacts: {e}")
            
            # Save final contact state
            if self.log_dir:
                self.contact_detector.save_final_state(self.log_dir)
            
            self.enable_logging = False
            self.logger = None

    def enqueue(self, frame_num, timestamp, robot_states, cam_l_img, cam_r_img):
        self.buffer.enqueue((frame_num, timestamp, robot_states, cam_l_img, cam_r_img))
    
    def check_contacts(self, env):
        """Check contact states and log success if all blocks are in contact."""
        if not self.enable_logging or not self.log_dir:
            return
        
        # Get the scene object
        scene = env.unwrapped.scene if hasattr(env, 'unwrapped') else env.scene
        
        # Update contact states
        contact_info = self.contact_detector.update_contacts(scene)

    def _logger_loop(self):
        while True:
            item = self.buffer.dequeue()
            if item:
                self._log_frame(*item)
            else:
                time.sleep(0.001)

    def _log_frame(self, frame_num, timestamp, robot_states, cam_l_img, cam_r_img):
        if not self.logger:
            return

        cam_l_bgr = cv2.cvtColor(cam_l_img, cv2.COLOR_RGB2BGR)
        cam_r_bgr = cv2.cvtColor(cam_r_img, cv2.COLOR_RGB2BGR)

        left_path = os.path.join(self.log_dir, "left_images", f"camera_left_{frame_num}.png")
        right_path = os.path.join(self.log_dir, "right_images", f"camera_right_{frame_num}.png")

        cv2.imwrite(left_path, cam_l_bgr)
        cv2.imwrite(right_path, cam_r_bgr)

        self.logger.log(frame_num, timestamp, robot_states, left_path, right_path)
