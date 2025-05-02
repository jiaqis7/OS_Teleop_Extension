import os
import time
import threading
import json
from datetime import datetime
import cv2
import numpy as np
import torch
from pathlib import Path
from logger_utils import CSVLogger
from scipy.spatial.transform import Rotation as R


def reset_cube_pose(env, log_dir, position, orientation, cube_key="cube_rigid"):
    cube = env.scene[cube_key]
    device = cube.data.root_state_w.device

    root_state = cube.data.root_state_w.clone()
    root_state[:, 0:3] = torch.tensor(position, dtype=torch.float32, device=device)
    root_state[:, 3:7] = torch.tensor(orientation, dtype=torch.float32, device=device)
    root_state[:, 7:13] = 0.0  # zero velocities
    cube.write_root_state_to_sim(root_state)

    # Save latest pose (for debugging or reference)
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    cube_pose = {"position": list(position), "orientation": list(orientation)}
    with open(os.path.join(log_dir, "cube_pose.json"), "w") as f:
        json.dump(cube_pose, f, indent=4)

    print(f"[RESET] Cube pose reset and saved to {log_dir}/cube_pose.json")


def log_current_pose(env, log_dir, cube_key="cube_rigid"):
    cube = env.scene[cube_key]
    pos = cube.data.root_pos_w[0].cpu().numpy().tolist()
    quat = cube.data.root_quat_w[0].cpu().numpy().tolist()

    # Also log world-frame tip pose of both PSMs
    psm1 = env.scene["robot_1"]
    psm2 = env.scene["robot_2"]

    psm1_tip_pos = psm1.data.body_link_pos_w[0][-1].cpu().numpy().tolist()
    psm1_tip_quat = psm1.data.body_link_quat_w[0][-1].cpu().numpy().tolist()

    psm2_tip_pos = psm2.data.body_link_pos_w[0][-1].cpu().numpy().tolist()
    psm2_tip_quat = psm2.data.body_link_quat_w[0][-1].cpu().numpy().tolist()

    log_data = {
        "cube": {
            "position": pos,
            "orientation": quat
        },
        "robot_1_tip": {
            "position": psm1_tip_pos,
            "orientation": psm1_tip_quat
        },
        "robot_2_tip": {
            "position": psm2_tip_pos,
            "orientation": psm2_tip_quat
        }
    }

    Path(log_dir).mkdir(parents=True, exist_ok=True)
    json_path = os.path.join(log_dir, "pose.json")
    with open(json_path, "w") as f:
        json.dump(log_data, f, indent=4)

    print(f"[LOG] Logged current cube and robot tip poses to {json_path}")



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
    def __init__(self, trigger_file, psm_name_dict, log_duration=30.0):
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

    def check_and_start_logging(self, env):
        if not self.enable_logging and os.path.exists(self.trigger_file):
            print("[LOG] Trigger file detected. Starting logging.")
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.log_dir = os.path.join(os.getcwd(), f"teleop_logs_{timestamp}")
            os.makedirs(os.path.join(self.log_dir, "left_images"), exist_ok=True)
            os.makedirs(os.path.join(self.log_dir, "right_images"), exist_ok=True)

            self.start_time = time.time()
            self.logger = CSVLogger(os.path.join(self.log_dir, "teleop_log.csv"), self.psm_name_dict, start_time=self.start_time)
            self.enable_logging = True
            self.frame_num = 0

            log_current_pose(env, self.log_dir)

            os.remove(self.trigger_file)

    def stop_logging(self):
        if self.enable_logging and (time.time() - self.start_time) > self.log_duration:
            print("[LOG] Logging duration ended. Stopping.")
            self.enable_logging = False
            self.logger = None

    def enqueue(self, frame_num, timestamp, robot_states, cam_l_img, cam_r_img):
        self.buffer.enqueue((frame_num, timestamp, robot_states, cam_l_img, cam_r_img))

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
