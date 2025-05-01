# import os
# import time
# import cv2
# import numpy as np
# import threading
# import queue
# from datetime import datetime
# from scipy.spatial.transform import Rotation as R
# from logger_utils import CSVLogger

# class TeleopLogger:
#     def __init__(self, log_trigger_file: str, psm_name_dict: dict):
#         self.log_trigger_file = log_trigger_file
#         self.psm_name_dict = psm_name_dict
#         self.enable_logging = False
#         self.logging_start_time = None
#         self.frame_num = 0

#         self.queue = queue.Queue()
#         self.thread = threading.Thread(target=self._logger_thread_fn, daemon=True)
#         self.thread.start()

#     def check_and_start_logging(self):
#         if os.path.exists(self.log_trigger_file) and not self.enable_logging:
#             print("[LOG] Trigger file detected. Logging started.")

#             timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
#             self.run_folder = os.path.join(os.getcwd(), f"teleop_logs_{timestamp}")
#             self.left_image_folder = os.path.join(self.run_folder, "left_images")
#             self.right_image_folder = os.path.join(self.run_folder, "right_images")
#             os.makedirs(self.left_image_folder, exist_ok=True)
#             os.makedirs(self.right_image_folder, exist_ok=True)

#             self.logging_start_time = time.time()
#             self.logger = CSVLogger(os.path.join(self.run_folder, "teleop_log.csv"), self.psm_name_dict, start_time=self.logging_start_time)
#             self.frame_num = 0
#             self.enable_logging = True
#             os.remove(self.log_trigger_file)

#     def stop_logging(self):
#         if self.enable_logging and (time.time() - self.logging_start_time > 30):
#             print("[LOG] 30 seconds of logging elapsed. Stopping logging.")
#             self.enable_logging = False
#             self.logger = None

#     def enqueue(self, frame_num, timestamp, robot_states, cam_l_img, cam_r_img):
#         self.queue.put((frame_num, timestamp, robot_states, cam_l_img, cam_r_img))


#     def _logger_thread_fn(self):
#         while True:
#             item = self.queue.get()
#             if item is None:
#                 break  # graceful shutdown
#             frame_num, timestamp, env, camera_l, camera_r = item
#             self._log_frame(frame_num, timestamp, env, camera_l, camera_r)

#     def _log_frame(self, frame_num, timestamp, robot_states, cam_l_img, cam_r_img):
#         cam_l_input_bgr = cv2.cvtColor(cam_l_img, cv2.COLOR_RGB2BGR)
#         cam_r_input_bgr = cv2.cvtColor(cam_r_img, cv2.COLOR_RGB2BGR)

#         camera_left_path = os.path.join(self.left_image_folder, f"camera_left_{frame_num}.png")
#         camera_right_path = os.path.join(self.right_image_folder, f"camera_right_{frame_num}.png")

#         cv2.imwrite(camera_left_path, cam_l_input_bgr)
#         cv2.imwrite(camera_right_path, cam_r_input_bgr)

#         self.logger.log(frame_num, timestamp, robot_states, camera_left_path, camera_right_path)


#     def shutdown(self):
#         self.queue.put(None)
#         self.thread.join()


import os
import time
import threading
from datetime import datetime
import cv2
import json
import numpy as np
import torch
from logger_utils import CSVLogger

def log_initial_cube_pose(env, log_dir="teleop_logs/initial_cube", cube_key="cube_rigid"):
    cube = env.scene[cube_key]
    pos = cube.data.root_pos_w.cpu().numpy().tolist()
    quat = cube.data.root_quat_w.cpu().numpy().tolist()

    os.makedirs(log_dir, exist_ok=True)
    cube_pose = {"position": pos, "orientation": quat}
    json_path = os.path.join(log_dir, "initial_cube_pose.json")
    with open(json_path, "w") as f:
        json.dump(cube_pose, f, indent=4)

    print(f"[INIT] Logged initial cube pose to {json_path}")
    

def reset_cube_pose_and_log(env, log_dir, position, orientation, cube_key="cube_rigid"):
    cube = env.scene[cube_key]
    device = cube.data.root_state_w.device

    # Get full root state and clone
    root_state = cube.data.root_state_w.clone()

    # Set position and orientation
    root_state[:, 0:3] = torch.tensor(position, dtype=torch.float32, device=device)       # pos
    root_state[:, 3:7] = torch.tensor(orientation, dtype=torch.float32, device=device)    # quat

    # Zero velocities (linear and angular)
    root_state[:, 7:10] = 0.0  # linear vel
    root_state[:, 10:13] = 0.0  # angular vel

    # Write to sim
    cube.write_root_state_to_sim(root_state)

    # Save pose
    os.makedirs(log_dir, exist_ok=True)
    cube_pose = {"position": list(position), "orientation": list(orientation)}
    json_path = os.path.join(log_dir, "cube_pose.json")
    with open(json_path, "w") as f:
        json.dump(cube_pose, f, indent=4)

    print(f"[RESET+LOG] Cube pose set and saved to {json_path}")



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

        self.logger = None
        self.left_folder = None
        self.right_folder = None

    def check_and_start_logging(self, env):
        if not self.enable_logging and os.path.exists(self.trigger_file):
            print("[LOG] Trigger file detected. Starting logging.")
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.log_dir = os.path.join(os.getcwd(), f"teleop_logs_{timestamp}")
            self.left_folder = os.path.join(self.log_dir, "left_images")
            self.right_folder = os.path.join(self.log_dir, "right_images")
            os.makedirs(self.left_folder, exist_ok=True)
            os.makedirs(self.right_folder, exist_ok=True)

            self.start_time = time.time()
            self.logger = CSVLogger(os.path.join(self.log_dir, "teleop_log.csv"), self.psm_name_dict, start_time=self.start_time)
            self.enable_logging = True
            self.frame_num = 0

            os.remove(self.trigger_file)

    def stop_logging(self):
        if self.enable_logging and (time.time() - self.start_time) > self.log_duration:
            print("[LOG] Logging duration ended. Stopping.")
            self.enable_logging = False
            self.logger = None
            self.left_folder = None
            self.right_folder = None
            self.log_dir = None

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
        if self.logger is None:
            return

        cam_l_bgr = cv2.cvtColor(cam_l_img, cv2.COLOR_RGB2BGR)
        cam_r_bgr = cv2.cvtColor(cam_r_img, cv2.COLOR_RGB2BGR)

        left_path = os.path.join(self.left_folder, f"camera_left_{frame_num}.png")
        right_path = os.path.join(self.right_folder, f"camera_right_{frame_num}.png")

        cv2.imwrite(left_path, cam_l_bgr)
        cv2.imwrite(right_path, cam_r_bgr)

        self.logger.log(frame_num, timestamp, robot_states, left_path, right_path)
