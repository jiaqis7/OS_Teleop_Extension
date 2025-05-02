import csv
import os
import numpy as np
from datetime import datetime
import json
import shutil

# def get_log_dir(base_dir, demo_name, replace_flag_path="replace_flag.txt", counter_file="demo_counter.json"):
#     os.makedirs(base_dir, exist_ok=True)
#     counter_path = os.path.join(base_dir, counter_file)

#     # Load or initialize demo counters
#     if os.path.exists(counter_path):
#         with open(counter_path, "r") as f:
#             demo_counters = json.load(f)
#     else:
#         demo_counters = {}

#     replace = os.path.exists(replace_flag_path)

#     # Default to 1 if not found
#     current_idx = demo_counters.get(demo_name, 1)

#     if replace:
#         log_folder = f"{demo_name}_1"
#         full_path = os.path.join(base_dir, log_folder)
#         if os.path.exists(full_path):
#             shutil.rmtree(full_path)
#         os.remove(replace_flag_path)

#         # Reset counter to 1
#         demo_counters[demo_name] = 1
#     else:
#         log_folder = f"{demo_name}_{current_idx}"
#         full_path = os.path.join(base_dir, log_folder)

#         # Only increment counter if the folder didn't exist yet
#         while os.path.exists(full_path):
#             current_idx += 1
#             log_folder = f"{demo_name}_{current_idx}"
#             full_path = os.path.join(base_dir, log_folder)

#         demo_counters[demo_name] = current_idx

#     # Save updated counter
#     with open(counter_path, "w") as f:
#         json.dump(demo_counters, f, indent=4)

#     return full_path


class CSVLogger:
    def __init__(self, log_file_path, psm_name_dict, start_time=None):
        """
        Initialize the logger with the file path and robot name mapping.
        """
        self.log_file_path = log_file_path
        self.psm_name_dict = psm_name_dict
        self.start_time = start_time  # Track first time


        # Create the CSV file and write the header
        with open(self.log_file_path, mode="w", newline="") as log_file:
            csv_writer = csv.writer(log_file)
            # Write header
            header = ["Epoch Time (Seconds)", "Time (Seconds)", "Frame Number"]
            for psm, robot in self.psm_name_dict.items():
                header += [f"{psm}_joint_{i}" for i in range(1, 7)]  # Joint positions
                header += [f"{psm}_jaw_angle"]
                header += [f"{psm}_ee_x", f"{psm}_ee_y", f"{psm}_ee_z"]  # End-effector positions
                header += [
                    f"{psm}_Orientation_Matrix_[{row},{col}]"
                    for row in range(1, 4)
                    for col in range(1, 4)
                ]  # Orientation matrix
            header += ["Camera Left", "Camera Right"]  # Camera image paths
            csv_writer.writerow(header)

    def log(self, frame_num, real_time, robot_states, camera_left_path, camera_right_path):
        """
        Log a single row of data to the CSV file.

        :param frame_num: Current frame number
        :param sim_time: Simulation time
        :param robot_states: Dictionary of robot states (joint positions, end-effector positions, orientation matrix)
        :param camera_left_path: Path to the left camera image
        :param camera_right_path: Path to the right camera image
        """
        if self.start_time is None:
            self.start_time = real_time  # Set start time on first call

        relative_time = real_time - self.start_time
        epoch_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')


        row = [epoch_time, relative_time, frame_num]

        for psm, state in robot_states.items():
            joint_positions = state["joint_positions"]
            jaw_angle = state["jaw_angle"]
            ee_position = state["ee_position"]
            orientation_matrix = state["orientation_matrix"]

            row += list(joint_positions)  # Joint positions
            row += [jaw_angle]  # Jaw angle
            row += list(ee_position)  # End-effector positions
            row += list(orientation_matrix.flatten())  # Orientation matrix

        row += [
            os.path.relpath(camera_left_path, os.path.dirname(self.log_file_path)),
            os.path.relpath(camera_right_path, os.path.dirname(self.log_file_path))
        ]

        # Append the row to the CSV file
        with open(self.log_file_path, mode="a", newline="") as log_file:
            csv_writer = csv.writer(log_file)
            csv_writer.writerow(row)