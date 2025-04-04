import csv
import os
import numpy as np
from datetime import datetime

class CSVLogger:
    def __init__(self, log_file_path, psm_name_dict):
        """
        Initialize the logger with the file path and robot name mapping.
        """
        self.log_file_path = log_file_path
        self.psm_name_dict = psm_name_dict

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

    def log(self, frame_num, sim_time, robot_states, camera_left_path, camera_right_path):
        """
        Log a single row of data to the CSV file.

        :param frame_num: Current frame number
        :param sim_time: Simulation time
        :param robot_states: Dictionary of robot states (joint positions, end-effector positions, orientation matrix)
        :param camera_left_path: Path to the left camera image
        :param camera_right_path: Path to the right camera image
        """
        epoch_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        row = [epoch_time, sim_time, frame_num]
        for psm, state in robot_states.items():
            joint_positions = state["joint_positions"]
            jaw_angle = state["jaw_angle"]
            ee_position = state["ee_position"]
            orientation_matrix = state["orientation_matrix"]

            row += list(joint_positions)  # Joint positions
            row += [jaw_angle]  # Jaw angle
            row += list(ee_position)  # End-effector positions
            row += list(orientation_matrix.flatten())  # Orientation matrix

        row += [camera_left_path, camera_right_path]  # Camera image paths

        # Append the row to the CSV file
        with open(self.log_file_path, mode="a", newline="") as log_file:
            csv_writer = csv.writer(log_file)
            csv_writer.writerow(row)