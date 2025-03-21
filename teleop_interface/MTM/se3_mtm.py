#!/usr/bin/env python3
import rospy
import numpy as np
from omni_msgs.msg import OmniButtonEvent
from omni.isaac.lab.devices import DeviceBase
from scipy.spatial.transform.rotation import Rotation
from sensor_msgs.msg import JointState, Joy
from geometry_msgs.msg import PoseStamped
import os
os.environ['ROS_MASTER_URI'] = 'http://172.24.95.120:11311'
os.environ['ROS_IP'] = '10.34.166.95'

def transformation_matrix_to_pose(transformation_matrix):
    """
    Convert a 4x4 transformation matrix to position and quaternion.

    Args:
        transformation_matrix (np.ndarray): 4x4 transformation matrix.

    Returns:
        tuple: Position array (x, y, z) and quaternion array (w, x, y, z).
    """
    # Extract the translation part
    position = transformation_matrix[:3, 3]

    # Extract the rotation part and convert to quaternion
    rotation_matrix = transformation_matrix[:3, :3]
    quaternion = Rotation.from_matrix(rotation_matrix).as_quat()

    # Reorder quaternion to (w, x, y, z)
    quaternion = np.array([quaternion[3], quaternion[0], quaternion[1], quaternion[2]])

    return position, quaternion


def pose_to_transformation_matrix(position, quaternion):
    """
    Convert position and quaternion to a 4x4 transformation matrix.

    Args:
        position (np.ndarray): Position array (x, y, z).
        quaternion (np.ndarray): Quaternion array (w, x, y, z).

    Returns:
        np.ndarray: 4x4 transformation matrix.
    """
    # Create a 4x4 identity matrix
    transformation_matrix = np.eye(4)

    # Set the translation part
    transformation_matrix[:3, 3] = position

    # Convert quaternion to rotation matrix
    rotation_matrix = Rotation.from_quat([quaternion[1], quaternion[2], quaternion[3], quaternion[0]]).as_matrix()

    # Set the rotation part
    transformation_matrix[:3, :3] = rotation_matrix

    return transformation_matrix

class MTMTeleop(DeviceBase):
    def __init__(self, pos_sensitivity=1.0, rot_sensitivity=1.0):
        super().__init__()

        # Sensitivity scaling
        self.pos_sensitivity = pos_sensitivity
        self.rot_sensitivity = rot_sensitivity

        # ROS node initialization
        rospy.init_node("phantom_omni_teleop", anonymous=True)

        rospy.Subscriber("/MTML/measured_cp", PoseStamped, self.mtml_callback)
        rospy.Subscriber("/MTMR/measured_cp", PoseStamped, self.mtmr_callback)
        rospy.Subscriber("/MTML/gripper/measured_js", JointState, self.mtml_gripper_callback)
        rospy.Subscriber("/MTMR/gripper/measured_js", JointState, self.mtmr_gripper_callback)
        rospy.Subscriber("/footpedals/clutch", Joy, self.clutch_callback)

        # State variables
        self.enabled = False
        self.clutch = True
        self.mtml_pose = None
        self.mtmr_pose = None
        self.l_jaw_angle = None
        self.r_jaw_angle = None

        # Naive Assumption on watching angle
        self.eye_theta = 45 * np.pi / 180
        self.eye_T_hrsv = np.linalg.inv(np.array([[0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 0, 1]]) @ 
        np.array(
            [
                [np.cos(self.eye_theta), 0, np.sin(self.eye_theta), 0],
                [0, 1, 0, 0],
                [-np.sin(self.eye_theta), 0, np.cos(self.eye_theta), 0],
                [0, 0, 0, 1],
            ]
        ))

        # Transformation matrices to align orientation with the simulation
        self.mtm_T_mtm_sim = np.array([[0, 0, 1, 0], 
                                       [-1, 0, 0, 0],
                                       [0, -1, 0, 0],
                                       [0, 0, 0, 1]])

        # Set the rate at which to check for the transform
        self.rate = rospy.Rate(10.0)  # 10 Hz

    def mtml_callback(self, msg):
        self.mtml_pose = msg.pose

    def mtmr_callback(self, msg):
        self.mtmr_pose = msg.pose
    
    def mtml_gripper_callback(self, msg):
        self.l_jaw_angle = msg.position[0]

    def mtmr_gripper_callback(self, msg):
        self.r_jaw_angle = msg.position[0]

    def clutch_callback(self, msg):
        if msg.buttons[0] == 1:
            if not self.enabled:
                self.enabled = True
            self.clutch = True
            print("Clutch Pressed")
            return

        elif msg.buttons[0] == 0:
            if not self.enabled:
                self.clutch = True
                print("Ignoring the clutch releasing output before the first clutch press")
                return
            self.clutch = False
            print("Clutch Released")

    def advance(self):
        """Retrieve the latest teleoperation command."""
        if not self.mtml_pose or not self.mtmr_pose or not self.l_jaw_angle or not self.r_jaw_angle:
            print("Waiting for subscription... Cannot start teleoperation yet")
            return None, None, None, None, None, None, None
        # Extract rotation (Rotation vector)
        hrsv_p_mtml = self.mtml_pose.position
        hrsv_q_mtml = self.mtml_pose.orientation
        hrsv_T_mtml = pose_to_transformation_matrix(np.array([hrsv_p_mtml.x, hrsv_p_mtml.y, hrsv_p_mtml.z]), 
                                                    np.array([hrsv_q_mtml.w, hrsv_q_mtml.x, hrsv_q_mtml.y, hrsv_q_mtml.z]))
        eye_T_mtml_sim = self.eye_T_hrsv @ hrsv_T_mtml @ self.mtm_T_mtm_sim
        eye_p_mtml_sim, eye_q_mtml_sim = transformation_matrix_to_pose(eye_T_mtml_sim)
        target_mtml_rot = Rotation.from_quat(np.concatenate([[eye_q_mtml_sim[3]], eye_q_mtml_sim[:3]])).as_rotvec()
        target_mtml_rot = np.array(target_mtml_rot) * self.rot_sensitivity

        hrsv_p_mtmr = self.mtmr_pose.position
        hrsv_q_mtmr = self.mtmr_pose.orientation
        hrsv_T_mtmr = pose_to_transformation_matrix(np.array([hrsv_p_mtmr.x, hrsv_p_mtmr.y, hrsv_p_mtmr.z]), 
                                                    np.array([hrsv_q_mtmr.w, hrsv_q_mtmr.x, hrsv_q_mtmr.y, hrsv_q_mtmr.z]))
        eye_T_mtmr_sim = self.eye_T_hrsv @ hrsv_T_mtmr @ self.mtm_T_mtm_sim
        eye_p_mtmr_sim, eye_q_mtmr_sim = transformation_matrix_to_pose(eye_T_mtmr_sim)
        target_mtmr_rot = Rotation.from_quat(np.concatenate([[eye_q_mtmr_sim[3]], eye_q_mtmr_sim[:3]])).as_rotvec()
        target_mtmr_rot = np.array(target_mtmr_rot) * self.rot_sensitivity

        return eye_p_mtml_sim * self.pos_sensitivity, target_mtml_rot, self.l_jaw_angle, eye_p_mtmr_sim * self.pos_sensitivity, target_mtmr_rot, self.r_jaw_angle, self.clutch

    def reset(self):
        """Reset the teleoperation state."""
        self.clutch = True
        self.gripper_open = False

    def add_callback(self, key, func):
        """
        Adds a callback function triggered by a specific key input.
        """
        pass
