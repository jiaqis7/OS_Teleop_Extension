#!/usr/bin/env python3
import rospy
import numpy as np
import tf
from omni_msgs.msg import OmniButtonEvent
from omni.isaac.lab.devices import DeviceBase
from scipy.spatial.transform.rotation import Rotation


class PhantomOmniTeleop(DeviceBase):
    def __init__(self):
        super().__init__()

        self.eye_theta = 30 * np.pi / 180
        self.eye_T_po = np.array(
            [
                [0, -np.cos(self.eye_theta), -np.sin(self.eye_theta), 0],
                [1, 0, 0, 0],
                [0, -np.sin(self.eye_theta), np.cos(self.eye_theta), 0],
                [0, 0, 0, 1],
            ]
        )
        self.pose_to_psm_frame_orientation = np.array([[-1, 0, 0, 0], [0, 0, -1, 0], [0, -1, 0, 0], [0, 0, 0, 1]])


   
            
        self.listener = tf.TransformListener()
        rospy.Subscriber("phantom/button", OmniButtonEvent, self.button_callback)

        # State variables
        self.gripper_command = False
        self.clutch = True

        # Set the rate at which to check for the transform
        self.rate = rospy.Rate(50.0)  # 50 Hz

    def button_callback(self, msg):
        # As long as the grey button is pressed continuously, self.clutch will be true
        # Clutch (grey button)
        self.clutch = (msg.grey_button == 1)

        # Gripper (white button)
        self.gripper_command = True if msg.white_button == 1 else None

    def transform_to_matrix(self, trans, rot):
        """Convert a tf translation and quaternion rotation to a 4x4 transformation matrix."""
        matrix = tf.transformations.quaternion_matrix(rot)
        matrix[0:3, 3] = trans
        return matrix

    def get_stylus_pose(self):
        """Get pose of the stylus in the OS base frame, and return the pose changed to PSM end link frame orientation."""
        try:
            (trans, rot) = self.listener.lookupTransform("base", "stylus", rospy.Time(0))
            return self.transform_to_matrix(trans, rot) @ self.pose_to_psm_frame_orientation
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            return None

    def advance(self):
        """Retrieve the latest teleoperation command."""
        current_pose = self.get_stylus_pose()
        if current_pose is None:
            return None, None, None

        # Get the pose of the stylus in the eye frame
        eye_T_stylus = self.eye_T_po @ current_pose

        # Extract translation (position delta)
        target_pos = eye_T_stylus[:3, 3]

        # Extract rotation (Rotation vector)
        target_rot = Rotation.from_matrix(eye_T_stylus[:3, :3]).as_rotvec()
        target_rot = np.array(target_rot)

        return (np.concatenate([target_pos, target_rot]), self.gripper_command, self.clutch)

    def reset(self):
        """Reset the teleoperation state."""
        self.clutch = True
        self.gripper_open = False

    def add_callback(self, key, func):
        """
        Adds a callback function triggered by a specific key input.
        """
        pass
