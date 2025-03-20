#!/usr/bin/env python3

import rospy
import numpy as np
import tf
from omni_msgs.msg import OmniButtonEvent


class PoseTransformer:
    def __init__(self):
        # Initialize the ROS node
        rospy.init_node("omni_pose_transform", anonymous=True)

        # Create a tf listener
        self.listener = tf.TransformListener()
        rospy.Subscriber("phantom/button", OmniButtonEvent, self.button_callback)

        # State variables
        self.was_in_clutch = True
        self.clutch = True
        self.enabled = False
        self.open_jaw = False
        self.was_in_open_jaw = False

        self.po_T_stylus_prev = np.eye(4)

        # Set the rate at which to check for the transform
        self.rate = rospy.Rate(10.0)  # 10 Hz

    def button_callback(self, msg):
        # As long as the grey button is pressed continuously, self.clutch will be true
        if msg.grey_button == 1:

            if self.was_in_clutch == False:
                rospy.loginfo("Clutch Pressed")
            self.clutch = True

        # When the grey button is released, a message is passed msg.grey_button=0
        # self.clutch will be set to false
        elif msg.grey_button == 0:
            self.clutch = False

        if msg.white_button == 1:
            self.open_jaw = True
            rospy.loginfo("Opening Jaw")

        elif msg.white_button == 0:
            self.open_jaw = False
            rospy.loginfo("Closing Jaw")

    def transform_to_matrix(self, trans, rot):
        # Create the homogeneous transformation matrix from translation and rotation
        matrix = tf.transformations.quaternion_matrix(rot)
        matrix[0:3, 3] = trans
        return matrix

    def run(self):
        while not rospy.is_shutdown():
            try:
                # Look up the transform from base to stylus
                (trans, rot) = self.listener.lookupTransform("base", "stylus", rospy.Time(0))

                # Convert the transform to a 4x4 transformation matrix po_T_stylus
                po_T_stylus = self.transform_to_matrix(trans, rot)
                # print(po_T_stylus)

                # Calculate the relative transformation matrix from the previous pose
                rel_T = np.dot(np.linalg.inv(self.po_T_stylus_prev), po_T_stylus)
                print(rel_T)

                # Update the previous pose
                self.po_T_stylus_prev = po_T_stylus

            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                continue

            self.rate.sleep()


if __name__ == "__main__":
    try:
        transformer = PoseTransformer()
        # Run the transformer
        transformer.run()
    except rospy.ROSInterruptException:
        pass
