#!/usr/bin/env python

# Author: Joonwon Kang
# Date: 2025-03-25
# Node to adjust the impedance, position, and orientation of the MTM for teleoperation

import crtk
import dvrk
import numpy
import sys
import PyKDL
import rospy
from sensor_msgs.msg import Joy
from scipy.spatial.transform import Rotation


class MTMManipulator:
    def __init__(self, expected_interval=0.01):


        print('configuring MTM Manipulator for Teleoperation')
        self.ral = crtk.ral('dvrk_mtm_manipulator')
        self.expected_interval = expected_interval
        self.mtml= dvrk.mtm(ral = self.ral,
                            arm_name = 'MTML',
                            expected_interval = expected_interval)
        self.mtmr= dvrk.mtm(ral = self.ral,
                            arm_name = 'MTMR',
                            expected_interval = expected_interval)
        
        # To lock and unlock the orientation of the MTM
        self.teleop_ready = False
        self.enabled = False
        self.first_clutch_triggered = False  # whether we've seen the first clutch press
        self.reset_done = False
        rospy.Subscriber("/footpedals/clutch", Joy, self.clutch_callback)
    
    def clutch_callback(self, msg):
        if not self.teleop_ready:
            return
         
        if msg.buttons[0] == 1:
            if not self.enabled:
                self.enabled = True
            print("Lock Orientation due to clutch pressed")
            self.mtml.lock_orientation_as_is()
            self.mtmr.lock_orientation_as_is()
            return

        elif msg.buttons[0] == 0:
            if not self.enabled:
                print("Ignoring the clutch releasing output before the first clutch press. Lock Orientation")
                self.mtml.lock_orientation_as_is()
                self.mtmr.lock_orientation_as_is()
                return
            print("Unlock Orientation due to clutch released")
            self.mtml.unlock_orientation()
            self.mtml.body.servo_cf(numpy.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
            self.mtmr.unlock_orientation()
            self.mtmr.body.servo_cf(numpy.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))

            
    def home(self):
        self.ral.check_connections()

        print('starting enable')
        if not self.mtml.enable(10):
            sys.exit('failed to enable MTML within 10 seconds')
        if not self.mtmr.enable(10):
            sys.exit('failed to enable MTMR within 10 seconds')
        print('starting home')
        if not self.mtml.home(10):
            sys.exit('failed to home MTML within 10 seconds')
        if not self.mtmr.home(10):
            sys.exit('failed to home MTMR within 10 seconds')
        print('move to starting position')

        goal = numpy.copy(self.mtml.setpoint_jp())
        goal.fill(0)

        self.mtml.move_jp(goal).wait()
        # try to move again to make sure waiting is working fine, i.e. not blocking
        move_mtml = self.mtml.move_jp(goal)
        move_mtml.wait()

        self.mtmr.move_jp(goal).wait()
        # try to move again to make sure waiting is working fine, i.e. not blocking
        move_mtmr = self.mtml.move_jp(goal)
        move_mtmr.wait()

        self.teleop_ready = True

        print('home complete')

    def release_force(self):
        self.mtml.use_gravity_compensation(True)
        self.mtmr.use_gravity_compensation(True)
        self.mtml.body.servo_cf(numpy.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
        self.mtmr.body.servo_cf(numpy.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))

    def adjust_orientation(self, hrsv_T_mtml, hrsv_T_mtmr):
        mtml_goal = PyKDL.Frame()
        mtmr_goal = PyKDL.Frame()

        mtml_goal.p = self.mtml.setpoint_cp().p
        mtmr_goal.p = self.mtmr.setpoint_cp().p

        mtml_goal.M = PyKDL.Rotation(hrsv_T_mtml[0][0], hrsv_T_mtml[0][1], hrsv_T_mtml[0][2],
                                     hrsv_T_mtml[1][0], hrsv_T_mtml[1][1], hrsv_T_mtml[1][2],
                                     hrsv_T_mtml[2][0], hrsv_T_mtml[2][1], hrsv_T_mtml[2][2])
        mtmr_goal.M = PyKDL.Rotation(hrsv_T_mtmr[0][0], hrsv_T_mtmr[0][1], hrsv_T_mtmr[0][2],
                                     hrsv_T_mtmr[1][0], hrsv_T_mtmr[1][1], hrsv_T_mtmr[1][2],
                                     hrsv_T_mtmr[2][0], hrsv_T_mtmr[2][1], hrsv_T_mtmr[2][2])

        self.mtml.move_cp(mtml_goal).wait()
        self.mtmr.move_cp(mtmr_goal).wait()

    def adjust_orientation_right(self, hrsv_T_mtmr):
        """
        Adjust only the MTMR orientation based on the provided transformation matrix.
        """
        print("[MTM] Adjusting orientation of MTMR...")

        # Preserve current position, update only orientation
        mtmr_goal = PyKDL.Frame()
        mtmr_goal.p = self.mtmr.setpoint_cp().p

        mtmr_goal.M = PyKDL.Rotation(
            hrsv_T_mtmr[0][0], hrsv_T_mtmr[0][1], hrsv_T_mtmr[0][2],
            hrsv_T_mtmr[1][0], hrsv_T_mtmr[1][1], hrsv_T_mtmr[1][2],
            hrsv_T_mtmr[2][0], hrsv_T_mtmr[2][1], hrsv_T_mtmr[2][2]
        )

        self.mtmr.move_cp(mtmr_goal).wait()
        print("[MTM] MTMR orientation updated.")

    def hold_position(self):
        self.mtml.hold()
        self.mtmr.hold()     

    # tests
    def tests(self):
        initial_cartesian_position = PyKDL.Frame()
        initial_cartesian_position.p = self.mtml.setpoint_cp().p
        initial_cartesian_position.M = self.mtml.setpoint_cp().M
        print(initial_cartesian_position)
        goal = PyKDL.Frame()
        goal.p = self.mtml.setpoint_cp().p
        goal.M = self.mtml.setpoint_cp().M

        # motion parameters
        amplitude = 0.05 # 5 cm

        # first motion
        goal.p[0] =  initial_cartesian_position.p[0] - amplitude
        self.mtml.move_cp(goal).wait()
        # second motion
        goal.p[0] =  initial_cartesian_position.p[0] + amplitude
        self.mtml.move_cp(goal).wait()
        # back to initial position
        goal.p[0] =  initial_cartesian_position.p[0]
        self.mtml.move_cp(goal).wait()

        # first motion
        goal.p[1] =  initial_cartesian_position.p[1] - amplitude
        self.mtml.move_cp(goal).wait()
        # second motion
        goal.p[1] =  initial_cartesian_position.p[1] + amplitude
        self.mtml.move_cp(goal).wait()
        # back to initial position
        goal.p[1] =  initial_cartesian_position.p[1]
        self.mtml.move_cp(goal).wait()

        # first motion
        goal.p[2] =  initial_cartesian_position.p[2] - amplitude
        self.mtml.move_cp(goal).wait()
        # second motion
        goal.p[2] =  initial_cartesian_position.p[2] + amplitude
        self.mtml.move_cp(goal).wait()
        # back to initial position
        goal.p[2] =  initial_cartesian_position.p[2]
        self.mtml.move_cp(goal).wait()

    # main method
    def run(self):
        self.home()
        self.tests()


    def move_to_pose(self, pos_l, rotvec_l, pos_r, rotvec_r):
        """
        Move MTML and MTMR to specified poses (position + rotation vector).
        """
        mtml_goal = PyKDL.Frame()
        mtmr_goal = PyKDL.Frame()

        mtml_goal.p = PyKDL.Vector(*pos_l)
        mtmr_goal.p = PyKDL.Vector(*pos_r)

        rotmat_l = Rotation.from_rotvec(rotvec_l).as_matrix()
        rotmat_r = Rotation.from_rotvec(rotvec_r).as_matrix()

        mtml_goal.M = PyKDL.Rotation(
            rotmat_l[0,0], rotmat_l[0,1], rotmat_l[0,2],
            rotmat_l[1,0], rotmat_l[1,1], rotmat_l[1,2],
            rotmat_l[2,0], rotmat_l[2,1], rotmat_l[2,2]
        )

        mtmr_goal.M = PyKDL.Rotation(
            rotmat_r[0,0], rotmat_r[0,1], rotmat_r[0,2],
            rotmat_r[1,0], rotmat_r[1,1], rotmat_r[1,2],
            rotmat_r[2,0], rotmat_r[2,1], rotmat_r[2,2]
        )

        self.mtml.move_cp(mtml_goal).wait()
        self.mtmr.move_cp(mtmr_goal).wait()

    


def main():
    application = MTMManipulator()
    application.run()

if __name__ == '__main__':
    main()
