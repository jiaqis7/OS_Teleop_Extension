#!/usr/bin/env python

# Author: Joonwon Kang
# Date: 2025-03-25
# Node to adjust the impedance, position, and orientation of the MTM for teleoperation

import crtk
import dvrk
import numpy
import sys
import PyKDL

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
        goal[-3]=1.55
        self.mtml.move_jp(goal).wait()
        self.mtmr.move_jp(goal).wait()

    def release_force(self):
        self.mtml.use_gravity_compensation(True)
        self.mtmr.use_gravity_compensation(True)
        self.mtml.body.servo_cf(numpy.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
        self.mtmr.body.servo_cf(numpy.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))

    def prepare_teleop(self):
        self.home()
        self.release_force()

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

        # print('a force in body frame will be applied (direction depends on wrist orientation)')
        # self.coag.wait(value = 0)
        # self.arm.body_set_cf_orientation_absolute(False)
        # self.arm.body.servo_cf(numpy.array([0.0, 0.0, -3.0, 0.0, 0.0, 0.0]))

        # print('a force in world frame will be applied (fixed direction)')
        # self.coag.wait(value = 0)
        # self.arm.body_set_cf_orientation_absolute(True)
        # self.arm.body.servo_cf(numpy.array([0.0, 0.0, -3.0, 0.0, 0.0, 0.0]))

        # print('orientation will be locked')
        # self.coag.wait(value = 0)
        # self.arm.lock_orientation_as_is()

        # print('force will be removed')
        # self.coag.wait(value = 0)
        # self.arm.body.servo_cf(numpy.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))

        # print('orientation will be unlocked')
        # self.coag.wait(value = 0)
        # self.arm.unlock_orientation()

        # print('press and release coag to end')
        # self.coag.wait(value = 0)

    # main method
    def run(self):
        self.home()
        self.tests()


def main():
    application = MTMManipulator()
    application.run()

if __name__ == '__main__':
    main()
