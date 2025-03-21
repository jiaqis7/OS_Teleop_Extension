Requirements
1. dVRK
2. ORBIT-Surgical
+) run conda install conda-forge::rospkg if it shows a 'module not found error' for the rospkg

If all requirements were installed properly, there should be no additional errors you have to resolve when running scripts.
Test the connection with existing packages by running the line below in the orbitsurgical conda environment.

python scripts/zero_agent.py --task Isaac-CustomTest-v1

It should show up a simulation window with two PSMs and one needle without any movements.

When running custom environments with cameras, you have to add --enable_cameras.
For example, to run the teleoperation script using PhantomOmni device, use

python scripts/teleoperation/teleop_po.py --enable_cameras
