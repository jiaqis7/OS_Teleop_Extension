## Requirements
1. dVRK
2. ORBIT-Surgical
   
+) Additional packages to add to your orbitsurgical conda environment (to resolve ROS package and PyKDL issues)
```bash
conda install conda-forge::rospkg
conda install conda-forge::python-orocos-kdl
```

If all requirements were installed properly, you should not have to resolve any additional errors when running scripts.

Test the connection with existing packages by running the line below in the orbitsurgical conda environment.

```bash
python scripts/zero_agent.py --task Isaac-CustomTest-v1
```

It should show a simulation window with two PSMs and one needle without any movements.

## Running Teleoperations
When running custom environments with cameras, you have to add --enable_cameras.

For example, to run the teleoperation script for a PhantomOmni device, use

```bash
python scripts/teleoperation/teleop_po.py --enable_cameras
```
