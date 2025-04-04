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
python scripts/example/zero_agent.py --task Isaac-CustomTest-v1
```

It should show a simulation window with two PSMs and one needle without any movements.

## Running Teleoperations
All teleoperation environments and interfaces were developed to match hand-eye coordination. Therefore, they use additional cameras in the scene, and it is required to add --enable_cameras when running teleoperation environments.
### MTM Teleoperation
If you are using the real MTMs for teleoperation, run
```bash
python scripts/teleoperation/teleop_mtm.py --enable_cameras
```
To terminate teleoperation, press MONO button or simply use keyboard interrupt by Ctrl + C

By default, it uses teleoperation scaling of 0.4. You can change it by adding the argument. For example, if you want to run the teleoperation with the scale of 1.0, run
```bash
python scripts/teleoperation/teleop_mtm.py --scale 1.0 --enable_cameras
```

#### Using Simulated MTM Inputs
It is also available to mimic the output from MTMs using the dvrk_model ROS package, through
```bash
roslaunch dvrk_model surgeon_console.launch
```
To output the clutch signal, you can use 
```bash
#clutch released
rostopic pub /console/clutch sensor_msgs/Joy '{header: {stamp: {secs: 0, nsecs: 0}, frame_id: ""}, axes: [0.0, 0.0], buttons: [0]}'
#clutch pressed
rostopic pub /console/clutch sensor_msgs/Joy '{header: {stamp: {secs: 0, nsecs: 0}, frame_id: ""}, axes: [0.0, 0.0], buttons: [1]}' 
```

In this case, you have to run
```bash
python scripts/teleoperation/teleop_mtm.py --is_simulated True --enable_cameras
```

#### Enable Logging
To enable logging, run 
```bash
python scripts/teleoperation/teleop_mtm.py --enable_logging True --enable_cameras
```

### PhantomOmni Teleoperation
To run the teleoperation script for a PhantomOmni device, use

```bash
python scripts/teleoperation/teleop_po.py --enable_cameras
```

### MTM + PhantomOmni Teleoperation
To run the script for the teleoperation using both MTM and PhantomOmni simultaneously, run

```bash
python scripts/teleoperation/teleop_mtm_po.py --enable_cameras
```

