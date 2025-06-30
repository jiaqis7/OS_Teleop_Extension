## Autonomous-Surgical-Robot-Isaacgym
Project with CHARM and IPRL on the Da Vinci Surgical Robot. The project aims to automate one of the arms as an assistant (using imitation learning) to collaborate with the surgeon. 

## Table of contents
## Requirements
1. dVRK
2. ORBIT-Surgical
   
+) Additional packages to add to your orbitsurgical conda environment (to resolve ROS package and PyKDL issues)
```bash
conda install conda-forge::rospkg
conda install conda-forge::python-orocos-kdl
```

If all requirements were installed properly, you should not have to resolve any additional errors when running scripts.

## Overview
This repo mainly focus on the training & testing in isaacgym simulation environment. There are three functionalities - teleoperation, data collection and model rollout. The instructions have been described in detail below. It is assumed that the commands are executed from the SRC (Stanford Robotics Centre) computer in the Medical Bay connected to the Da Vinci Robot (Si Model).

## File Structure

## Teleoperation

### Step 1: Launch the dvrk console (New Terminal)

```bash
roslaunch teleop arms_real.launch
```

You should see two windows appearing one after another. The first one is an RViz window and the second one is the console (the GUI to control the robot). The `arms_real.launch` launch files will run the `dvrk_console_json` node from the dVRK package and other static coordinate transformations that are required for the teleoperation.

### Step 2: Click the Power On button followed by the Home button in the console

Clicking the `Power On button` turns the LED on the arms to blue. Clicking the `Home` button turns them green and you will notice the MTMs moving towards their home position. Wait for all the arms to turn green, sometimes it takes longer for SUJ to turn green.

### Step 3: Launching the Phantom Omni device (New Terminal)

```bash
roslaunch teleop phantom_real.launch
```

The `phantom_real.launch` file contains the nodes required to simulate the digital twin and publish the pose of the phantom omni's stylus with respect to it's base. You should be a simulated model of the phantom omni in RViz.

Sometimes, this command can throw permission errors (when the phantom omni is re-plugged or the computer is restarted). Run the following command when that happens:

```bash
sudo chmod 777 /dev/ttyACM0
```

and relaunch the `phantom_real.launch` using the command above.

### Step 4: Launching the isaacgym simulation environment (New Terminal)

There are three main control patterns for teleoperation:

1. Two PSMs: Both PSMs in the simulation are controlled by Both MTMs
```bash
cd ~/OS_Teleop_Extension
conda activate orbitsurgical
python script/teleoperation/teleop_mtm.py --enable_cameras
```

2. Two PSMs: The left PSM in the simulation is controlled by the Phantom Omni, while the right PSM in the simulation is controlled by the right MTM
```bash
cd ~/OS_Teleop_Extension
conda activate orbitsurgical
python script/teleoperation/teleop_mtmr_po.py --enable_cameras
```

3. Three PSMs: Both left/right PSMs in the simulation are controlled by the left/right MTMs, while the central PSM in the simulation is controlled by the Phantom Omni
```bash
cd ~/OS_Teleop_Extension
conda activate orbitsurgical
python script/teleoperation/teleop_mtm_po.py --enable_cameras
```

After launching, there would be three windows, the bigger window shows the main view, while there are two small windows that show the view of left camera and right camera (the left camera view is hided behind the right camera view). In order to move these two small windows to the screen of comsol above the MTMs, here are the steps:

1. Right click window of right camera view, click move to external window
2. Right click window of left camera view, click move to external window
3. Click window of left camera view, type `Win + Shift + <` **TWICE**
4. Click window of right camera view, type `Win + Shift + <` **ONCE**

Now both views of left camera and right camera would be on the comsol of MTMs, go and check them!

The pose of MTM would be automatically set to match the orientation of PSM that it controls before clutch, while for Phantom Omni the user needs to hold it at the pose that almost match the orientation of PSM before clutch for better manipulation during teleoperation. To start teleoperation, press and release the `clutch` buttom under the MTMs to start MTM control, press the black button on the Phantom Omni to start PO control. 

## Data Collection (New Terminal)

During the teleoperation, users could launch the function of data collection to record the robot's states and camera views during teleoperation by sending the command
```bash
cd ~/OS_Teleop_Extension
touch log_trigger_demo_1.txt
```
That would create a folder called `demo_1` under `OS_Teleop_Extension` which include a csv file called `teleop_log.csv` that record the robot states, two folders record the images from left camera and right camera, and a json file record the environment related parameters during teleoperation. If not satisfied with this demo_1, just relaunch the script 
```bash
touch log_trigger_demo_1.txt
```
again and the new folder demo_1 would replace the old one. 

## Reset (New Terminal)

After one round of demo, we need to reset the environment to do the next one. By sending the command
```bash
cd ~/OS_Teleop_Extension
touch reset_trigger.txt
```

The pose of PSMs would be set back to the pose when first launching the simulation environment, while the pose of other objects in the environment would be randomly assigned due to the requirement of randomization of task environment. After resetting, both MTMs and Phantom Omni needs to be reclutched in order to restart the teleoperation.

## Replay 

In order to check whether the data in teleop_log.csv is valid for model training, we also add the function to playback the demo recorded by sending the command
```bash
cd ~/OS_Teleop_Extension
conda activate orbitsurgical
python scripts/playback/playback_three_arm.py --enable_cameras --csv_file demo_1/teleop_log.csv
```

Remember to kill the teleoperation environment before launching this replay environment!
The motion of PSMs during the episode recorded would be replayed by reading the robot states data in the csv file. 

## Model Rollout

There are several schemes for the model rollout:
1. Fully Autonomous: All 3 PSMs are controlled by the model
```bash
cd ~/OS_Teleop_Extension
conda activate orbitsurgical
python scripts/act/ACT_Three_Arm.py --enable_cameras --model_control all
```
2. Collaborative: The PSM set in `--model_control` is controlled by the model, while others are controlled by the human
```bash
cd ~/OS_Teleop_Extension
conda activate orbitsurgical
python scripts/act/ACT_Three_Arm.py --enable_cameras --model_control psm3
```
in which `psm3` here could be chosen from `psm1/psm2/psm3/psm12/all/none`. If PSM3 is controlled by human, it's controlled by the Phantom Omni. If PSM1 or PSM2 is controlled by human, it's controlled by MTMs.

After launching the simulation environment, trigger the model control by sending the command
```bash
cd ~/OS_Teleop_Extension
touch model_trigger.txt
```
in a separate terminal. 

In order to record the model process, the user could launch the data collection function in a new terminal by sending the command
```bash
cd ~/OS_Teleop_Extension
touch log_trigger_rollout_1.txt
```
The logic is same as data collection during teleoperation, and it could be replayed if needed by
```bash
cd ~/OS_Teleop_Extension
conda activate orbitsurgical
python scripts/playback/playback_three_arm.py --enable_cameras --csv_file rollout_1/teleop_log.csv
```


