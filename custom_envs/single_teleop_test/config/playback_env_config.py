from omni.isaac.lab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from omni.isaac.lab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from omni.isaac.lab.utils import configclass
import orbit.surgical.tasks.surgical.reach_dual.mdp as mdp
from . import base_env_cfg

##
# Pre-defined configs
##
from custom_assets.psm_fast import PSM_FAST_CFG

# Now using as controlling robot tip with absolute and gripper with joint angle
# For teleoperation using MTM
@configclass
class PBEnvCfg(base_env_cfg.SingleTeleopBaseEnv):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set PSM as robot
        # We switch here to a stiffer PD controller for IK tracking to be better.
        self.scene.robot_1 = PSM_FAST_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot_1")
        self.scene.robot_1.init_state.pos = (0.145, 0.0, 0.145)
        self.scene.robot_1.init_state.rot = (0.9659, 0.0, 0.2588, 0.0)
        self.scene.robot_2 = PSM_FAST_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot_2")
        self.scene.robot_2.init_state.pos = (-0.145, 0.0, 0.145)
        self.scene.robot_2.init_state.rot = (0.9659, 0.0, -0.2588, 0.0)
        self.scene.robot_3 = None
        # Set actions for the specific robot type (PSM)
        
        self.actions.arm_1_action = mdp.JointPositionActionCfg(
            asset_name="robot_1",
            joint_names=[
                "psm_yaw_joint",
                "psm_pitch_end_joint",
                "psm_main_insertion_joint",
                "psm_tool_roll_joint",
                "psm_tool_pitch_joint",
                "psm_tool_yaw_joint",
            ],
            scale=1.0,
            use_default_offset=True,
        )

        self.actions.arm_2_action = mdp.JointPositionActionCfg(
            asset_name="robot_2",
            joint_names=[
                "psm_yaw_joint",
                "psm_pitch_end_joint",
                "psm_main_insertion_joint",
                "psm_tool_roll_joint",
                "psm_tool_pitch_joint",
                "psm_tool_yaw_joint",
            ],
            scale=1.0,
            use_default_offset=True,
        )

        self.actions.gripper_1_action = mdp.JointPositionActionCfg(
            asset_name="robot_1",
            joint_names=[
                "psm_tool_gripper1_joint",
                "psm_tool_gripper2_joint",
            ],
            scale=1.0,
            use_default_offset=False,
        )
        self.actions.gripper_2_action = mdp.JointPositionActionCfg(
            asset_name="robot_2",
            joint_names=[
                "psm_tool_gripper1_joint",
                "psm_tool_gripper2_joint",
            ],
            scale=1.0,
            use_default_offset=False,
        )




