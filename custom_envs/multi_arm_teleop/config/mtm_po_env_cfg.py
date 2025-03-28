# Copyright (c) 2024, The ORBIT-Surgical Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math

from orbit.surgical.assets import ORBITSURGICAL_ASSETS_DATA_DIR

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import RigidObjectCfg
from omni.isaac.lab.assets import AssetBaseCfg
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.sensors import CameraCfg, FrameTransformerCfg
from omni.isaac.lab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from omni.isaac.lab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from omni.isaac.lab.utils import configclass

import custom_envs.multi_arm_teleop.mdp as mdp
from custom_envs.multi_arm_teleop.multi_teleop_env_cfg import MultiTeleopEnvCfg

from omni.isaac.lab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from omni.isaac.lab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg

##
# Pre-defined configs
##
from omni.isaac.lab.markers.config import FRAME_MARKER_CFG  # isort: skip
from orbit.surgical.assets.psm import PSM_CFG  # isort: skip
from orbit.surgical.assets.ecm import ECM_CFG  # isort: skip
from orbit.surgical.assets.psm import PSM_HIGH_PD_CFG  # isort: skip


##
# Environment configuration
##


@configclass
class MTMPOTeleopEnvCfg(MultiTeleopEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # simulation settings
        self.viewer.eye = (0.0, 0.8, 0.4)
        self.viewer.lookat = (0.0, 0.0, 0.05)
        self.scene.replicate_physics = False

        self.scene.table = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Table",
            spawn=sim_utils.UsdFileCfg(
                usd_path=f"{ORBITSURGICAL_ASSETS_DATA_DIR}/Props/Table/table.usd",
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.457)),
        )

        # sensors
        self.scene.camera_left = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot_4/ecm_end_link/camera_left",
            update_period=0.1,
            height=480,
            width=640,
            data_types=["rgb", "distance_to_image_plane", "semantic_segmentation"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(0.0, -2.75e-3, 0.0), rot=(0.7071068, 0.0, 0.0, 0.7071068), convention="ros"
            ),
        )

        self.scene.camera_right = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot_4/ecm_end_link/camera_right",
            update_period=0.1,
            height=480,
            width=640,
            data_types=["rgb", "distance_to_image_plane", "semantic_segmentation"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
            ),
            offset=CameraCfg.OffsetCfg(pos=(0.0, 2.75e-3, 0.0), rot=(0.7071068, 0.0, 0.0, 0.7071068), convention="ros"),
        )

        self.scene.object1 = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object1",
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, -0.1, 0.015), rot=(1, 0, 0, 0)),
            spawn=UsdFileCfg(
                usd_path=f"{ORBITSURGICAL_ASSETS_DATA_DIR}/Props/Surgical_needle/needle_sdf.usd",
                scale=(0.4, 0.4, 0.4),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=8,
                    max_angular_velocity=200,
                    max_linear_velocity=200,
                    max_depenetration_velocity=1.0,
                    disable_gravity=False,
                ),
            ),
        )

        # Set Peg Block as object
        self.scene.object2 = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object2",
            init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.015, -0.04, 0.025), rot=(1, 0, 0, 0)),
            spawn=UsdFileCfg(
                usd_path=f"{ORBITSURGICAL_ASSETS_DATA_DIR}/Props/Surgical_block/block.usd",
                scale=(0.01, 0.01, 0.02),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=8,
                    max_angular_velocity=200,
                    max_linear_velocity=200,
                    max_depenetration_velocity=1.0,
                    disable_gravity=False,
                ),
            ),
        )

        # Set Peg Block as object
        self.scene.object3 = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object3",
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.015, -0.04, 0.025), rot=(1, 0, 0, 0)),
            spawn=UsdFileCfg(
                usd_path=f"{ORBITSURGICAL_ASSETS_DATA_DIR}/Props/Surgical_block/block.usd",
                scale=(0.01, 0.01, 0.02),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=8,
                    max_angular_velocity=200,
                    max_linear_velocity=200,
                    max_depenetration_velocity=1.0,
                    disable_gravity=False,
                ),
            ),
        )

        # switch robot to PSM
        self.scene.robot_1 = PSM_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot_1")
        self.scene.robot_1.init_state.pos = (0.08, 0.0, 0.12)
        self.scene.robot_1.init_state.rot = (0.9848078, 0.0, 0.1736482, 0.0)
        self.scene.robot_2 = PSM_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot_2")
        self.scene.robot_2.init_state.pos = (-0.08, 0.0, 0.12)
        self.scene.robot_2.init_state.rot = (0.9848078, 0.0, -0.1736482, 0.0)
        self.scene.robot_3 = PSM_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot_3")
        self.scene.robot_3.init_state.pos = (0.0, -0.08, 0.12)
        self.scene.robot_3.init_state.rot = (1.0, 0.0, 0.0, 0.0)
        # Set actions for the specific robot type (PSM)
        self.scene.robot_4 = ECM_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot_4")
        self.scene.robot_4.init_state.pos = (0.0, 0.45, 0.55)
        self.scene.robot_4.init_state.rot = (0.9238795, -0.3826834, 0, 0)

        self.actions.arm_1_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot_1",
            joint_names=[
                "psm_yaw_joint",
                "psm_pitch_end_joint",
                "psm_main_insertion_joint",
                "psm_tool_roll_joint",
                "psm_tool_pitch_joint",
                "psm_tool_yaw_joint",
            ],
            body_name="psm_tool_tip_link",
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
            scale=1.0,
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
        self.actions.arm_2_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot_2",
            joint_names=[
                "psm_yaw_joint",
                "psm_pitch_end_joint",
                "psm_main_insertion_joint",
                "psm_tool_roll_joint",
                "psm_tool_pitch_joint",
                "psm_tool_yaw_joint",
            ],
            body_name="psm_tool_tip_link",
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
            scale=1.0,
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
        # Expect it to be ran by PO
        self.actions.arm_3_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot_3",
            joint_names=[
                "psm_yaw_joint",
                "psm_pitch_end_joint",
                "psm_main_insertion_joint",
                "psm_tool_roll_joint",
                "psm_tool_pitch_joint",
                "psm_tool_yaw_joint",
            ],
            body_name="psm_tool_tip_link",
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
            scale=1.0,
        )
        self.actions.gripper_3_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot_3",
            joint_names=["psm_tool_gripper.*_joint"],
            open_command_expr={"psm_tool_gripper1_joint": -0.55, "psm_tool_gripper2_joint": 0.55},
            close_command_expr={"psm_tool_gripper1_joint": -0.1, "psm_tool_gripper2_joint": 0.1},
        )

        self.events.reset_robot_1_joints = EventTerm(
            func=mdp.reset_joints_by_scale,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot_1"),
                "position_range": (0.01, 0.1),
                "velocity_range": (0.0, 0.0),
            },
        )

        self.events.reset_robot_2_joints = EventTerm(
            func=mdp.reset_joints_by_scale,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot_2"),
                "position_range": (0.01, 0.1),
                "velocity_range": (0.0, 0.0),
            },
        )

        self.events.reset_robot_3_joints = EventTerm(
            func=mdp.reset_joints_by_scale,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot_3"),
                "position_range": (0.01, 0.1),
                "velocity_range": (0.0, 0.0),
            },
        )