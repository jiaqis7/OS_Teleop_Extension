import global_cfg
from omni.isaac.lab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from omni.isaac.lab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg, JointPositionActionCfg
from omni.isaac.lab.utils import configclass
import orbit.surgical.tasks.surgical.reach_dual.mdp as mdp
from . import base_env_cfg

import math

from orbit.surgical.assets import ORBITSURGICAL_ASSETS_DATA_DIR

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import RigidObjectCfg, DeformableObjectCfg
from omni.isaac.lab.assets import AssetBaseCfg
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.sensors import CameraCfg, FrameTransformerCfg
from omni.isaac.lab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg, DeformableBodyPropertiesCfg
from omni.isaac.lab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from omni.isaac.lab.sim.spawners.shapes.shapes_cfg import CuboidCfg
from omni.isaac.lab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg, DeformableBodyMaterialCfg
from omni.isaac.lab.sim.spawners.materials.visual_materials_cfg import PreviewSurfaceCfg
from omni.isaac.lab.sim.spawners.materials.visual_materials import spawn_preview_surface
from omni.isaac.lab.sim.spawners.meshes.meshes_cfg import MeshCuboidCfg
from omni.isaac.lab.utils import configclass

##
# Pre-defined configs
##
from custom_assets.psm_fast import PSM_FAST_CFG

# Control mode mapping for cleaner logic
CONTROL_MODE_MAPPING = {
    "psm1": {"PSM1": "model", "PSM2": "human", "PSM3": "human"},
    "psm2": {"PSM1": "human", "PSM2": "model", "PSM3": "human"},
    "psm3": {"PSM1": "human", "PSM2": "human", "PSM3": "model"},
    "psm12": {"PSM1": "model", "PSM2": "model", "PSM3": "human"},
    "all": {"PSM1": "model", "PSM2": "model", "PSM3": "model"},
    "none": {"PSM1": "human", "PSM2": "human", "PSM3": "human"}
}

# Now using as controlling robot tip with absolute and gripper with joint angle
# For teleoperation using MTM
@configclass
class ThreeACTEnvCfg(base_env_cfg.SingleTeleopBaseEnv):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        model_control = global_cfg.model_control

        # Set PSM as robot
        # We switch here to a stiffer PD controller for IK tracking to be better.
        self.scene.robot_1 = PSM_FAST_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot_1")
        self.scene.robot_1.init_state.pos = (0.12, 0.0, 0.101)
        self.scene.robot_1.init_state.rot = (0.9659, 0.0, 0.2588, 0.0)

        self.scene.robot_2 = PSM_FAST_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot_2")
        self.scene.robot_2.init_state.pos = (-0.12, 0.0, 0.101)
        self.scene.robot_2.init_state.rot = (0.9659, 0.0, -0.2588, 0.0)

        self.scene.robot_3 = PSM_FAST_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot_3")
        self.scene.robot_3.init_state.pos = (0.0, -0.034, 0.093)
        self.scene.robot_3.init_state.rot = (0.99144486, 0.13052619, 0.0, 0.0)

        # Set cube objects (unchanged)
        self.scene.cube_rigid_1 = RigidObjectCfg(
            prim_path="/World/Objects/CubeRigid",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(0.0, 0.0, 0.0),
                rot=(1.0, 0.0, 0.0, 0.0),
            ),
            spawn=sim_utils.CuboidCfg(
                size=(0.02, 0.004, 0.004),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    linear_damping=0.05,
                    angular_damping=0.05,
                    solver_position_iteration_count=30,
                    solver_velocity_iteration_count=10,
                ),
                mass_props=sim_utils.MassPropertiesCfg(
                    mass=0.03,
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    contact_offset=0.005,
                    rest_offset=-0.001,
                ),
                visual_material=PreviewSurfaceCfg(
                    diffuse_color=(0.8, 0.0, 0.0),
                    roughness=0.4,
                    metallic=0.0,
                    opacity=1.0,
                ),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=3.0,
                    dynamic_friction=3.0,
                    restitution=0.0,
                ),
            )
        )

        self.scene.cube_rigid_2 = RigidObjectCfg(
            prim_path="/World/Objects/CubeRigid2",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(0.05, 0.01, 0.0),
                rot=(1.0, 0.0, 0.0, 0.0),
            ),
            spawn=sim_utils.CuboidCfg(
                size=(0.02, 0.004, 0.004),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    linear_damping=0.05,
                    angular_damping=0.05,
                    solver_position_iteration_count=30,
                    solver_velocity_iteration_count=10,
                ),
                mass_props=sim_utils.MassPropertiesCfg(
                    mass=0.03,
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    contact_offset=0.005,
                    rest_offset=-0.001,
                ),
                visual_material=PreviewSurfaceCfg(
                    diffuse_color=(0.0, 0.8, 0.0),
                    roughness=0.4,
                    metallic=0.0,
                    opacity=1.0,
                ),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=3.0,
                    dynamic_friction=3.0,
                    restitution=0.0,
                ),
            )
        )

        self.scene.cube_rigid_3 = RigidObjectCfg(
            prim_path="/World/Objects/CubeRigid3",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(-0.05, 0.01, 0.0),
                rot=(1.0, 0.0, 0.0, 0.0),
            ),
            spawn=sim_utils.CuboidCfg(
                size=(0.02, 0.004, 0.004),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    linear_damping=0.05,
                    angular_damping=0.05,
                    solver_position_iteration_count=30,
                    solver_velocity_iteration_count=10,
                ),
                mass_props=sim_utils.MassPropertiesCfg(
                    mass=0.03,
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    contact_offset=0.005,
                    rest_offset=-0.001,
                ),
                visual_material=PreviewSurfaceCfg(
                    diffuse_color=(0.0, 0.0, 0.8),
                    roughness=0.4,
                    metallic=0.0,
                    opacity=1.0,
                ),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=3.0,
                    dynamic_friction=3.0,
                    restitution=0.0,
                ),
            )
        )

        self.scene.cube_rigid_4 = RigidObjectCfg(
            prim_path="/World/Objects/CubeRigid4",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(0.0, 0.055, 0.0),
                rot=(1.0, 0.0, 0.0, 0.0),
            ),
            spawn=sim_utils.CuboidCfg(
                size=(0.05, 0.05, 0.01),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    linear_damping=0.05,
                    angular_damping=0.05,
                    solver_position_iteration_count=30,
                    solver_velocity_iteration_count=10,
                ),
                mass_props=sim_utils.MassPropertiesCfg(
                    mass=0.5,
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    contact_offset=0.005,
                    rest_offset=-0.001,
                ),
                visual_material=PreviewSurfaceCfg(
                    diffuse_color=(1.0, 1.0, 1.0),
                    roughness=0.4,
                    metallic=0.0,
                    opacity=1.0,
                ),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=3.0,
                    dynamic_friction=3.0,
                    restitution=0.0,
                ),
            )
        )

        # PSM joint names
        psm_joint_names = [
            "psm_yaw_joint",
            "psm_pitch_end_joint",
            "psm_main_insertion_joint",
            "psm_tool_roll_joint",
            "psm_tool_pitch_joint",
            "psm_tool_yaw_joint",
        ]

        # Get control configuration
        if model_control in CONTROL_MODE_MAPPING:
            control_config = CONTROL_MODE_MAPPING[model_control]
        else:
            print(f"[WARNING] Unknown model_control: {model_control}, using 'none'")
            control_config = CONTROL_MODE_MAPPING["none"]

        # Configure PSM1 action based on control mode
        if control_config["PSM1"] == "model":
            # Model control: 6D joint-space action
            self.actions.arm_1_action = JointPositionActionCfg(
                asset_name="robot_1",
                joint_names=psm_joint_names,
                scale=1.0,
                use_default_offset=False,
            )
        else:
            # Human control: 7D task-space action
            self.actions.arm_1_action = DifferentialInverseKinematicsActionCfg(
                asset_name="robot_1",
                joint_names=psm_joint_names,
                body_name="psm_tool_tip_link",
                controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
            )

        # Configure PSM2 action based on control mode
        if control_config["PSM2"] == "model":
            # Model control: 6D joint-space action
            self.actions.arm_2_action = JointPositionActionCfg(
                asset_name="robot_2",
                joint_names=psm_joint_names,
                scale=1.0,
                use_default_offset=False,
            )
        else:
            # Human control: 7D task-space action
            self.actions.arm_2_action = DifferentialInverseKinematicsActionCfg(
                asset_name="robot_2",
                joint_names=psm_joint_names,
                body_name="psm_tool_tip_link",
                controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
            )

        # Configure PSM3 action based on control mode
        if control_config["PSM3"] == "model":
            # Model control: 6D joint-space action
            self.actions.arm_3_action = JointPositionActionCfg(
                asset_name="robot_3",
                joint_names=psm_joint_names,
                scale=1.0,
                use_default_offset=False,
            )
        else:
            # Human control: 7D task-space action
            self.actions.arm_3_action = DifferentialInverseKinematicsActionCfg(
                asset_name="robot_3",
                joint_names=psm_joint_names,
                body_name="psm_tool_tip_link",
                controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
            )

        # Gripper actions are always joint position control (2D each)
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

        self.actions.gripper_3_action = mdp.JointPositionActionCfg(
            asset_name="robot_3",
            joint_names=[
                "psm_tool_gripper1_joint",
                "psm_tool_gripper2_joint",
            ],
            scale=1.0,
            use_default_offset=False,
        )

        # Print action configuration for debugging
        total_dim = 0
        for psm_name, control_mode in control_config.items():
            if control_mode == "model":
                total_dim += 8  # 6 joints + 2 gripper
                print(f"[ENV_CONFIG] {psm_name}: Model control (6D joint + 2D gripper = 8D)")
            else:
                total_dim += 9  # 7 pose + 2 gripper  
                print(f"[ENV_CONFIG] {psm_name}: Human control (7D pose + 2D gripper = 9D)")
        
        print(f"[ENV_CONFIG] Total expected action dimension: {total_dim}D")