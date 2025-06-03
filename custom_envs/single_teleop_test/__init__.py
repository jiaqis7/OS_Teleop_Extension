import gymnasium as gym
from .config.base_env_cfg import SingleTeleopBaseEnv
from .config.po_env_cfg import POTeleopEnvCfg
from .config.mtm_env_cfg import MTMTeleopEnvCfg
from .config.playback_env_config import PBEnvCfg
from .config.mtm_po_env_cfg import MTMPOTeleopEnvCfg
from .config.mtml_act_cfg import MTMLACTEnvCfg
from .config.playback_three_arm_env_cfg import PBThreeEnvCfg
from .config.three_act_cfg import ThreeACTEnvCfg

gym.register(
    id="Isaac-MultiArm-dVRK-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": SingleTeleopBaseEnv},
    disable_env_checker=True,
)

gym.register(
    id="Isaac-PO-Teleop-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": POTeleopEnvCfg},
    disable_env_checker=True,
)

gym.register(
    id="Isaac-MTM-Teleop-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": MTMTeleopEnvCfg},
    disable_env_checker=True,
)

gym.register(
    id="Isaac-MTM-Teleop-pb",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": PBEnvCfg},
    disable_env_checker=True,
)

gym.register(
    id="Isaac-MTML-MTMR-PO-Teleop-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": MTMPOTeleopEnvCfg},
    disable_env_checker=True,
)

gym.register(
    id="Isaac-MTM-Teleop-pb-three-arm",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": PBThreeEnvCfg},
    disable_env_checker=True,
)

gym.register(
    id="Isaac-MTML-ACT-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": MTMLACTEnvCfg},
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Three-ACT-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": ThreeACTEnvCfg},
    disable_env_checker=True,
)