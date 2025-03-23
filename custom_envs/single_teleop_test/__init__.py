import gymnasium as gym
from .config.base_env_cfg import SingleTeleopBaseEnv
from .config.po_env_cfg import POTeleopEnvCfg
from .config.mtm_env_cfg import MTMTeleopEnvCfg

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
