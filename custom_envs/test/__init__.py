import gymnasium as gym
from .config.needle.joint_pos_env_cfg import NeedleHandoverEnvCfg  # Import environment config

gym.register(
    id="Isaac-CustomTest-v1",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": NeedleHandoverEnvCfg},
    disable_env_checker=True,
)
