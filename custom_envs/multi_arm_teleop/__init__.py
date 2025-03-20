import gymnasium as gym
from .config.mtm_po_env_cfg import MTMPOTeleopEnvCfg

gym.register(
    id="Isaac-MTM-PO-Teleop-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": MTMPOTeleopEnvCfg},
    disable_env_checker=True,
)
