from orbit.surgical.assets.psm import PSM_HIGH_PD_CFG  # isort: skip

PSM_FAST_CFG = PSM_HIGH_PD_CFG.copy()
PSM_FAST_CFG.actuators["psm"].velocity_limit = 3.5
PSM_FAST_CFG.actuators["psm_tool"].velocity_limit = 0.7
"""
Configuration of dVRK PSM robot arm with higher joint velocity limit"
This configuration is to mitigate unintuitive delay during teleoperation
"""