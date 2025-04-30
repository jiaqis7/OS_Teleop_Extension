from orbit.surgical.assets.psm import PSM_HIGH_PD_CFG  # isort: skip

PSM_FAST_CFG = PSM_HIGH_PD_CFG.copy()
PSM_FAST_CFG.actuators["psm"].velocity_limit = 3.5
PSM_FAST_CFG.actuators["psm"].stiffness = 900.0
PSM_FAST_CFG.actuators["psm"].damping = 100.0
PSM_FAST_CFG.actuators["psm_tool"].velocity_limit = 0.4
PSM_FAST_CFG.actuators["psm_tool"].effort_limit = 100.0
PSM_FAST_CFG.actuators["psm_tool"].stiffness = 5000.0
PSM_FAST_CFG.actuators["psm_tool"].damping = 60.0
# PSM_FAST_CFG.actuators["psm_tool"].armature = 0.03
# PSM_FAST_CFG.actuators["psm_tool"].friction = 0.03
"""
Configuration of dVRK PSM robot arm with higher joint velocity limit"
This configuration is to mitigate unintuitive delay during teleoperation
"""