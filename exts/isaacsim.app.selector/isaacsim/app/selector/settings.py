# Copyright (c) 2018-2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
import platform

# this file own the settings Strings so they can easily be shared and access from the other part of the App Selector
SHOW_CONSOLE_SETTING = "/persistent/ext/isaacsim.app.selector/show_console"
PERSISTENT_SELECTOR_SETTING = "/persistent/ext/isaacsim.app.selector/persistent_selector"

APPS_SETTING = "/ext/isaacsim.app.selector/apps"
EXPERIMENTAL_APPS_SETTING = "/ext/isaacsim.app.selector/experimental_apps"
AUTO_START_SETTING = "/persistent/ext/isaacsim.app.selector/auto_start"

DEFAULT_APP_SETTING = "/persistent/ext/isaacsim.app.selector/default_app"
EXTRA_ARGS_SETTING = "/persistent/ext/isaacsim.app.selector/extra_args"
ECO_MODE_SETTING = "/persistent/rtx/ecoMode/enabled"

PERSISTENT_ROS_BRIDGE_SETTING = "/persistent/ext/isaacsim.app.selector/ros_bridge_extension"
if platform.system().lower() == "windows":
    ROS_BRIDGE_EXTENSIONS = ["", "isaacsim.ros2.bridge"]
else:
    ROS_BRIDGE_EXTENSIONS = ["", "isaacsim.ros1.bridge", "isaacsim.ros2.bridge"]

PERSISTENT_ROS_INTERNAL_LIBS_SETTING = "/persistent/ext/isaacsim.app.selector/ros_internal_libs"
