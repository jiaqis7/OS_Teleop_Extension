# Copyright (c) 2022-2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

import omni.ext


class Extension(omni.ext.IExt):
    """Object that tracks the lifetime of the Python part of the extension loading"""

    def on_startup(self):
        """Set up initial conditions for the Python part of the extension"""
        pass

    def on_shutdown(self):
        """Shutting down this part of the extension prepares it for hot reload"""
        pass
