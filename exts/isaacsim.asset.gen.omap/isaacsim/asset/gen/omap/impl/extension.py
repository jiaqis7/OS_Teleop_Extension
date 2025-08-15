# Copyright (c) 2018-2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
import omni.ext
import omni.kit.commands

from ..bindings import _omap


class Extension(omni.ext.IExt):
    def on_startup(self, ext_id: str):
        self._interface = _omap.acquire_omap_interface()

    def on_shutdown(self):
        _omap.release_omap_interface(self._interface)
