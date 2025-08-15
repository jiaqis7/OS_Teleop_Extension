# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.

# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import omni.ext
import omni.kit.window.property
from isaacsim.replicator.behavior.global_variables import EXPOSED_ATTR_NS

from .exposed_variables_widget import ExposedVariablesPropertyWidget
from .global_variables import WIDGET_NAME, WIDGET_TITLE


class Extension(omni.ext.IExt):
    def __init__(self):
        super().__init__()
        self._registered = False

    def on_startup(self, ext_id):
        self._register_widget()

    def on_shutdown(self):
        if self._registered:
            self._unregister_widget()

    def _register_widget(self):
        property_window = omni.kit.window.property.get_window()
        if property_window:
            property_window.register_widget(
                "prim",
                WIDGET_NAME,
                ExposedVariablesPropertyWidget(title=WIDGET_TITLE, attribute_namespace_filter=[EXPOSED_ATTR_NS]),
            )
            self._registered = True

    def _unregister_widget(self):
        property_window = omni.kit.window.property.get_window()
        if property_window:
            property_window.unregister_widget("prim", WIDGET_NAME)
            self._registered = False
