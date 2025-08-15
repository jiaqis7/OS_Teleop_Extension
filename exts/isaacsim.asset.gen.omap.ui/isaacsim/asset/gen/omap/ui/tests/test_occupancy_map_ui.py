# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
import asyncio

import numpy as np
import omni.kit.test

# from omni.kit.test_suite.helpers import StageEventHandler
import omni.kit.ui_test as ui_test
import omni.timeline
import omni.ui as ui


class TestOccupancyMapUI(omni.kit.test.AsyncTestCase):
    async def setup(self):
        # wait for material to be preloaded so create menu is complete & menus don't rebuild during tests
        await omni.kit.material.library.get_mdl_list_async()
        await ui_test.human_delay()

        # TODO: get omni.kit.test_suite.
        # self._stage_event_handler = StageEventHandler("omni.kit.stage_templates")

    async def tearDown(self):
        pass

    async def testLoading(self):
        await omni.usd.get_context().new_stage_async()
        menu_widget = ui_test.get_menubar()
        await menu_widget.find_menu("Tools").click()
        await menu_widget.find_menu("Robotics").click()
        await menu_widget.find_menu("Occupancy Map").click()
