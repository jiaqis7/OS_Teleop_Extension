# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

import asyncio

import omni.kit.test
from isaacsim.core.utils.articulations import (
    find_all_articulation_base_paths,
    move_articulation_root,
    remove_articulation_root,
)
from isaacsim.core.utils.prims import delete_prim, get_prim_at_path
from isaacsim.core.utils.stage import add_reference_to_stage, create_new_stage_async, update_stage_async
from isaacsim.storage.native import get_assets_root_path_async


# Having a test class derived from omni.kit.test.AsyncTestCase declared on the root of module will
# make it auto-discoverable by omni.kit.test
class TestArticulationUtils(omni.kit.test.AsyncTestCase):
    # Before running each test
    async def setUp(self):
        await create_new_stage_async()

    # After running each test
    async def tearDown(self):
        while omni.usd.get_context().get_stage_loading_status()[2] > 0:
            print("tearDown, assets still loading, waiting to finish...")
            await asyncio.sleep(1.0)
        await update_stage_async()

    def assertListsSame(self, l1, l2):
        for item in l1:
            self.assertTrue(item in l2, f"{l1}, {l2}")

        self.assertTrue(len(l2) == len(l1), f"{l1}, {l2}")

    async def test_find_articulation_base_paths(self):
        assets_root_path = await get_assets_root_path_async()
        add_reference_to_stage(assets_root_path + "/Isaac/Robots/UniversalRobots/ur10e/ur10e.usd", "/World/ur10e")

        self.assertListsSame(find_all_articulation_base_paths(), ["/World/ur10e"])

        move_articulation_root(get_prim_at_path("/World/ur10e/root_joint"), get_prim_at_path("/World"))
        self.assertListsSame(find_all_articulation_base_paths(), ["/World"])

        move_articulation_root(get_prim_at_path("/World"), get_prim_at_path("/World/ur10e/root_joint"))

        add_reference_to_stage(assets_root_path + "/Isaac/Robots/UniversalRobots/ur3e/ur3e.usd", "/World/ur3e")

        self.assertListsSame(find_all_articulation_base_paths(), ["/World/ur10e", "/World/ur3e"])

        move_articulation_root(get_prim_at_path("/World/ur10e/root_joint"), get_prim_at_path("/World/ur10e"))

        self.assertListsSame(find_all_articulation_base_paths(), ["/World/ur10e", "/World/ur3e"])

        move_articulation_root(get_prim_at_path("/World/ur10e"), get_prim_at_path("/World"))

        self.assertListsSame(find_all_articulation_base_paths(), ["/World/ur3e"])

        delete_prim("/World/ur10e")

        add_reference_to_stage(
            assets_root_path + "/Isaac/Robots/UniversalRobots/ur10e/ur10e.usd", "/World/ur3e/nested/ur10e"
        )
        remove_articulation_root(get_prim_at_path("/World/ur3e/nested/ur10e/root_joint"))

        self.assertListsSame(find_all_articulation_base_paths(), ["/World/ur3e"])
        await update_stage_async()
