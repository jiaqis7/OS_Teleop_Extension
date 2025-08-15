# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

import json
import os

import numpy as np
import omni.kit.app
import omni.kit.test
import omni.usd
from isaacsim.storage.native import get_assets_root_path_async
from PIL import Image


class TestBehaviorsSDGScenario(omni.kit.test.AsyncTestCase):
    """
    SDG pipeline test using behavior scripts and comparing to the golden data
    """

    RGB_MEAN_DIFF_TOLERANCE = 100

    async def setUp(self):
        pass

    async def tearDown(self):
        pass

    # Compare PNG files with the specified prefix using mean pixel difference
    def compare_images_with_mean_diff(self, golden_dir, test_dir, prefix, tolerance):
        print(f"Comparing {prefix} images with tolerance: {tolerance}")
        # Get the list of .png files matching the prefix
        golden_files = sorted([f for f in os.listdir(golden_dir) if f.startswith(prefix) and f.endswith(".png")])
        test_files = sorted([f for f in os.listdir(test_dir) if f.startswith(prefix) and f.endswith(".png")])

        # Ensure the file lists match
        self.assertListEqual(
            golden_files,
            test_files,
            f"File names mismatch for {prefix}*.png: {golden_files} in golden_dir vs {test_files} in test_dir.",
        )

        # Iterate over the matching .png files and compare them pixel-wise using mean difference
        for file_name in golden_files:
            golden_file_path = os.path.join(golden_dir, file_name)
            test_file_path = os.path.join(test_dir, file_name)

            # Open images and convert to arrays
            golden_image = Image.open(golden_file_path)
            test_image = Image.open(test_file_path)
            golden_array = np.array(golden_image)
            test_array = np.array(test_image)

            # Ensure shapes match
            self.assertTrue(
                golden_array.shape == test_array.shape,
                f"Image shapes do not match for {file_name}: expected {golden_array.shape}, got {test_array.shape}.",
            )

            # Calculate mean difference
            diff_array = np.abs(golden_array - test_array)
            mean_diff = np.mean(diff_array)
            print(f"\t'{file_name}': mean_diff={mean_diff}")

            self.assertTrue(
                mean_diff < tolerance,
                f"Mean difference for {file_name} is {mean_diff}, which exceeds tolerance {tolerance}.",
            )

    # Compare .json files with the specified prefix
    def compare_json_files(self, golden_dir, test_dir, prefix):
        print(f"Comparing {prefix} json files")
        # Get the list of .json files matching the prefix
        golden_files = sorted([f for f in os.listdir(golden_dir) if f.startswith(prefix) and f.endswith(".json")])
        test_files = sorted([f for f in os.listdir(test_dir) if f.startswith(prefix) and f.endswith(".json")])

        # Ensure the file lists match
        self.assertListEqual(
            golden_files,
            test_files,
            f"File names mismatch for {prefix}*.json: {golden_files} in golden_dir vs {test_files} in test_dir.",
        )

        # Compare the content of each file
        for file_name in golden_files:
            golden_file_path = os.path.join(golden_dir, file_name)
            test_file_path = os.path.join(test_dir, file_name)

            # Load JSON content
            with open(golden_file_path, "r") as golden_file:
                golden_data = json.load(golden_file)

            with open(test_file_path, "r") as test_file:
                test_data = json.load(test_file)

            # Compare the structure and content
            self.assertEqual(
                golden_data, test_data, f"Content mismatch in file {file_name}: {golden_file_path} vs {test_file_path}."
            )

    # Compare .npy files with the specified prefix
    def compare_npy_files(self, golden_dir, test_dir, prefix):
        print(f"Comparing {prefix} npy files")
        # Get the list of .npy files matching the prefix
        golden_files = sorted([f for f in os.listdir(golden_dir) if f.startswith(prefix) and f.endswith(".npy")])
        test_files = sorted([f for f in os.listdir(test_dir) if f.startswith(prefix) and f.endswith(".npy")])

        # Ensure the file lists match
        self.assertListEqual(
            golden_files,
            test_files,
            f"File names mismatch for {prefix}*.npy: {golden_files} in golden_dir vs {test_files} in test_dir.",
        )

        # Compare the content of each file
        for file_name in golden_files:
            golden_file_path = os.path.join(golden_dir, file_name)
            test_file_path = os.path.join(test_dir, file_name)

            # Load numpy data
            golden_data = np.load(golden_file_path)
            test_data = np.load(test_file_path)

            print(f"'{file_name}' shape, dtype:")
            print(f"\tgolden: shape={golden_data.shape}, dtype={golden_data.dtype}")
            print(f"\ttest: shape={test_data.shape}, dtype={test_data.dtype}")

            # Compare the structure and content
            is_array_equal = np.array_equal(golden_data, test_data)
            self.assertTrue(
                is_array_equal,
                f"Content mismatch in file {file_name}: {golden_file_path} vs {test_file_path}.",
            )

    async def setup_and_run_behaviors_sdg(self):
        import asyncio
        import inspect
        import os
        import random

        import omni.kit.app
        import omni.replicator.core as rep
        import omni.timeline
        import omni.usd
        from isaacsim.core.utils.semantics import add_update_semantics, remove_all_semantics
        from isaacsim.replicator.behavior.behaviors import (
            LightRandomizer,
            LocationRandomizer,
            LookAtBehavior,
            RotationRandomizer,
            TextureRandomizer,
            VolumeStackRandomizer,
        )
        from isaacsim.replicator.behavior.global_variables import EXPOSED_ATTR_NS
        from isaacsim.replicator.behavior.utils.behavior_utils import (
            add_behavior_script_with_parameters_async,
            publish_event_and_wait_for_completion_async,
        )
        from isaacsim.storage.native import get_assets_root_path_async
        from pxr import Gf, UsdGeom

        async def setup_and_run_stacking_simulation_async(prim):
            STACK_ASSETS_CSV = (
                "/Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxC_01.usd,"
                "/Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxD_01.usd,"
                "/Isaac/Props/KLT_Bin/small_KLT_visual.usd,"
            )

            # Add the behavior script with custom parameters
            script_path = inspect.getfile(VolumeStackRandomizer)
            parameters = {
                f"{EXPOSED_ATTR_NS}:{VolumeStackRandomizer.BEHAVIOR_NS}:assets:csv": STACK_ASSETS_CSV,
                f"{EXPOSED_ATTR_NS}:{VolumeStackRandomizer.BEHAVIOR_NS}:assets:numRange": Gf.Vec2i(2, 15),
            }
            await add_behavior_script_with_parameters_async(prim, script_path, parameters)

            # Helper function to handle publishing and waiting for events
            async def handle_event(action, expected_state, max_wait):
                return await publish_event_and_wait_for_completion_async(
                    publish_payload={"prim_path": prim.GetPath(), "action": action},
                    expected_payload={"prim_path": prim.GetPath(), "state_name": expected_state},
                    publish_event_name=VolumeStackRandomizer.EVENT_NAME_IN,
                    subscribe_event_name=VolumeStackRandomizer.EVENT_NAME_OUT,
                    max_wait_updates=max_wait,
                )

            # Define and execute the stacking simulation steps
            actions = [("reset", "RESET", 10), ("setup", "SETUP", 500), ("run", "FINISHED", 1500)]
            for action, state, wait in actions:
                print(f"Executing '{action}' and waiting for state '{state}'...")
                if not await handle_event(action, state, wait):
                    print(f"Failed to complete '{action}' with state '{state}'.")
                    return

            print("Stacking simulation finished.")

        async def setup_texture_randomizer_async(prim):
            TEXTURE_ASSETS_CSV = (
                "/Isaac/Materials/Textures/Patterns/nv_bamboo_desktop.jpg,"
                "/Isaac/Materials/Textures/Patterns/nv_wood_boards_brown.jpg,"
                "/Isaac/Materials/Textures/Patterns/nv_wooden_wall.jpg,"
            )

            script_path = inspect.getfile(TextureRandomizer)
            parameters = {
                f"{EXPOSED_ATTR_NS}:{TextureRandomizer.BEHAVIOR_NS}:interval": 5,
                f"{EXPOSED_ATTR_NS}:{TextureRandomizer.BEHAVIOR_NS}:textures:csv": TEXTURE_ASSETS_CSV,
            }
            await add_behavior_script_with_parameters_async(prim, script_path, parameters)

        async def setup_light_behaviors_async(prim):
            # Light randomization
            light_script_path = inspect.getfile(LightRandomizer)
            light_parameters = {
                f"{EXPOSED_ATTR_NS}:{LightRandomizer.BEHAVIOR_NS}:interval": 4,
                f"{EXPOSED_ATTR_NS}:{LightRandomizer.BEHAVIOR_NS}:range:intensity": Gf.Vec2f(20000, 120000),
            }
            await add_behavior_script_with_parameters_async(prim, light_script_path, light_parameters)

            # Location randomization
            location_script_path = inspect.getfile(LocationRandomizer)
            location_parameters = {
                f"{EXPOSED_ATTR_NS}:{LocationRandomizer.BEHAVIOR_NS}:interval": 2,
                f"{EXPOSED_ATTR_NS}:{LocationRandomizer.BEHAVIOR_NS}:range:minPosition": Gf.Vec3d(-1.25, -1.25, 0.0),
                f"{EXPOSED_ATTR_NS}:{LocationRandomizer.BEHAVIOR_NS}:range:maxPosition": Gf.Vec3d(1.25, 1.25, 0.0),
            }
            await add_behavior_script_with_parameters_async(prim, location_script_path, location_parameters)

        async def setup_target_asset_behaviors_async(prim):
            # Rotation randomization with default parameters
            rotation_script_path = inspect.getfile(RotationRandomizer)
            await add_behavior_script_with_parameters_async(prim, rotation_script_path, {})

            # Location randomization
            location_script_path = inspect.getfile(LocationRandomizer)
            location_parameters = {
                f"{EXPOSED_ATTR_NS}:{LocationRandomizer.BEHAVIOR_NS}:interval": 3,
                f"{EXPOSED_ATTR_NS}:{LocationRandomizer.BEHAVIOR_NS}:range:minPosition": Gf.Vec3d(-0.2, -0.2, -0.2),
                f"{EXPOSED_ATTR_NS}:{LocationRandomizer.BEHAVIOR_NS}:range:maxPosition": Gf.Vec3d(0.2, 0.2, 0.2),
            }
            await add_behavior_script_with_parameters_async(prim, location_script_path, location_parameters)

        async def setup_camera_behaviors_async(prim, target_prim_path):
            # Look at behavior following the target asset
            script_path = inspect.getfile(LookAtBehavior)
            parameters = {
                f"{EXPOSED_ATTR_NS}:{LookAtBehavior.BEHAVIOR_NS}:targetPrimPath": target_prim_path,
            }
            await add_behavior_script_with_parameters_async(prim, script_path, parameters)

        async def setup_writer_and_capture_data_async(camera_path, num_captures):
            # Create the writer and the render product
            rp = rep.create.render_product(camera_path, (512, 512))
            writer = rep.writers.get("BasicWriter")
            output_directory = os.getcwd() + "/out_behaviors_sdg"
            print(f"output_directory: {output_directory}")
            writer.initialize(output_dir=output_directory, rgb=True)
            writer.attach(rp)

            # Disable capture on play, data is captured manually using the step function
            rep.orchestrator.set_capture_on_play(False)

            # Start the timeline for the behavior scripts to run
            timeline = omni.timeline.get_timeline_interface()
            timeline.play()
            await omni.kit.app.get_app().next_update_async()

            # Capture frames
            for i in range(num_captures):
                # Advance the app (including the timeline)
                await omni.kit.app.get_app().next_update_async()

                # Capture and write frame
                print(f"Capturing frame {i} at time {timeline.get_current_time():.4f}")
                await rep.orchestrator.step_async(rt_subframes=32, delta_time=0.0, pause_timeline=False)

            # Stop the timeline (and the behavior scripts triggering)
            timeline.stop()

            # Free the renderer resources
            writer.detach()
            rp.destroy()

            # Make sure all the frames are written from the backend queue
            await rep.orchestrator.wait_until_complete_async()

        async def run_example_async():
            STAGE_URL = "/Isaac/Samples/Replicator/Stage/warehouse_pallets_behavior_scripts.usd"
            PALLETS_ROOT_PATH = "/Root/Pallets"
            LIGHTS_ROOT_PATH = "/Root/Lights"
            CAMERA_PATH = "/Root/Camera_01"
            TARGET_ASSET_URL = "/Isaac/Props/YCB/Axis_Aligned/035_power_drill.usd"
            TARGET_ASSET_PATH = "/Root/Target"
            TARGET_ASSET_LABEL = "power_drill"
            TARGET_ASSET_LOCATION = (-1.5, 5.5, 1.5)

            # Open stage
            assets_root_path = await get_assets_root_path_async()
            print(f"Opening stage from {assets_root_path + STAGE_URL}")
            await omni.usd.get_context().open_stage_async(assets_root_path + STAGE_URL)
            stage = omni.usd.get_context().get_stage()

            # Check if all required prims exist in the stage
            pallets_root_prim = stage.GetPrimAtPath(PALLETS_ROOT_PATH)
            lights_root_prim = stage.GetPrimAtPath(LIGHTS_ROOT_PATH)
            camera_prim = stage.GetPrimAtPath(CAMERA_PATH)
            if not all([pallets_root_prim.IsValid(), lights_root_prim.IsValid(), camera_prim.IsValid()]):
                print(f"Not all required prims exist in the stage.")
                return

            # Spawn the target asset at the requested location, label it with the target asset label
            target_prim = stage.DefinePrim(TARGET_ASSET_PATH, "Xform")
            target_prim.GetReferences().AddReference(assets_root_path + TARGET_ASSET_URL)
            if not target_prim.HasAttribute("xformOp:translate"):
                UsdGeom.Xformable(target_prim).AddTranslateOp()
            target_prim.GetAttribute("xformOp:translate").Set(TARGET_ASSET_LOCATION)
            remove_all_semantics(target_prim, recursive=True)
            add_update_semantics(target_prim, TARGET_ASSET_LABEL)

            # Setup and run the stacking simulation before capturing the data
            await setup_and_run_stacking_simulation_async(pallets_root_prim)

            # Setup texture randomizer
            await setup_texture_randomizer_async(pallets_root_prim)

            # Setup the light behaviors
            await setup_light_behaviors_async(lights_root_prim)

            # Setup the target asset behaviors
            await setup_target_asset_behaviors_async(target_prim)

            # Setup the camera behaviors
            await setup_camera_behaviors_async(camera_prim, str(target_prim.GetPath()))

            # Setup the writer and capture the data, behavior scripts are triggered by running the timeline
            await setup_writer_and_capture_data_async(camera_path=camera_prim.GetPath(), num_captures=6)

        random.seed(10)
        rep.set_global_seed(10)

        await run_example_async()

    async def test_behavior_sdg_pipeline_warehouse(self):
        print(f"Starting the behaviors SDG pipeline")
        await self.setup_and_run_behaviors_sdg()

        print(f"Comparing the results against the golden images")
        golden_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", "golden", "out_behaviors_sdg")
        test_dir = os.path.join(os.getcwd(), "out_behaviors_sdg")

        # Compare the rgb data
        self.compare_images_with_mean_diff(golden_dir, test_dir, "rgb", self.RGB_MEAN_DIFF_TOLERANCE)

        ## Compare the bounding box 3d data
        # self.compare_json_files(golden_dir, test_dir, "bounding_box_3d_labels")
        # self.compare_json_files(golden_dir, test_dir, "bounding_box_3d_paths")
        # self.compare_npy_files(golden_dir, test_dir, "bounding_box_3d")
