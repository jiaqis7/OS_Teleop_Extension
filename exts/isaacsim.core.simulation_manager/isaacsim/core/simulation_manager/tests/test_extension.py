"""
The Kit extension system tests for Python has additional wrapping 
to make test auto-discoverable add support for async/await tests.
The easiest way to set up the test class is to have it derive from
the omni.kit.test.AsyncTestCase class that implements them.

Visit the next link for more details:
  https://docs.omniverse.nvidia.com/kit/docs/kit-manual/latest/guide/testing_exts_python.html
"""

import omni.kit.test
import omni.timeline
from isaacsim.core.simulation_manager import IsaacEvents, SimulationManager
from isaacsim.core.utils.stage import create_new_stage_async, update_stage_async


class TestExtension(omni.kit.test.AsyncTestCase):
    async def setUp(self):
        """Method called to prepare the test fixture"""
        super().setUp()
        # ---------------
        # Do custom setUp
        # ---------------

    async def tearDown(self):
        """Method called immediately after the test method has been called"""
        # ------------------
        # Do custom tearDown
        # ------------------
        super().tearDown()

    # --------------------------------------------------------------------
    async def test_extension(self):
        # Kit extension system test for Python is based on the unittest module.
        # Visit https://docs.python.org/3/library/unittest.html to see the
        # available assert methods to check for and report failures.
        print("Test case: test_extension")
        await create_new_stage_async()
        self._callbacks = []
        self._callbacks.append(
            SimulationManager.register_callback(lambda x: print("working"), event=IsaacEvents.PHYSICS_READY)
        )
        self._callbacks.append(
            SimulationManager.register_callback(lambda x: print("working"), event=IsaacEvents.POST_RESET)
        )
        timeline = omni.timeline.get_timeline_interface()
        timeline.play()
        await update_stage_async()
        await update_stage_async()
        for callback_id in self._callbacks:
            SimulationManager.deregister_callback(callback_id)
