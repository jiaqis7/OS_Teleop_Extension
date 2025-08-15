# Copyright (c) 2022-2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

import builtins

if hasattr(builtins, "ISAAC_LAUNCHED_FROM_TERMINAL") and builtins.ISAAC_LAUNCHED_FROM_TERMINAL is False:
    # ISAAC_LAUNCHED_FROM_TERMINAL is set to False by SimulationApp, so this will be triggered by standalone Python
    # workflows only
    from .base_isaac_benchmark import BaseIsaacBenchmark
if (hasattr(builtins, "ISAAC_LAUNCHED_FROM_TERMINAL") and builtins.ISAAC_LAUNCHED_FROM_TERMINAL is True) or not hasattr(
    builtins, "ISAAC_LAUNCHED_FROM_TERMINAL"
):
    # This will be triggered if running from an async workflow (non-standalone)
    from .base_isaac_benchmark_async import BaseIsaacBenchmarkAsync
