# Copyright (c) 2021-2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

from enum import Enum

import carb


class IsaacEvents(Enum):
    PHYSICS_WARMUP = carb.events.type_from_string("isaac.physics_warmup")
    SIMULATION_VIEW_CREATED = carb.events.type_from_string("isaac.simulation_view_created")
    PHYSICS_READY = carb.events.type_from_string("isaac.physics_ready")
    POST_RESET = carb.events.type_from_string("isaac.post_reset")
    PRIM_DELETION = carb.events.type_from_string("isaac.prim_deletion")
    PHYSICS_STEP = carb.events.type_from_string("isaac.physics_step")
    TIMELINE_STOP = carb.events.type_from_string("isaac.timeline_stop")
