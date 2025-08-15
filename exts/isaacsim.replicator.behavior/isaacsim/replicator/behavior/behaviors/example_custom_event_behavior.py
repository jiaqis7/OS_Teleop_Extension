# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.

# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import carb
import carb.events
import omni.kit.app
import omni.kit.window.property
from isaacsim.replicator.behavior.global_variables import EXPOSED_ATTR_NS, EXTENSION_NAME
from isaacsim.replicator.behavior.utils.behavior_utils import (
    check_if_exposed_variables_should_be_removed,
    create_exposed_variables,
    get_exposed_variable,
    remove_exposed_variables,
)
from omni.kit.scripting import BehaviorScript
from pxr import Sdf, Usd


class ExampleCustomEventBehavior(BehaviorScript):
    BEHAVIOR_NS = "exampleCustomEventBehavior"
    EVENT_NAME_IN = f"{EXTENSION_NAME}.{BEHAVIOR_NS}.in"
    EVENT_NAME_OUT = f"{EXTENSION_NAME}.{BEHAVIOR_NS}.out"
    ALLOWED_FUNCTIONS = ["setup", "update", "reset"]

    VARIABLES_TO_EXPOSE = [
        {
            "attr_name": "includeChildren",
            "attr_type": Sdf.ValueTypeNames.Bool,
            "default_value": True,
            "doc": "Include valid prim children to the behavior.",
        },
        {
            "attr_name": "event:input",
            "attr_type": Sdf.ValueTypeNames.String,
            "default_value": f"{EVENT_NAME_IN}",
            "doc": (
                "Event to subscribe to for controlling the behavior.\n"
                "NOTE: Changing this value will not have any effect since the event subscription is done on init."
            ),
            "lock": True,
        },
        {
            "attr_name": "event:output",
            "attr_type": Sdf.ValueTypeNames.String,
            "default_value": f"{EVENT_NAME_OUT}",
            "doc": "Event name to publish to on behavior update.",
        },
    ]

    def on_init(self):
        """Called when the script is assigned to a prim."""
        self._event_type_out = carb.events.type_from_string(self.EVENT_NAME_OUT)
        self._valid_prims = []

        # App event stream, used to listen to incoming control events, and to publish the state of the behavior script
        self._event_stream = omni.kit.app.get_app().get_message_bus_event_stream()

        # Subscribe to the event stream to listen for incoming control events
        self._event_sub = self._event_stream.create_subscription_to_pop_by_type(
            carb.events.type_from_string(self.EVENT_NAME_IN), self._on_event
        )

        # Expose the variables as USD attributes
        create_exposed_variables(self.prim, EXPOSED_ATTR_NS, self.BEHAVIOR_NS, self.VARIABLES_TO_EXPOSE)

        # Refresh the property windows to show the exposed variables
        omni.kit.window.property.get_window().request_rebuild()

    def on_destroy(self):
        """Called when the script is unassigned from a prim."""
        # Unsubscribe from the event stream
        self._reset()

        self._event_sub.unsubscribe()
        self._event_sub = None

        # Exposed variables should be removed if the script is no longer assigned to the prim
        if check_if_exposed_variables_should_be_removed(self.prim, __file__):
            remove_exposed_variables(self.prim, EXPOSED_ATTR_NS, self.BEHAVIOR_NS, self.VARIABLES_TO_EXPOSE)
            omni.kit.window.property.get_window().request_rebuild()

    def setup(self):
        print(f"[ExampleCustomEventBehavior][{self.prim_path}] setup()")
        self._setup()
        self._event_stream.push(
            self._event_type_out, payload={"prim_path": str(self.prim_path), "function_name": "setup"}
        )

    def update(self):
        print(f"[ExampleCustomEventBehavior][{self.prim_path}] update()")
        self._apply_behavior()
        self._event_stream.push(
            self._event_type_out, payload={"prim_path": str(self.prim_path), "function_name": "update"}
        )

    def reset(self):
        print(f"[ExampleCustomEventBehavior][{self.prim_path}] reset()")
        self._reset()
        self._event_stream.push(
            self._event_type_out, payload={"prim_path": str(self.prim_path), "function_name": "reset"}
        )

    def _on_event(self, event: carb.events.IEvent):
        payload_dict = event.payload.get_dict()

        # If the prim_path is provided in the payload, check if it matches the prim_path of this script
        payload_prim_path = payload_dict.get("prim_path")
        if payload_prim_path and payload_prim_path != self.prim_path:
            return

        # Check if the function_name is valid
        function_name = payload_dict.get("function_name")
        if function_name in self.ALLOWED_FUNCTIONS:
            getattr(self, function_name)()

    def _setup(self):
        # Fetch the exposed attributes
        self._include_children = self._get_exposed_variable("includeChildren")
        publish_event_name = self._get_exposed_variable("event:output")
        self._event_type_out = carb.events.type_from_string(publish_event_name)

        # Get the prims to apply the behavior to
        if self._include_children:
            self._valid_prims = [prim for prim in Usd.PrimRange(self.prim) if prim.IsValid()]
        elif self.prim.IsValid():
            self._valid_prims = [self.prim]
        else:
            self._valid_prims = []
            carb.log_warn(f"[{self.prim_path}] No valid prims found.")

    def _reset(self):
        self._valid_prims.clear()

    def _apply_behavior(self):
        if not self._valid_prims:
            print(f"[ExampleCustomEventBehavior][{self.prim_path}] No valid prims found.")
            return
        for prim in self._valid_prims:
            print(f"[ExampleCustomEventBehavior][{self.prim_path}] Applying behavior to prim {prim.GetPath()}")

    def _get_exposed_variable(self, attr_name):
        full_attr_name = f"{EXPOSED_ATTR_NS}:{self.BEHAVIOR_NS}:{attr_name}"
        return get_exposed_variable(self.prim, full_attr_name)
