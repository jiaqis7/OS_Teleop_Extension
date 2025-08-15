import omni.kit.menu.utils
from omni.kit.menu.utils import LayoutSourceSearch, MenuItemDescription, MenuLayout, add_menu_items


class ToolsMenuExtension:
    def __init__(self, ext_id):
        self.__menu_layout = [
            MenuLayout.Menu(
                "Tools",
                [
                    MenuLayout.SubMenu(
                        "Animation",
                        [
                            MenuLayout.Item("People Simulation", source="Window/People Simulation"),
                            MenuLayout.Item("Retargeting", source="Window/Animation/Retargeting"),  # ??
                            MenuLayout.Item(
                                "Simplify Animation Curve", source="Tools/Animation/Curve Processing/Simplify Curve"
                            ),
                            MenuLayout.Item("Stage Recorder", source="Window/Animation/Stage Recorder"),
                            MenuLayout.Item(
                                "Timesamples to Curves", source="Tools/Animation/Convert/USD TimeSample to Curves"
                            ),
                        ],
                    ),
                    MenuLayout.SubMenu(
                        "Physics",
                        [
                            MenuLayout.Item(
                                "Collision Groups Filtering Matrix",
                                source="Window/Physics/Collision Groups Filtering Matrix",
                            ),
                            MenuLayout.Item("Physics API Editor"),
                            MenuLayout.Item("Physics Inspector"),
                            MenuLayout.Item("PhysX Character Controller", source="Window/Physics/Character Controller"),
                        ],
                    ),
                    MenuLayout.SubMenu(
                        "Replicator",
                        [
                            MenuLayout.Item(name="Replicator YAML", source="Replicator/Replicator YAML"),
                            MenuLayout.Item(
                                name="Semantics Schema Editor", source="Replicator/Semantics Schema Editor"
                            ),
                            MenuLayout.Item(
                                name="Synthetic Data Recorder", source="Replicator/Synthetic Data Recorder"
                            ),
                            MenuLayout.Seperator("Orchestrator"),
                            MenuLayout.Item(name="Preview", source="Replicator/Preview"),
                            MenuLayout.Item(name="Start", source="Replicator/Start"),
                            MenuLayout.Item(name="Step", source="Replicator/Step"),
                            MenuLayout.Seperator("Agent SDG"),
                            MenuLayout.Item("Agent SDG"),
                            MenuLayout.Item("Calibration Tool"),
                            MenuLayout.Item("Command Injection"),
                            MenuLayout.Item("Custom Command"),
                        ],
                    ),
                    MenuLayout.SubMenu(
                        "Robotics",
                        [
                            MenuLayout.SubMenu(
                                "Asset Editors",
                                [
                                    MenuLayout.Item("Gain Tuner"),
                                    MenuLayout.Item("Mesh Merge Tool"),
                                    MenuLayout.Item("Robot Assembler"),
                                ],
                            ),
                            MenuLayout.SubMenu(
                                "OmniGraph Controllers",
                                [
                                    MenuLayout.Item(name="Differential Controller"),
                                    MenuLayout.Item(name="Joint Position"),
                                    MenuLayout.Item(name="Joint Velocity"),
                                    MenuLayout.Item(name="Open Loop Gripper Controller"),
                                    MenuLayout.Item(name="Surface Gripper"),
                                ],
                            ),
                            MenuLayout.Item("ROS 2 OmniGraphs"),
                            MenuLayout.Seperator("Navigation"),
                            MenuLayout.Item("Block World Generator"),
                            MenuLayout.Item("Occupancy Map"),
                            MenuLayout.Seperator("Manipulation"),
                            MenuLayout.Item("Grasp Editor"),
                            MenuLayout.Item("Lula Robot Description Editor"),
                            MenuLayout.Seperator("Sensors"),
                            MenuLayout.Item("Camera Inspector"),
                        ],
                    ),
                    MenuLayout.SubMenu(
                        "USD",
                        [
                            MenuLayout.Item("USD Paths", source="Window/USD Paths"),
                            MenuLayout.Item("Variant Presenter", source="Tools/Variants/Variant Presenter"),
                            MenuLayout.Item("Variant Editor", source="Tools/Variants/Variant Editor"),
                        ],
                    ),
                    MenuLayout.Seperator("Toolbars"),
                    MenuLayout.Item("Main ToolBar", source="Window/Main ToolBar"),
                    MenuLayout.Item("Physics Toolbar", "Window/Physics/Physics Authoring Toolbar"),
                ],
            )
        ]
        omni.kit.menu.utils.add_layout(self.__menu_layout)

        physics_inspector = MenuItemDescription(
            name="Physics Inspector",
            onclick_action=("omni.physx.supportui", "show_physics_inspector"),
        )

        add_menu_items([physics_inspector], "Tools")

    def __del__(self):
        omni.kit.menu.utils.remove_layout(self.__menu_layout)
