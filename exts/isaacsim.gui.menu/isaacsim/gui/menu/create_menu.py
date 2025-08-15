import asyncio
import weakref
from functools import partial
from pathlib import Path

import carb
import omni.kit.menu.utils
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.gui.components.menu import make_menu_item_description
from isaacsim.storage.native.nucleus import get_assets_root_path
from omni.kit.menu.utils import MenuItemDescription, MenuLayout, add_menu_items


class CreateMenuExtension:
    def __init__(self, ext_id):
        self.__menu_layout = [
            MenuLayout.Menu(
                "Create",
                [
                    MenuLayout.Item("Mesh"),
                    MenuLayout.Item("Shape"),
                    MenuLayout.Item("Lights", source="Create/Light"),
                    MenuLayout.Item("Audio"),
                    MenuLayout.Item("Camera"),
                    MenuLayout.Item("Scope"),
                    MenuLayout.Item("Xform", source="Create/Xform"),
                    MenuLayout.Item("Materials", source="Create/Material"),
                    MenuLayout.Item("Graphs", source="Create/Visual Scripting"),
                    MenuLayout.Item("Arbitrary Output Variables (AOV)", source="Create/AOV"),
                    MenuLayout.Item("Physics", source="Create/Physics"),
                    MenuLayout.Seperator("Assets"),
                    MenuLayout.SubMenu(
                        "Robots",
                        [
                            MenuLayout.Item("Asset Browser"),
                            MenuLayout.Seperator("Examples"),
                        ],
                    ),
                    MenuLayout.SubMenu(
                        "Environments",
                        [
                            MenuLayout.Item(name="Asset Browser"),
                            MenuLayout.Seperator("Examples"),
                        ],
                    ),
                    MenuLayout.SubMenu(
                        "Sensors",
                        [
                            MenuLayout.Item(name="Asset Browser"),
                            MenuLayout.Seperator("Generic Sensors"),
                        ],
                    ),
                    # MenuLayout.Item("ROS 2 Assets", source="Create/ROS 2 Assets"),
                    MenuLayout.SubMenu(
                        "ROS 2 Assets",
                        [
                            MenuLayout.Item("Asset Browser", source="Create/ROS 2 Assets/Asset Browser"),
                            MenuLayout.Seperator("Examples"),
                            MenuLayout.Item("Room", source="Create/ROS 2 Assets/Room"),
                            MenuLayout.Item("Room 2", source="Create/ROS 2 Assets/Room 2"),
                        ],
                    ),
                ],
            )
        ]

        omni.kit.menu.utils.add_layout(self.__menu_layout)

        ## Example Robots
        robot_sub_menu = [
            MenuItemDescription(
                name="Ant",
                onclick_fn=lambda a=weakref.proxy(self): a.create_asset(
                    "/Isaac/Robots/Ant/ant_instanceable.usd", "/Ant"
                ),
            ),
            MenuItemDescription(
                name="Boston Dynamics Spot (Quadruped)",
                onclick_fn=lambda a=weakref.proxy(self): a.create_asset(
                    "/Isaac/Robots/BostonDynamics/spot/spot.usd", "/spot"
                ),
            ),
            MenuItemDescription(
                name="Franka Emika Panda Arm",
                onclick_fn=lambda a=weakref.proxy(self): a.create_asset("/Isaac/Robots/Franka/franka.usd", "/Franka"),
            ),
            MenuItemDescription(
                name="Humanoid",
                onclick_fn=lambda a=weakref.proxy(self): a.create_asset(
                    "/Isaac/Robots/Humanoid/humanoid_instanceable.usd", "/Humanoid"
                ),
            ),
            MenuItemDescription(
                name="Nova Carter with Sensors",
                onclick_fn=lambda a=weakref.proxy(self): a.create_asset(
                    "/Isaac/Robots/NVIDIA/Carter/nova_carter/nova_carter.usd", "/Nova_Carter"
                ),
            ),
            MenuItemDescription(
                name="Quadcopter",
                onclick_fn=lambda a=weakref.proxy(self): a.create_asset(
                    "/Isaac/Robots/Quadcopter/quadcopter.usd", "/Quadcopter"
                ),
            ),
            MenuItemDescription(
                name="Asset Browser", onclick_action=("isaacsim.asset.browser", "open_isaac_sim_asset_browser")
            ),
        ]

        icon_dir = omni.kit.app.get_app().get_extension_manager().get_extension_path_by_module(__name__)

        robot_icon_path = str(Path(icon_dir).joinpath("data/robot.svg"))

        robot_menu = [MenuItemDescription(name="Robots", glyph=robot_icon_path, sub_menu=robot_sub_menu)]
        add_menu_items(robot_menu, "Create")

        ## Environments
        environment_sub_menu = [
            MenuItemDescription(
                name="Black Grid",
                onclick_fn=lambda a=weakref.proxy(self): a.create_asset(
                    "/Isaac/Environments/Grid/gridroom_black.usd", "/BlackGrid"
                ),
            ),
            MenuItemDescription(
                name="Flat Grid",
                onclick_fn=lambda a=weakref.proxy(self): a.create_asset(
                    "/Isaac/Environments/Grid/default_environment.usd", "/FlatGrid"
                ),
            ),
            MenuItemDescription(
                name="Simple Room",
                onclick_fn=lambda a=weakref.proxy(self): a.create_asset(
                    "/Isaac/Environments/Simple_Room/simple_room.usd", "/SimpleRoom", [3.15, 3.15, 2.0], [0, 0, 0]
                ),
            ),
            MenuItemDescription(
                name="Asset Browser", onclick_action=("isaacsim.asset.browser", "open_isaac_sim_asset_browser")
            ),
        ]

        environment_icon_path = str(Path(icon_dir).joinpath("data/environment.svg"))
        environment_menu = [
            MenuItemDescription(name="Environments", glyph=environment_icon_path, sub_menu=environment_sub_menu)
        ]
        add_menu_items(environment_menu, "Create")

        ## Sensor
        sensor_sub_menu = [
            MenuItemDescription(
                name="Asset Browser", onclick_action=("isaacsim.asset.browser", "open_isaac_sim_asset_browser")
            ),
        ]
        sensor_icon_path = str(Path(icon_dir).joinpath("data/sensor.svg"))
        sensor_menu = [MenuItemDescription(name="Sensors", glyph=sensor_icon_path, sub_menu=sensor_sub_menu)]
        add_menu_items(sensor_menu, "Create")

        ## add apriltag selection
        action_registry = omni.kit.actions.core.get_action_registry()
        action_registry.register_action(
            ext_id,
            "isaac_create_apriltag",
            lambda a=weakref.proxy(self): a.create_apriltag(
                "/Isaac/Materials/AprilTag/AprilTag.mdl",
                "AprilTag",
                "/Looks/AprilTag",
                "/Isaac/Materials/AprilTag/Textures/tag36h11.png",
            ),
            display_name="Create AprilTag",
            description="Create a AprilTag",
            tag="Create AprilTag",
        )
        apriltag_icon_path = str(Path(icon_dir).joinpath("data/apriltag.svg"))
        self._menu_items = [
            MenuItemDescription(
                name="April Tags",
                glyph=apriltag_icon_path,
                onclick_action=(ext_id, "isaac_create_apriltag"),
            )
        ]
        add_menu_items(self._menu_items, "Create")

    def __del__(self):
        omni.kit.menu.utils.remove_layout(self.__menu_layout)

    def create_apriltag(self, usd_path, shader_name, stage_path, tag_path):
        from pxr import Sdf

        self._assets_root_path = get_assets_root_path()
        if self._assets_root_path is None:
            carb.log_error("Could not find Isaac Sim assets folder")
            return

        stage = omni.usd.get_context().get_stage()
        stage_path = omni.usd.get_stage_next_free_path(stage, stage_path, False)

        async def create_tag():
            omni.kit.commands.execute(
                "CreateMdlMaterialPrim",
                mtl_url=self._assets_root_path + usd_path,
                mtl_name=shader_name,
                mtl_path=stage_path,
                select_new_prim=True,
            )
            mtl = stage.GetPrimAtPath(stage_path + "/Shader")
            attr = mtl.CreateAttribute("inputs:tag_mosaic", Sdf.ValueTypeNames.Asset)
            attr.Set(Sdf.AssetPath(self._assets_root_path + tag_path))

        asyncio.ensure_future(create_tag())

    def create_asset(self, usd_path, stage_path, camera_position=None, camera_target=None):

        self._assets_root_path = get_assets_root_path()
        if self._assets_root_path is None:
            carb.log_error("Could not find Isaac Sim assets folder")
            return

        path_to = omni.kit.commands.execute(
            "CreateReferenceCommand",
            usd_context=omni.usd.get_context(),
            path_to=stage_path,
            asset_path=self._assets_root_path + usd_path,
            instanceable=False,
        )

        carb.log_info(f"Added reference to {stage_path} at {path_to}")

        if camera_position is not None and camera_target is not None:
            set_camera_view(camera_position, camera_target)

        pass
