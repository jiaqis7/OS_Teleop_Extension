import omni.kit.menu.utils
import omni.ui as ui
from omni.kit.menu.utils import LayoutSourceSearch, MenuAlignment, MenuItemDescription, MenuLayout


class FixmeMenuExtension:
    class MenuDelegate(ui.MenuDelegate):
        def get_menu_alignment(self):
            return MenuAlignment.RIGHT

        # override build fn to not show menu item...
        def build_item(self, item: ui.MenuHelper):
            pass

    def __init__(self, ext_id):
        self._menu_placeholder = [MenuItemDescription(name="FixMe!!!", show_fn=lambda: False)]
        omni.kit.menu.utils.add_menu_items(self._menu_placeholder, "FixMe", delegate=FixmeMenuExtension.MenuDelegate())

        self.__menu_layout = [
            MenuLayout.Menu(
                "FixMe",
                [
                    MenuLayout.Item("Replicator", source="Replicator"),
                    MenuLayout.SubMenu(
                        "Replicator",
                        [
                            # have to remove hidden menu items too...
                            MenuLayout.Item("Capture On Play", source="Replicator/Capture On Play"),
                            MenuLayout.Item(name="Stop", source="Replicator/Stop"),
                            MenuLayout.Item(name="Resume", source="Replicator/Resume"),
                            MenuLayout.Item(name="Pause", source="Replicator/Pause"),
                            MenuLayout.Item(name="Starting...", source="Replicator/Starting..."),
                            MenuLayout.Item(name="Starting...", source="Replicator/Starting..."),
                            MenuLayout.Item(name="Stopping...", source="Replicator/Stopping..."),
                            MenuLayout.Item(name="Stopping...", source="Replicator/Stopping..."),
                            MenuLayout.Item(name="Capture On Play", source="Replicator/Capture On Play"),
                        ],
                    ),
                    MenuLayout.SubMenu(
                        "Profiler",
                        [
                            MenuLayout.Item("Start or Stop", source="Profiler/Start\\Stop"),
                            MenuLayout.Item("Profile Startup (Restart)", source="Profiler/Profile Startup (Restart)"),
                        ],
                    ),
                    MenuLayout.SubMenu(
                        "Help",
                        [
                            MenuLayout.Item("Discover Kit SDK", source="Help/Discover Kit SDK"),
                            MenuLayout.Item("Developers Manual", source="Help/Developers Manual"),
                        ],
                    ),
                    MenuLayout.SubMenu(
                        "Layout",
                        [MenuLayout.Item("Quick Save", remove=True), MenuLayout.Item("Quick Load", remove=True)],
                    ),
                    MenuLayout.Item("Isaac Sim App Selector", source="Help/Isaac Sim App Selector"),
                ],
            ),
        ]
        omni.kit.menu.utils.add_layout(self.__menu_layout)

    def __del__(self):
        omni.kit.menu.utils.remove_layout(self.__menu_layout)
        omni.kit.menu.utils.remove_menu_items(self._menu_placeholder, "FixMe", 9999)
