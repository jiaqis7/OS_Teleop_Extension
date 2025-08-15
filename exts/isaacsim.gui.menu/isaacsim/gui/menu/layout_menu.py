import omni.kit.menu.utils
from omni.kit.menu.utils import LayoutSourceSearch, MenuItemDescription, MenuLayout


class LayoutMenuExtension:
    def __init__(self, ext_id):
        self._menu_placeholder = [MenuItemDescription(name="placeholder", show_fn=lambda: False)]
        omni.kit.menu.utils.add_menu_items(self._menu_placeholder, "Layouts")

        self.__menu_layout = [
            MenuLayout.Menu(
                "Layouts",
                [
                    MenuLayout.Seperator("UI Controls"),
                    MenuLayout.Item("UI Toggle Visibility", source="Window/UI Toggle Visibility"),
                    MenuLayout.Item("Fullscreen Mode", source="Window/Fullscreen Mode"),
                    MenuLayout.Seperator("Templates"),
                    MenuLayout.Item("Default Layout"),
                    MenuLayout.Item("Visual Scripting"),
                    MenuLayout.Seperator("Save/Load"),
                    MenuLayout.Item("Save Layout", source="Window/Layout/Save Layout..."),
                    MenuLayout.Item("Load Layout", source="Window/Layout/Load Layout..."),
                    MenuLayout.Seperator(),
                    MenuLayout.Item("Quick Save"),
                    MenuLayout.Item("Quick Load"),
                ],
            )
        ]
        omni.kit.menu.utils.add_layout(self.__menu_layout)

    def __del__(self):
        omni.kit.menu.utils.remove_layout(self.__menu_layout)
        omni.kit.menu.utils.remove_menu_items(self._menu_placeholder, "FixMe", 9999)
