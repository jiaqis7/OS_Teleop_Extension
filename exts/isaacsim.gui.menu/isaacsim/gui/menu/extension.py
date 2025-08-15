# Copyright (c) 2018-2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

import gc

import omni.ext
import omni.kit.commands
from omni.kit.menu.utils import MenuItemDescription, add_menu_items, remove_menu_items

from .create_menu import CreateMenuExtension
from .edit_menu.edit_menu import EditMenuExtension
from .file_menu.file_menu import FileMenuExtension
from .fixme_menu import FixmeMenuExtension
from .help_menu import HelpMenuExtension
from .hooks_menu import HookMenuHandler
from .layout_menu import LayoutMenuExtension
from .tools_menu import ToolsMenuExtension
from .utilities_menu import UtilitiesMenuExtension
from .window_menu import WindowMenuExtension

# TODO: correct colors


class Extension(omni.ext.IExt):
    def on_startup(self, ext_id: str):

        # kit menus
        self.__hook_menu = HookMenuHandler()
        self.__file_menu = FileMenuExtension(ext_id)
        self.__edit_menu = EditMenuExtension(ext_id)
        self.__create_menu = CreateMenuExtension(ext_id)
        self.__window_menu = WindowMenuExtension(ext_id)
        self.__tools_menu = ToolsMenuExtension(ext_id)
        self.__utilities_menu = UtilitiesMenuExtension(ext_id)
        self.__layout_menu = LayoutMenuExtension(ext_id)
        self.__help_menu = HelpMenuExtension(ext_id)
        self.__fixme_menu = FixmeMenuExtension(ext_id)

        # update order
        menu_self = omni.kit.menu.utils.get_instance()
        menu_defs, menu_order, menu_delegates = menu_self.get_menu_data()
        menu_order["File"] = -10
        menu_order["Edit"] = -9
        menu_order["Create"] = -8
        menu_order["Window"] = -7
        menu_order["Tools"] = 4
        menu_order["Utilities"] = 5
        menu_order["Layouts"] = 6
        menu_order["Help"] = 99

    def on_shutdown(self):
        # remove_menu_items(self._menu_items, "Create")
        del self.__hook_menu
        del self.__file_menu
        del self.__edit_menu
        del self.__create_menu
        del self.__window_menu
        del self.__tools_menu
        del self.__utilities_menu
        del self.__layout_menu
        del self.__help_menu
        del self.__fixme_menu

        gc.collect()


# FIXME
# headings containing submenus don't appear
