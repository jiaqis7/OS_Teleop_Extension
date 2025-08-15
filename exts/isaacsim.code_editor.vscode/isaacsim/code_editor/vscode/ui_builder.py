# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
import os
import subprocess
import sys
import weakref

import carb


class UIBuilder:
    """Manage extension UI"""

    def __init__(self, menu_name, menu_item_name, host, port):
        self._menu_items = []

        self._host = host
        self._port = port
        self._menu_name = menu_name
        self._menu_item_name = menu_item_name

        # get application folder
        self._app_folder = carb.settings.get_settings().get_as_string("/app/folder")
        if not self._app_folder:
            self._app_folder = carb.tokens.get_tokens_interface().resolve("${app}")
        self._app_folder = os.path.normpath(os.path.join(self._app_folder, os.pardir))

    def startup(self):
        """Create menu item"""
        try:
            from omni.kit.menu.utils import MenuItemDescription, add_menu_items

            self._menu_items = [
                MenuItemDescription(
                    name=self._menu_item_name,
                    onclick_fn=lambda p=weakref.proxy(self): p._launch(),
                )
            ]
            add_menu_items(self._menu_items, self._menu_name)
        except ImportError:
            pass

    def shutdown(self):
        """Clean up menu item"""
        try:
            from omni.kit.menu.utils import remove_menu_items

            remove_menu_items(self._menu_items, "Window")
        except ImportError:
            pass
        self._menu_items = []

    def _launch(self, *args, **kwargs):
        """Launch a new VS Code window on the application path"""
        command = ["code", "-n", self._app_folder]
        if sys.platform.startswith("linux"):
            command = [" ".join(command)]
        carb.log_info(f"Launching VS Code: {command}")
        result = subprocess.run(command, shell=True, close_fds=True)
        # check process execution
        notification = f"Serving at {self._host}:{self._port}"
        if result.returncode:
            notification += f"\n\nUnable to launch VS Code (error code: {result.returncode})"
            if result.returncode == 1 or result.returncode == 127:
                notification += ".\nMake sure VS Code is installed and accessible on the system via the command 'code'"
            carb.log_warn(notification)
        # show notification in Kit window
        try:
            import omni.kit.notification_manager as notification_manager
        except ImportError:
            pass
        else:
            if result.returncode:
                status = notification_manager.NotificationStatus.WARNING
            else:
                status = notification_manager.NotificationStatus.INFO
            notification_manager.post_notification(
                notification,
                hide_after_timeout=not result.returncode,
                duration=3,
                status=status,
                button_infos=[notification_manager.NotificationButtonInfo("OK", on_complete=None)],
            )
