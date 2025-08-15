# Copyright (c) 2022-2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

from pathlib import Path

from omni.ui import color as cl

CURRENT_PATH = Path(__file__).parent
ICON_PATH = CURRENT_PATH.parent.parent.parent.joinpath("icons")

# Use same context menu style with content browser
cl.context_menu_background = cl.shade(cl("#343432"))
cl.context_menu_separator = cl.shade(0x449E9E9E)
cl.context_menu_text = cl.shade(cl("#9E9E9E"))

CONTEXT_MENU_STYLE = {
    "Menu": {"background_color": cl.context_menu_background_color, "color": cl.context_menu_text, "border_radius": 2},
    "Menu.Item": {"background_color": 0x0, "margin": 0},
    "Separator": {"background_color": 0x0, "color": cl.context_menu_separator},
}


PROPERTY_STYLE = {
    "RadioButton": {"background_color": 0x0, "padding": 0},
    "RadioButton.Image": {"image_url": f"{ICON_PATH}/radio_off.svg"},
    "RadioButton.Image:checked": {"image_url": f"{ICON_PATH}/radio_on.svg"},
}
